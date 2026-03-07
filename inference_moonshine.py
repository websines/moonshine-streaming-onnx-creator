"""
Real-time streaming ASR CLI using Moonshine v2 ONNX INT8 models.

Uses continuous audio accumulation with incremental encoding — the encoder
output is cached and only new audio is encoded each update, keeping latency
constant regardless of total buffer length. The decoder re-decodes from
scratch using the full concatenated encoder output.

Usage:
    python inference_moonshine.py [--model-dir moonshine_streaming_medium]

Speaks into your mic, transcribes in real-time.  Ctrl+C to stop.
"""

import argparse
import os
import sys
import time
import resource
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
FRAME_SAMPLES = 80          # 5ms @ 16kHz — encoder frame size
BOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
MAX_DECODE_TOKENS = 256     # safety limit per transcription
BLOCK_MS = 100              # audio block size in ms
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)

# Streaming defaults
UPDATE_INTERVAL = 0.5       # re-run inference every 0.5s
SILENCE_SEC = 1.5           # finalize line after 1.5s of silence
SPEECH_THRESHOLD = 0.01     # RMS energy threshold for speech
MAX_BUFFER_SEC = 10.0       # max audio buffer before force-flush (keeps RAM bounded)
MIN_AUDIO_SEC = 0.3         # minimum audio to bother processing

# Incremental encoding
OVERLAP_FRAMES = 12         # frames of overlap for sliding window context (~240ms)


# ── Model wrapper ────────────────────────────────────────────────────────────

class MoonshineASR:
    def __init__(self, model_dir: str, use_int8: bool = True):
        self.model_dir = Path(model_dir)
        suffix = "_int8" if use_int8 else ""

        t0 = time.perf_counter()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(
            str(self.model_dir / f"encoder_model{suffix}.onnx"),
            sess_options=opts, providers=providers,
        )
        self.decoder = ort.InferenceSession(
            str(self.model_dir / f"decoder_model{suffix}.onnx"),
            sess_options=opts, providers=providers,
        )
        self.decoder_past = ort.InferenceSession(
            str(self.model_dir / f"decoder_with_past_model{suffix}.onnx"),
            sess_options=opts, providers=providers,
        )
        self.load_time = time.perf_counter() - t0

        tok_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_path))

        # Build KV cache name mappings
        dec_out_names = [o.name for o in self.decoder.get_outputs()][1:]
        dec_past_in_names = [
            inp.name for inp in self.decoder_past.get_inputs()
            if inp.name not in ("decoder_input_ids", "encoder_hidden_states")
        ]
        dec_past_in_set = {n: n for n in dec_past_in_names}

        self._kv_out_to_in = {}
        for out_name in dec_out_names:
            past_name = out_name.replace("present_", "past_", 1)
            if past_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = past_name
            elif out_name + "_orig" in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name + "_orig"
            elif out_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name

        dec_past_out_names = [o.name for o in self.decoder_past.get_outputs()][1:]
        self._kv_past_out_to_in = {}
        for out_name in dec_past_out_names:
            past_name = out_name.replace("present_", "past_", 1)
            if past_name in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = past_name
            elif out_name + "_orig" in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = out_name + "_orig"
            elif out_name in dec_past_in_set:
                self._kv_past_out_to_in[out_name] = out_name

        # Moonshine streaming: 16kHz input, 50Hz encoder output = 320 samples/frame
        self._samples_per_frame = 320.0
        self._overlap_samples = int(OVERLAP_FRAMES * self._samples_per_frame)

        # Incremental encoding state
        self._cached_enc = None       # (1, T_cached, hidden)
        self._n_samples_encoded = 0   # how many audio samples the cache covers

    def _encode_raw(self, audio: np.ndarray) -> np.ndarray:
        """Run encoder on audio array. No caching."""
        orig_len = len(audio)
        remainder = len(audio) % FRAME_SAMPLES
        if remainder:
            audio = np.pad(audio, (0, FRAME_SAMPLES - remainder))
        inp = audio[np.newaxis, :].astype(np.float32)
        mask = np.zeros((1, len(audio)), dtype=np.int64)
        mask[0, :orig_len] = 1
        (enc_out,) = self.encoder.run(None, {
            "input_values": inp,
            "attention_mask": mask,
        })
        return enc_out

    def encode_incremental(self, audio: np.ndarray) -> np.ndarray:
        """
        Incrementally encode audio, reusing cached encoder output for
        previously-seen audio. Only the new portion (+ overlap for
        sliding window context) is encoded.
        """
        n_total = len(audio)

        # First call or after reset: encode everything
        if self._cached_enc is None:
            enc_out = self._encode_raw(audio)
            self._cached_enc = enc_out
            self._n_samples_encoded = n_total
            return enc_out

        n_new = n_total - self._n_samples_encoded
        if n_new <= 0:
            return self._cached_enc  # no new audio

        # Encode new audio with overlap for sliding window context
        chunk_start = max(0, self._n_samples_encoded - self._overlap_samples)
        chunk = audio[chunk_start:]
        chunk_enc = self._encode_raw(chunk)  # (1, T_chunk, hidden)

        # Frames to keep from cache (everything before the overlap region)
        cached_frames = round(chunk_start / self._samples_per_frame)

        if cached_frames > 0 and cached_frames < self._cached_enc.shape[1]:
            enc_out = np.concatenate([
                self._cached_enc[:, :cached_frames, :],
                chunk_enc
            ], axis=1)
        else:
            enc_out = chunk_enc

        self._cached_enc = enc_out
        self._n_samples_encoded = n_total
        return enc_out

    def reset_encoder_cache(self):
        """Clear encoder cache (call when starting a new utterance)."""
        self._cached_enc = None
        self._n_samples_encoded = 0

    def decode_first(self, enc_out: np.ndarray):
        bos = np.array([[BOS_TOKEN]], dtype=np.int64)
        outs = self.decoder.run(None, {
            "decoder_input_ids": bos,
            "encoder_hidden_states": enc_out,
        })
        logits = outs[0]
        dec_out_names = [o.name for o in self.decoder.get_outputs()]
        kv_dict = {}
        for out_name, tensor in zip(dec_out_names[1:], outs[1:]):
            if out_name in self._kv_out_to_in:
                kv_dict[self._kv_out_to_in[out_name]] = tensor
        token_id = int(np.argmax(logits[0, -1, :]))
        return token_id, kv_dict

    def decode_next(self, token_id: int, enc_out: np.ndarray, kv_dict: dict):
        inputs = {
            "decoder_input_ids": np.array([[token_id]], dtype=np.int64),
            "encoder_hidden_states": enc_out,
        }
        inputs.update(kv_dict)

        dec_past_out_names = [o.name for o in self.decoder_past.get_outputs()]
        outs = self.decoder_past.run(None, inputs)
        logits = outs[0]

        new_kv = {}
        for out_name, tensor in zip(dec_past_out_names[1:], outs[1:]):
            if out_name in self._kv_past_out_to_in:
                new_kv[self._kv_past_out_to_in[out_name]] = tensor
        next_id = int(np.argmax(logits[0, -1, :]))
        return next_id, new_kv

    def transcribe(self, audio: np.ndarray, incremental: bool = False):
        """
        Transcribe audio → (text, stats).
        If incremental=True, reuse cached encoder output for previously-seen audio.
        """
        t_enc_start = time.perf_counter()
        if incremental:
            enc_out = self.encode_incremental(audio)
        else:
            enc_out = self._encode_raw(audio)
        t_enc = time.perf_counter() - t_enc_start

        t_dec_start = time.perf_counter()
        token_id, past_kvs = self.decode_first(enc_out)
        token_ids = [token_id]

        while token_id != EOS_TOKEN and len(token_ids) < MAX_DECODE_TOKENS:
            token_id, past_kvs = self.decode_next(token_id, enc_out, past_kvs)
            token_ids.append(token_id)

        t_dec = time.perf_counter() - t_dec_start
        text = self.tokenizer.decode(token_ids)

        stats = {
            "encode_ms": t_enc * 1000,
            "decode_ms": t_dec * 1000,
            "total_ms": (t_enc + t_dec) * 1000,
            "n_tokens": len(token_ids),
            "tokens_per_sec": len(token_ids) / t_dec if t_dec > 0 else 0,
            "audio_sec": len(audio) / SAMPLE_RATE,
            "rtf": (t_enc + t_dec) / (len(audio) / SAMPLE_RATE),
        }
        return text, stats


# ── Utilities ────────────────────────────────────────────────────────────────

def get_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / 1e6
    return ru.ru_maxrss / 1e3


def rms_energy(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Moonshine v2 Streaming ASR — real-time mic transcription")
    parser.add_argument("--model-dir", default="moonshine_streaming_medium", help="ONNX model directory")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 models instead of INT8")
    parser.add_argument("--update-interval", type=float, default=UPDATE_INTERVAL,
                        help="Seconds between re-inference updates (default: 0.5)")
    parser.add_argument("--silence-sec", type=float, default=SILENCE_SEC,
                        help="Seconds of silence to finalize a line (default: 1.5)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    use_int8 = not args.fp32
    suffix = "_int8" if use_int8 else ""
    model_dir = Path(args.model_dir)

    # ── Header ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Moonshine v2 Streaming ASR  (ONNX INT8)")
    print("=" * 60)

    enc_mb = model_size_mb(model_dir / f"encoder_model{suffix}.onnx")
    dec_mb = model_size_mb(model_dir / f"decoder_model{suffix}.onnx")
    dec_past_mb = model_size_mb(model_dir / f"decoder_with_past_model{suffix}.onnx")
    print(f"  Models:  encoder {enc_mb:.0f} MB  |  decoder {dec_mb:.0f} MB  |  decoder_past {dec_past_mb:.0f} MB")
    print(f"  Total:   {enc_mb + dec_mb + dec_past_mb:.0f} MB on disk")

    rss_before = get_rss_mb()
    print(f"\n  Loading models...", end="", flush=True)
    asr = MoonshineASR(str(model_dir), use_int8=use_int8)
    rss_after = get_rss_mb()
    print(f" done ({asr.load_time:.1f}s)")
    print(f"  RAM:     {rss_after:.0f} MB  (+{rss_after - rss_before:.0f} MB for models)")
    print(f"  Encoder: {asr._samples_per_frame:.1f} samples/frame "
          f"({1000 * asr._samples_per_frame / SAMPLE_RATE:.1f}ms), "
          f"overlap {OVERLAP_FRAMES} frames ({asr._overlap_samples / SAMPLE_RATE * 1000:.0f}ms)")

    dev_info = sd.query_devices(args.device or sd.default.device[0], "input")
    print(f"  Mic:     {dev_info['name']}  ({int(dev_info['default_samplerate'])} Hz native)")
    print(f"  Update:  every {args.update_interval}s  |  Silence flush: {args.silence_sec}s")
    print()
    print("  Speak into your microphone. Ctrl+C to stop.")
    print("-" * 60)

    # ── Streaming state ─────────────────────────────────────────────────
    audio_buffer = []
    speech_detected = False
    silence_duration = 0.0

    current_text = ""
    latest_stats = None

    total_audio_sec = 0.0
    total_infer_sec = 0.0
    total_tokens = 0
    n_lines = 0

    try:
        term_cols = os.get_terminal_size().columns
    except OSError:
        term_cols = 80
    prev_display_lines = [0]

    def clear_display():
        if prev_display_lines[0] > 0:
            for _ in range(prev_display_lines[0]):
                sys.stdout.write("\033[A\033[K")
            prev_display_lines[0] = 0
        sys.stdout.write("\r\033[K")

    def show_text(prefix, text):
        clear_display()
        line = f"  {prefix} {text}"
        prev_display_lines[0] = max(0, (len(line) - 1) // term_cols)
        sys.stdout.write(line)
        sys.stdout.flush()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio: {status}]", file=sys.stderr)
        audio_buffer.append(indata[:, 0].copy())

    last_update_time = time.perf_counter()
    last_displayed_text = ""
    level_tick = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SAMPLES,
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.05)

                if not audio_buffer:
                    continue
                last_block = audio_buffer[-1]
                dur = sum(len(b) for b in audio_buffer) / SAMPLE_RATE

                # ── VAD ──────────────────────────────────────────────
                last_rms = rms_energy(last_block)
                if last_rms >= SPEECH_THRESHOLD:
                    speech_detected = True
                    silence_duration = 0.0
                elif speech_detected:
                    silence_duration += 0.05

                # ── Display ──────────────────────────────────────────
                level_tick += 1
                if current_text and current_text != last_displayed_text:
                    show_text(">", current_text)
                    last_displayed_text = current_text
                elif level_tick % 10 == 0 and not current_text:
                    bar_len = min(int(last_rms * 200), 40)
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    state = "SPEECH" if speech_detected else "quiet"
                    sys.stdout.write(f"\r\033[K  [{bar}] rms={last_rms:.4f} {state}")
                    sys.stdout.flush()

                # ── Discard background noise ─────────────────────────
                if not speech_detected and dur > 5.0:
                    audio_buffer.clear()
                    silence_duration = 0.0
                    continue

                # ── Finalize on silence or max buffer ────────────────
                should_finalize = (
                    (speech_detected and silence_duration >= args.silence_sec)
                    or dur >= MAX_BUFFER_SEC
                )

                if should_finalize:
                    if dur >= MIN_AUDIO_SEC:
                        audio_snapshot = np.concatenate(audio_buffer).astype(np.float32)
                        text_now, stats_now = asr.transcribe(audio_snapshot, incremental=True)
                        text_now = text_now.strip()

                    if text_now and stats_now:
                        clear_display()
                        print(f"  [{stats_now['audio_sec']:.1f}s | enc {stats_now['encode_ms']:.0f}ms | "
                              f"dec {stats_now['decode_ms']:.0f}ms | "
                              f"{stats_now['n_tokens']} tok | "
                              f"RTF {stats_now['rtf']:.2f}]")
                        print(f"  >> {text_now}")
                        print()
                        total_audio_sec += stats_now["audio_sec"]
                        total_infer_sec += stats_now["total_ms"] / 1000
                        total_tokens += stats_now["n_tokens"]
                        n_lines += 1

                    audio_buffer.clear()
                    current_text = ""
                    latest_stats = None
                    asr.reset_encoder_cache()
                    speech_detected = False
                    silence_duration = 0.0
                    last_displayed_text = ""
                    last_update_time = time.perf_counter()
                    continue

                # ── Periodic inference (incremental) ─────────────────
                elapsed = time.perf_counter() - last_update_time
                if speech_detected and elapsed >= args.update_interval and dur >= MIN_AUDIO_SEC:
                    audio_snapshot = np.concatenate(audio_buffer).astype(np.float32)
                    try:
                        text, stats = asr.transcribe(audio_snapshot, incremental=True)
                        text = text.strip()
                        current_text = text
                        latest_stats = stats
                        last_update_time = time.perf_counter()
                        if text and text != last_displayed_text:
                            show_text(">", text)
                            last_displayed_text = text
                    except Exception as e:
                        print(f"\n  [inference error: {e}]", file=sys.stderr)

    except KeyboardInterrupt:
        pass

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Lines transcribed:   {n_lines}")
    print(f"  Total audio:         {total_audio_sec:.1f}s")
    print(f"  Total inference:     {total_infer_sec:.1f}s")
    if total_audio_sec > 0:
        print(f"  Overall RTF:         {total_infer_sec / total_audio_sec:.2f}x")
    print(f"  Total tokens:        {total_tokens}")
    if total_infer_sec > 0:
        print(f"  Avg tokens/sec:      {total_tokens / total_infer_sec:.1f}")
    print(f"  Peak RAM:            {get_rss_mb():.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
