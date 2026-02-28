"""
Real-time streaming ASR CLI using Moonshine v2 ONNX INT8 models.

Usage:
    python inference_moonshine.py [--model-dir moonshine_streaming_small]

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
MAX_DECODE_TOKENS = 256     # safety limit per chunk
N_LAYERS = 10

# How long to buffer before running inference (seconds)
CHUNK_SECONDS = 2.0
# Minimum RMS energy to consider a chunk as containing speech
SPEECH_THRESHOLD = 0.01
# How many consecutive silent blocks before we consider speech ended
SILENCE_FLUSH_BLOCKS = 5    # 5 × 100ms = 0.5s of silence
# Minimum audio duration (seconds) to bother processing
MIN_AUDIO_SEC = 0.5


# ── Model wrapper ────────────────────────────────────────────────────────────

class MoonshineASR:
    def __init__(self, model_dir: str, use_int8: bool = True):
        self.model_dir = Path(model_dir)
        suffix = "_int8" if use_int8 else ""

        t0 = time.perf_counter()

        # Session options — single thread for predictable latency
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

        # Tokenizer (pure tokenizers lib — no transformers needed)
        tok_path = self.model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_path))

        # Build mapping: decoder output names → decoder_with_past input names
        # The dynamo exporter names cross-attn KV as "present_cross_*_orig" in
        # decoder_with_past inputs, while first decoder outputs them as
        # "present_cross_*". Self-attn uses "present_self_*" → "past_self_*".
        dec_out_names = [o.name for o in self.decoder.get_outputs()][1:]  # skip logits
        dec_past_in_names = [
            inp.name for inp in self.decoder_past.get_inputs()
            if inp.name not in ("decoder_input_ids", "encoder_hidden_states")
        ]
        # Map decoder output → decoder_with_past input by matching KV type/layer/index
        # present_self_key_0 → past_self_key_0
        # present_cross_key_0 → present_cross_key_0_orig
        self._kv_out_to_in = {}
        dec_past_in_set = {n: n for n in dec_past_in_names}
        for out_name in dec_out_names:
            # Try direct past_ substitution (self-attn: present_self_* → past_self_*)
            past_name = out_name.replace("present_", "past_", 1)
            if past_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = past_name
            # Try _orig suffix (cross-attn: present_cross_* → present_cross_*_orig)
            elif out_name + "_orig" in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name + "_orig"
            # Exact match
            elif out_name in dec_past_in_set:
                self._kv_out_to_in[out_name] = out_name

        # Also need mapping for decoder_with_past outputs → its own inputs (for step 2+)
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

    # ── Encoder ──────────────────────────────────────────────────────────

    def encode(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 (samples,) → encoder_hidden_states (1, T, 620)."""
        orig_len = len(audio)
        # Pad to multiple of FRAME_SAMPLES
        remainder = len(audio) % FRAME_SAMPLES
        if remainder:
            audio = np.pad(audio, (0, FRAME_SAMPLES - remainder))
        inp = audio[np.newaxis, :].astype(np.float32)
        # Attention mask: 1 for real audio, 0 for padding
        mask = np.zeros((1, len(audio)), dtype=np.int64)
        mask[0, :orig_len] = 1
        (enc_out,) = self.encoder.run(None, {
            "input_values": inp,
            "attention_mask": mask,
        })
        return enc_out  # (1, enc_seq, 620)

    # ── Decoder (first token) ────────────────────────────────────────────

    def decode_first(self, enc_out: np.ndarray):
        """Returns (token_id, kv_dict mapping decoder_with_past input names → tensors)."""
        bos = np.array([[BOS_TOKEN]], dtype=np.int64)
        outs = self.decoder.run(None, {
            "decoder_input_ids": bos,
            "encoder_hidden_states": enc_out,
        })
        logits = outs[0]           # (1, 1, vocab)
        # Build dict mapping decoder_with_past input names → tensors
        dec_out_names = [o.name for o in self.decoder.get_outputs()]
        kv_dict = {}
        for out_name, tensor in zip(dec_out_names[1:], outs[1:]):
            if out_name in self._kv_out_to_in:
                kv_dict[self._kv_out_to_in[out_name]] = tensor
        token_id = int(np.argmax(logits[0, -1, :]))
        return token_id, kv_dict

    # ── Decoder (subsequent tokens) ──────────────────────────────────────

    def decode_next(self, token_id: int, enc_out: np.ndarray, kv_dict: dict):
        """Returns (next_token_id, updated kv_dict)."""
        inputs = {
            "decoder_input_ids": np.array([[token_id]], dtype=np.int64),
            "encoder_hidden_states": enc_out,
        }
        inputs.update(kv_dict)

        dec_past_out_names = [o.name for o in self.decoder_past.get_outputs()]
        outs = self.decoder_past.run(None, inputs)
        logits = outs[0]

        # Build updated kv_dict from decoder_with_past outputs
        new_kv = {}
        for out_name, tensor in zip(dec_past_out_names[1:], outs[1:]):
            if out_name in self._kv_past_out_to_in:
                new_kv[self._kv_past_out_to_in[out_name]] = tensor
        next_id = int(np.argmax(logits[0, -1, :]))
        return next_id, new_kv

    # ── Full transcribe ──────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, stream_cb=None):
        """
        Transcribe audio array → text.
        stream_cb(token_text): called after each token for real-time display.
        Returns (text, stats_dict).
        """
        t_enc_start = time.perf_counter()
        enc_out = self.encode(audio)
        t_enc = time.perf_counter() - t_enc_start

        t_dec_start = time.perf_counter()
        token_id, past_kvs = self.decode_first(enc_out)
        token_ids = [token_id]

        if stream_cb and token_id != EOS_TOKEN:
            txt = self.tokenizer.decode([token_id])
            stream_cb(txt)

        while token_id != EOS_TOKEN and len(token_ids) < MAX_DECODE_TOKENS:
            token_id, past_kvs = self.decode_next(token_id, enc_out, past_kvs)
            token_ids.append(token_id)
            if stream_cb and token_id != EOS_TOKEN:
                txt = self.tokenizer.decode(token_ids)
                stream_cb(txt)

        t_dec = time.perf_counter() - t_dec_start
        text = self.tokenizer.decode(token_ids)

        stats = {
            "encode_ms": t_enc * 1000,
            "decode_ms": t_dec * 1000,
            "total_ms": (t_enc + t_dec) * 1000,
            "n_tokens": len(token_ids),
            "tokens_per_sec": len(token_ids) / t_dec if t_dec > 0 else 0,
            "audio_sec": len(audio) / SAMPLE_RATE,
            "rtf": (t_enc + t_dec) / (len(audio) / SAMPLE_RATE),  # real-time factor
        }
        return text, stats


# ── Utilities ────────────────────────────────────────────────────────────────

def get_rss_mb():
    """Current RSS in MB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
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
    parser.add_argument("--model-dir", default="moonshine_streaming_small", help="ONNX model directory")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 models instead of INT8")
    parser.add_argument("--chunk-sec", type=float, default=CHUNK_SECONDS, help="Audio chunk length (seconds)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    # Late import so --help is fast
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    use_int8 = not args.fp32
    suffix = "_int8" if use_int8 else ""
    model_dir = Path(args.model_dir)

    # ── Print header ─────────────────────────────────────────────────────
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

    # Audio device info
    dev_info = sd.query_devices(args.device or sd.default.device[0], "input")
    print(f"  Mic:     {dev_info['name']}  ({int(dev_info['default_samplerate'])} Hz native)")
    print(f"  Chunk:   {args.chunk_sec}s  ({int(args.chunk_sec * SAMPLE_RATE)} samples)")
    print()
    print("  Speak into your microphone. Ctrl+C to stop.")
    print("-" * 60)

    # ── Audio capture loop ───────────────────────────────────────────────
    chunk_samples = int(args.chunk_sec * SAMPLE_RATE)
    min_samples = int(MIN_AUDIO_SEC * SAMPLE_RATE)
    audio_buffer = []
    speech_detected = False  # have we seen speech energy in current buffer?
    silence_blocks = 0       # consecutive silent 100ms blocks
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    total_tokens = 0
    n_chunks = 0

    def process_buffer():
        nonlocal audio_buffer, speech_detected, silence_blocks
        nonlocal total_audio_sec, total_infer_sec, total_tokens, n_chunks

        if not audio_buffer:
            return

        audio = np.concatenate(audio_buffer).astype(np.float32)
        audio_buffer.clear()
        speech_detected = False
        silence_blocks = 0

        # Skip if too short or too quiet (no speech energy)
        rms = rms_energy(audio)
        if len(audio) < min_samples or rms < SPEECH_THRESHOLD:
            return

        sys.stdout.write(f"\r\033[K")
        print(f"  [processing {len(audio)/SAMPLE_RATE:.1f}s, rms={rms:.4f}]", flush=True)

        def stream_print(txt):
            sys.stdout.write(f"\r\033[K  > {txt}")
            sys.stdout.flush()

        text, stats = asr.transcribe(audio, stream_cb=stream_print)
        text = text.strip()

        if text:
            sys.stdout.write(f"\r\033[K")
            print(f"  [{stats['audio_sec']:.1f}s | enc {stats['encode_ms']:.0f}ms | "
                  f"dec {stats['decode_ms']:.0f}ms | "
                  f"{stats['n_tokens']} tok | "
                  f"RTF {stats['rtf']:.2f}]")
            print(f"  >> {text}")
            print()
        else:
            sys.stdout.write(f"\r\033[K")

        total_audio_sec += stats["audio_sec"]
        total_infer_sec += stats["total_ms"] / 1000
        total_tokens += stats["n_tokens"]
        n_chunks += 1

    # Callback receives audio from sounddevice
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"  [audio: {status}]", file=sys.stderr)
        audio_buffer.append(indata[:, 0].copy())

    level_tick = [0]

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),   # 100ms blocks
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.1)

                current_len = sum(len(b) for b in audio_buffer)

                # Show audio level periodically
                level_tick[0] += 1
                if level_tick[0] % 5 == 0 and audio_buffer:
                    rms = rms_energy(audio_buffer[-1])
                    bar_len = min(int(rms * 200), 40)
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    state = "SPEECH" if speech_detected else "quiet"
                    sys.stdout.write(f"\r\033[K  [{bar}] rms={rms:.4f} {state}")
                    sys.stdout.flush()

                if not audio_buffer:
                    continue

                # Check latest block for speech energy
                last_rms = rms_energy(audio_buffer[-1])
                if last_rms >= SPEECH_THRESHOLD:
                    speech_detected = True
                    silence_blocks = 0
                else:
                    silence_blocks += 1

                # Process when buffer hits max chunk length
                if current_len >= chunk_samples:
                    process_buffer()
                    continue

                # If we had speech and now silence for a while, flush
                if speech_detected and silence_blocks >= SILENCE_FLUSH_BLOCKS and current_len >= min_samples:
                    process_buffer()
                    continue

                # If no speech detected and buffer is getting large, discard
                if not speech_detected and current_len >= chunk_samples:
                    audio_buffer.clear()
                    silence_blocks = 0

    except KeyboardInterrupt:
        pass

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Session Summary")
    print("=" * 60)
    print(f"  Chunks transcribed:  {n_chunks}")
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
