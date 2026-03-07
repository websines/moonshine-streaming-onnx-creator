"""
Export UsefulSensors/moonshine-streaming-medium (safetensors) to ONNX + INT8.

Produces in ./moonshine_streaming_medium/:
  encoder_model.onnx                — audio → encoder hidden states (fp32)
  decoder_model.onnx                — first decode step, no KV cache (fp32)
  decoder_with_past_model.onnx      — subsequent decode steps (fp32)
  encoder_model_int8.onnx           — INT8 dynamic-quantized encoder
  decoder_model_int8.onnx           — INT8 dynamic-quantized decoder
  decoder_with_past_model_int8.onnx — INT8 dynamic-quantized decoder_with_past

Usage:
    pip install "transformers>=5.2.0" "huggingface_hub>=0.23" torch onnx onnxruntime
    python export_moonshine_streaming_medium.py

INT8 strategy: dynamic quantization (weight-only).
  - No calibration dataset needed.
  - Weights stored as int8; activations remain float32 at runtime.
  - ~4× smaller model files; typically 1.5-2× faster on CPU.

For full static INT8 (weights + activations), enable STATIC_QUANT=True,
provide calibration audio files in CALIBRATION_AUDIO_DIR, and:
    pip install soundfile
"""

import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoProcessor,
    MoonshineStreamingForConditionalGeneration,
)
from types import SimpleNamespace
from transformers.cache_utils import EncoderDecoderCache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

MODEL_ID = "UsefulSensors/moonshine-streaming-medium"
# All output (weights + ONNX files) lands in this one local directory
OUTPUT_DIR      = Path("moonshine_streaming_medium")
MODEL_LOCAL_DIR = OUTPUT_DIR / "weights"
OPSET = 18
DEVICE = "cpu"

# ── Quantization settings ────────────────────────────────────────────────────
# Dynamic (weight-only) INT8: no calibration needed, always safe.
DYNAMIC_QUANT = True

# Static INT8: requires calibration audio. Set to True and point
# CALIBRATION_AUDIO_DIR at a directory of 16-kHz WAV files.
STATIC_QUANT = False
CALIBRATION_AUDIO_DIR = Path("calibration_audio")   # 10-100 files recommended

# Medium model dims (from config.json)
NUM_DECODER_LAYERS = 14
DECODER_HEADS = 10
HEAD_DIM = 64        # 640 hidden / 10 heads
ENC_HIDDEN = 768     # raw encoder output dim (decoder adapter projects this to 640 internally)


# ── Download helpers ─────────────────────────────────────────────────────────

def download_model(model_id: str, local_dir: Path) -> Path:
    """
    Download all model files from HuggingFace Hub into `local_dir`
    (not ~/.cache/huggingface). Returns the local directory path.
    """
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} → {local_dir.resolve()} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        # Don't use the shared HF cache; files go directly into local_dir
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"  Download complete.")
    return local_dir


# ── Wrapper modules ──────────────────────────────────────────────────────────

class EncoderWrapper(torch.nn.Module):
    """Wraps encoder + adapter. Input: raw audio + attention_mask → Output: encoder_hidden_states."""

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        # encoder_outputs is a BaseModelOutput; .last_hidden_state is the adapted features
        encoder_outputs = self.encoder(
            input_values, attention_mask=attention_mask, return_dict=True,
        )
        return encoder_outputs.last_hidden_state  # (B, T_enc, hidden)


class DecoderWrapper(torch.nn.Module):
    """
    First decode step — no past KV cache.
    Returns logits + flat past KV tensors (self + cross).
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.model = model
        self.n_layers = NUM_DECODER_LAYERS

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,       # (B, T_dec)
        encoder_hidden_states: torch.FloatTensor,  # (B, T_enc, hidden)
    ):
        # Run the full model forward with no past KV
        out = self.model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=None,
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits  # (B, T_dec, vocab)
        pkv: EncoderDecoderCache = out.past_key_values

        # Flatten into individual tensors for ONNX
        # DynamicCache v5: layers[i].keys / layers[i].values
        self_cache = pkv.self_attention_cache
        cross_cache = pkv.cross_attention_cache

        flat = [logits]
        for layer in self_cache.layers:
            flat.append(layer.keys)    # (B, heads, T_dec, head_dim)
            flat.append(layer.values)
        for layer in cross_cache.layers:
            flat.append(layer.keys)    # (B, heads, T_enc, head_dim)
            flat.append(layer.values)

        return tuple(flat)


def _layer_from_kv(k, v):
    """Create a DynamicLayer pre-filled with key/value tensors."""
    layer = DynamicLayer()
    layer.keys = k
    layer.values = v
    layer.is_initialized = True
    return layer


class DecoderWithPastWrapper(torch.nn.Module):
    """
    Subsequent decode steps — accepts and returns flat KV tensors.
    Only self-attention KV grows; cross-attention KV is constant.
    """

    def __init__(self, model: MoonshineStreamingForConditionalGeneration):
        super().__init__()
        self.model = model
        self.n_layers = NUM_DECODER_LAYERS

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,        # (B, 1)
        encoder_hidden_states: torch.FloatTensor,   # (B, T_enc, hidden)
        # past self-attention KV — n_layers × key + n_layers × value (interleaved)
        *flat_past,
    ):
        n = self.n_layers
        # Reconstruct EncoderDecoderCache (DynamicCache v5: layers[i].keys/.values)
        self_cache = DynamicCache()
        cross_cache = DynamicCache()

        for i in range(n):
            self_cache.layers.append(_layer_from_kv(flat_past[2 * i], flat_past[2 * i + 1]))
        for i in range(n):
            cross_cache.layers.append(_layer_from_kv(flat_past[2 * n + 2 * i], flat_past[2 * n + 2 * i + 1]))

        pkv = EncoderDecoderCache(self_cache, cross_cache)

        out = self.model(
            encoder_outputs=SimpleNamespace(
                last_hidden_state=encoder_hidden_states,
                attention_mask=None,
                hidden_states=None,
                attentions=None,
            ),
            decoder_input_ids=decoder_input_ids,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits  # (B, 1, vocab)
        new_pkv: EncoderDecoderCache = out.past_key_values

        new_self = new_pkv.self_attention_cache
        new_cross = new_pkv.cross_attention_cache

        flat_out = [logits]
        for layer in new_self.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)
        for layer in new_cross.layers:
            flat_out.append(layer.keys)
            flat_out.append(layer.values)

        return tuple(flat_out)


# ── Dynamic axes helpers ─────────────────────────────────────────────────────

def _kv_self_axes(n_layers, prefix):
    """Dynamic axes for n_layers × (key, value) self-attention tensors."""
    axes = {}
    for i in range(n_layers):
        axes[f"{prefix}_self_key_{i}"]   = {0: "batch", 2: "past_seq"}
        axes[f"{prefix}_self_value_{i}"] = {0: "batch", 2: "past_seq"}
    return axes


def _kv_cross_axes(n_layers, prefix):
    axes = {}
    for i in range(n_layers):
        axes[f"{prefix}_cross_key_{i}"]   = {0: "batch", 2: "enc_seq"}
        axes[f"{prefix}_cross_value_{i}"] = {0: "batch", 2: "enc_seq"}
    return axes


def _kv_output_names(n_layers, self_prefix="present", cross_prefix="present"):
    names = []
    for i in range(n_layers):
        names.append(f"{self_prefix}_self_key_{i}")
        names.append(f"{self_prefix}_self_value_{i}")
    for i in range(n_layers):
        names.append(f"{cross_prefix}_cross_key_{i}")
        names.append(f"{cross_prefix}_cross_value_{i}")
    return names


def _kv_input_names(n_layers):
    names = []
    for i in range(n_layers):
        names.append(f"past_self_key_{i}")
        names.append(f"past_self_value_{i}")
    for i in range(n_layers):
        names.append(f"past_cross_key_{i}")
        names.append(f"past_cross_value_{i}")
    return names


# ── DynamicLayer patch ───────────────────────────────────────────────────────
# DynamicLayer.lazy_initialization creates torch.tensor([]) (1D) then update()
# does cat([1D_empty, 4D_key_states]) which confuses both TorchScript and dynamo.
# Fix: on first call (not initialized), skip the cat and assign directly.

def _patched_dynamic_update(self, key_states, value_states, cache_kwargs=None):
    if not self.is_initialized:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states
        self.values = value_states
        self.is_initialized = True
        return self.keys, self.values
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values

DynamicLayer.update = _patched_dynamic_update


# ── Export functions ─────────────────────────────────────────────────────────
# We use the dynamo ONNX exporter (dynamo=True) for all models. Unlike the
# TorchScript tracer, dynamo properly handles dynamic shapes without baking
# Python .shape[i] values as constants.

from torch.onnx._internal.torchscript_exporter import registration, symbolic_helper

# asinh symbolic only needed for TorchScript encoder export path
def _asinh_symbolic(g, input):
    return g.op("Asinh", input)
torch.onnx.register_custom_op_symbolic("aten::asinh", _asinh_symbolic, opset_version=18)


def export_encoder(model, output_path: Path):
    wrapper = EncoderWrapper(model).eval()
    # Use a larger dummy to avoid edge-case constraints; must be multiple of 80
    dummy_audio = torch.randn(1, 32000)   # 2 seconds
    dummy_mask = torch.ones(1, 32000, dtype=torch.long)

    # Dynamo exporter — needed because attention_mask triggers vmap-based masking
    # code that TorchScript can't trace
    batch = torch.export.Dim("batch", min=1)
    audio_len = torch.export.Dim("audio_length", min=80, max=960000)  # up to 60s

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_audio, dummy_mask),
            str(output_path),
            dynamo=True,
            input_names=["input_values", "attention_mask"],
            output_names=["encoder_hidden_states"],
            dynamic_shapes={
                "input_values": {0: batch, 1: audio_len},
                "attention_mask": {0: batch, 1: audio_len},
            },
        )
    print(f"  encoder → {output_path}")


def export_decoder(model, output_path: Path):
    wrapper = DecoderWrapper(model).eval()
    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, ENC_HIDDEN)

    # Dynamo exporter with dynamic shapes
    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden),
            str(output_path),
            dynamo=True,
            input_names=["decoder_input_ids", "encoder_hidden_states"],
            output_names=["logits"] + _kv_output_names(NUM_DECODER_LAYERS),
            dynamic_shapes={
                "decoder_input_ids": {0: batch},
                "encoder_hidden_states": {0: batch, 1: enc_seq},
            },
        )
    print(f"  decoder → {output_path}")


def export_decoder_with_past(model, output_path: Path):
    wrapper = DecoderWithPastWrapper(model).eval()

    dummy_dec_ids = torch.ones(1, 1, dtype=torch.long)
    dummy_enc_hidden = torch.randn(1, 50, ENC_HIDDEN)

    B, H, HEAD = 1, DECODER_HEADS, HEAD_DIM
    dummy_self_past = [(torch.randn(B, H, 5, HEAD), torch.randn(B, H, 5, HEAD))
                       for _ in range(NUM_DECODER_LAYERS)]
    dummy_cross_past = [(torch.randn(B, H, 50, HEAD), torch.randn(B, H, 50, HEAD))
                        for _ in range(NUM_DECODER_LAYERS)]
    flat_past = []
    for k, v in dummy_self_past:
        flat_past += [k, v]
    for k, v in dummy_cross_past:
        flat_past += [k, v]

    # Dynamic dims for dynamo
    batch = torch.export.Dim("batch", min=1)
    enc_seq = torch.export.Dim("enc_seq", min=1)
    past_seq = torch.export.Dim("past_seq", min=1)

    # Build dynamic_shapes: *flat_past maps to a single "flat_past" key with a list
    n = NUM_DECODER_LAYERS
    flat_past_shapes = []
    for i in range(2 * n):  # self-attention pairs
        flat_past_shapes.append({0: batch, 2: past_seq})
    for i in range(2 * n):  # cross-attention pairs
        flat_past_shapes.append({0: batch, 2: enc_seq})

    dyn_shapes = {
        "decoder_input_ids": {0: batch},
        "encoder_hidden_states": {0: batch, 1: enc_seq},
        "flat_past": tuple(flat_past_shapes),
    }

    input_names = ["decoder_input_ids", "encoder_hidden_states"] + _kv_input_names(NUM_DECODER_LAYERS)
    output_names = ["logits"] + _kv_output_names(NUM_DECODER_LAYERS)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_dec_ids, dummy_enc_hidden, *flat_past),
            str(output_path),
            dynamo=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dyn_shapes,
        )
    print(f"  decoder_with_past → {output_path}")


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_dynamic_int8(src: Path, dst: Path):
    """Weight-only dynamic INT8 quantization — no calibration needed."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        # Quantize all MatMul nodes (the bulk of transformer compute)
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,      # per-tensor is faster; set True for slightly better accuracy
        reduce_range=False,
        extra_options={"WeightSymmetric": True},
    )
    print(f"  quantized (dynamic int8) → {dst}")


def quantize_static_int8(src: Path, dst: Path, processor, audio_dir: Path):
    """Full static INT8 — both weights and activations quantized."""
    import soundfile as sf
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        CalibrationDataReader,
        QuantFormat,
    )
    # Build calibration data from WAV files
    class AudioCalibReader(CalibrationDataReader):
        def __init__(self):
            wav_paths = sorted(audio_dir.glob("*.wav"))[:50]
            if not wav_paths:
                raise FileNotFoundError(f"No .wav files found in {audio_dir}")
            self._data = []
            for p in wav_paths:
                audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
                if sr != 16000:
                    raise ValueError(f"{p}: expected 16000 Hz, got {sr}")
                inputs = processor(audio, sampling_rate=16000, return_tensors="np")
                self._data.append({"input_values": inputs.input_values})
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=AudioCalibReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  quantized (static int8) → {dst}")


# ── Validation ───────────────────────────────────────────────────────────────

def validate(model, output_dir: Path):
    """Quick numerical check: ONNX encoder output vs PyTorch."""
    import onnxruntime as ort

    dummy_audio = np.random.randn(1, 16000).astype(np.float32)
    dummy_mask = np.ones((1, 16000), dtype=np.int64)

    # PyTorch reference
    with torch.no_grad():
        enc_wrapper = EncoderWrapper(model).eval()
        pt_out = enc_wrapper(
            torch.from_numpy(dummy_audio),
            torch.from_numpy(dummy_mask),
        ).numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(str(output_dir / "encoder_model.onnx"))
    ort_out = sess.run(None, {"input_values": dummy_audio, "attention_mask": dummy_mask})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    print(f"\n  Encoder validation — max absolute diff: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Validation failed! max_diff={max_diff}"
    print("  PASSED")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download safetensors + configs to local dir (skips if already present)
    if not (MODEL_LOCAL_DIR / "model.safetensors").exists():
        download_model(MODEL_ID, MODEL_LOCAL_DIR)
    else:
        print(f"Using cached weights in {MODEL_LOCAL_DIR.resolve()}")

    print(f"\nLoading model from {MODEL_LOCAL_DIR} ...")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(MODEL_LOCAL_DIR),
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",  # avoids SDPA enable_gqa=True export bug
    ).eval().to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_LOCAL_DIR),
        local_files_only=True,
    )

    model.config.use_cache = True

    # ONNX files go directly in OUTPUT_DIR (same dir as weights subdir)
    print("\nExporting to ONNX ...")
    export_encoder(model, OUTPUT_DIR / "encoder_model.onnx")
    export_decoder(model, OUTPUT_DIR / "decoder_model.onnx")
    export_decoder_with_past(model, OUTPUT_DIR / "decoder_with_past_model.onnx")

    # Save processor/config in OUTPUT_DIR so it's self-contained for ORT loading
    processor.save_pretrained(str(OUTPUT_DIR))
    model.config.save_pretrained(str(OUTPUT_DIR))

    print("\nValidating encoder ...")
    validate(model, OUTPUT_DIR)

    fp32_models = [
        OUTPUT_DIR / "encoder_model.onnx",
        OUTPUT_DIR / "decoder_model.onnx",
        OUTPUT_DIR / "decoder_with_past_model.onnx",
    ]

    if DYNAMIC_QUANT:
        print("\nQuantizing (dynamic INT8) ...")
        for src in fp32_models:
            dst = src.with_name(src.stem + "_int8.onnx")
            quantize_dynamic_int8(src, dst)

    if STATIC_QUANT:
        print("\nQuantizing (static INT8) — encoder only ...")
        enc_src = OUTPUT_DIR / "encoder_model.onnx"
        enc_dst = OUTPUT_DIR / "encoder_model_static_int8.onnx"
        quantize_static_int8(enc_src, enc_dst, processor, CALIBRATION_AUDIO_DIR)

    print(f"\nDone. All files in: {OUTPUT_DIR.resolve()}/")
    for f in sorted(OUTPUT_DIR.glob("*.onnx")):
        size_mb = f.stat().st_size / 1e6
        tag = "INT8" if "int8" in f.name else "fp32"
        print(f"  {f.name:50s}  {size_mb:7.1f} MB  [{tag}]")


if __name__ == "__main__":
    main()
