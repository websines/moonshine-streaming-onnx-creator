# Moonshine v2 Streaming ASR: Safetensors → ONNX INT8

## Architecture Quick Ref

Moonshine v2 Streaming is an **encoder-decoder** model:

| Component | Details | Quantization |
|-----------|---------|-------------|
| **Audio Frontend** | Causal stride-2 convolutions, 50Hz features | ⚠️ **Keep FP16+ min** (inputs are 16-bit PCM as floats) |
| **Encoder** | Sliding-window attention, no positional embeddings (ergodic) | ✅ INT8 MatMul |
| **Adapter** | Learned positional embeddings, dimension alignment | ✅ INT8 weights |
| **Decoder** | Causal Transformer, RoPE, autoregressive w/ cross-attention | ✅ INT8 weights + MatMul |

## Quick Start

### Prerequisites

```bash
pip install transformers optimum[onnxruntime] onnxruntime onnx soundfile
# Optional: Moonshine's own quantizer
pip install onnx-shrink-ray
```

### Option A: One-liner via Optimum CLI

```bash
# Step 1: Export to ONNX
optimum-cli export onnx \
  --model UsefulSensors/moonshine-streaming-small \
  --task automatic-speech-recognition-with-past \
  ./moonshine_onnx_fp32/

# Step 2: Quantize (dynamic INT8)
python moonshine_v2_streaming_to_onnx_int8.py \
  --model UsefulSensors/moonshine-streaming-small \
  --skip-export \
  --onnx-dir ./moonshine_onnx_fp32 \
  --int8-dir ./moonshine_onnx_int8 \
  --quant-method dynamic
```

### Option B: Full pipeline

```bash
python moonshine_v2_streaming_to_onnx_int8.py \
  --model UsefulSensors/moonshine-streaming-small \
  --quant-method dynamic
```

### Option C: Static quantization (best accuracy, needs calibration audio)

```bash
python moonshine_v2_streaming_to_onnx_int8.py \
  --model UsefulSensors/moonshine-streaming-small \
  --quant-method static \
  --calibration-audio-dir ./calibration_wavs/
```

### Option D: Use ONNX Shrink Ray (matches Moonshine's official process)

```bash
python moonshine_v2_streaming_to_onnx_int8.py \
  --model UsefulSensors/moonshine-streaming-small \
  --quant-method shrink_ray
```

## What Moonshine Officially Does

From the repo (`scripts/quantize-streaming-model.sh`):

1. **INT8 weights across the board** — all weight tensors quantized to 8-bit
2. **INT8 calculations for MatMul** — heavy compute ops in INT8
3. **Frontend Conv stays FP16+** — the audio frontend convolutions must keep higher precision because raw audio inputs are 16-bit signed integers encoded as floats
4. Uses a combination of **OnnxRuntime quantization tools** + **ONNX Shrink Ray**
5. Final format is `.ort` (OnnxRuntime flatbuffer) for cross-platform C++ deployment

## Key Gotchas

1. **Frontend precision**: The first Conv layers in the encoder process raw 16kHz PCM. Quantizing these to INT8 causes significant accuracy degradation. The script automatically excludes Conv/ConvTranspose ops from the encoder quantization.

2. **Decoder KV cache**: For streaming, the decoder uses past key-values. The `--task automatic-speech-recognition-with-past` flag in optimum handles this by merging the decoder graphs.

3. **Sliding window attention**: The encoder uses window sizes of (16,4) for first/last 2 layers and (16,0) for middle layers, giving 80ms lookahead. This is native ONNX ops and quantizes fine.

4. **Dynamic vs Static**: Dynamic quantization (weights-only INT8) is the safest starting point with ~70% size reduction and minimal accuracy loss. Static adds activation quantization for ~10-20% more speedup but needs calibration data.

## Model Sizes (approximate)

| Model | FP32 | INT8 (dynamic) | INT8 (shrink ray) |
|-------|------|-----------------|-------------------|
| Tiny | ~75MB | ~25MB | ~22MB |
| Small | ~200MB | ~65MB | ~55MB |
| Medium | ~800MB | ~250MB | ~220MB |

## Inference Example

```python
import onnxruntime as ort
import numpy as np

# Load quantized encoder + decoder
encoder = ort.InferenceSession("moonshine_onnx_int8/encoder_model.onnx",
                                providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession("moonshine_onnx_int8/decoder_model_merged.onnx",
                                providers=["CPUExecutionProvider"])

# Process audio chunk (16kHz, float32)
audio_chunk = np.random.randn(1, 16000).astype(np.float32)  # 1 second

# Encoder
enc_out = encoder.run(None, {"input_features": audio_chunk})

# Decoder (autoregressive loop)
decoder_ids = np.array([[1]], dtype=np.int64)  # BOS token
for _ in range(100):
    dec_out = decoder.run(None, {
        "input_ids": decoder_ids,
        "encoder_hidden_states": enc_out[0],
    })
    next_token = np.argmax(dec_out[0][:, -1:, :], axis=-1)
    if next_token[0, 0] == 2:  # EOS
        break
    decoder_ids = np.concatenate([decoder_ids, next_token], axis=1)
```