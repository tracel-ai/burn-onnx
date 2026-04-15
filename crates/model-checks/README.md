# Model Checks

This directory contains model verification and validation tests for burn-onnx. Each subdirectory
represents a different model that we test to ensure burn-onnx can correctly:

1. Import ONNX models
2. Generate Rust code from the models
3. Build and run the generated code

## Purpose

The model-checks serve as integration tests to verify that burn-onnx works correctly with real-world
models. These tests help catch regressions and ensure compatibility with various ONNX operators and
model architectures.

## Structure

Each model directory typically contains:

- Model download/preparation script (e.g., `get_model.py`)
- `build.rs` - Build script that uses burn-onnx to generate Rust code
- `src/main.rs` - Test code that runs the generated model
- `Cargo.toml` - Package configuration

Model artifacts (ONNX files, test data) are stored in the platform cache directory:

- macOS: `~/Library/Caches/burn-onnx/model-checks/<model-name>/`
- Linux: `~/.cache/burn-onnx/model-checks/<model-name>/`

Set `BURN_CACHE_DIR` to override the base cache path (useful for CI).

Generated files (not tracked in git):

- `target/` - Build artifacts and generated model code

## Two-Step Process

### Step 1: Download and Prepare the Model

First, download the model and convert it to the required ONNX format:

```bash
cd model-checks/<model-name>
python get_model.py
# or using uv:
uv run get_model.py
```

The model preparation script typically:

- Downloads the model (if not already present)
- Converts it to ONNX format with the appropriate opset version
- Validates the model structure
- Saves the prepared model to the cache directory

Scripts are designed to skip downloading if the ONNX model already exists, saving time and
bandwidth.

### Step 2: Build and Run the Model

Once the ONNX model is ready, build and run the Rust code:

```bash
cargo build
cargo run
```

The build process will:

- Check that the ONNX model exists (with helpful error messages if not)
- Generate Rust code from the ONNX model using burn-onnx
- Compile the generated code

## Backend and Device Selection

All model checks support multiple backends via Cargo features:

```bash
cargo run                              # default (ndarray, CPU)
cargo run --features wgpu              # WebGPU
cargo run --features metal             # Metal (macOS)
cargo run --features tch               # LibTorch
cargo run --no-default-features --features tch   # LibTorch only
```

For the `tch` backend, the best GPU device is selected automatically:

- **macOS**: MPS (Metal Performance Shaders)
- **Linux / Windows**: CUDA (GPU 0)

Override with the `BURN_DEVICE` environment variable:

```bash
BURN_DEVICE=cpu cargo run --features tch     # force CPU
BURN_DEVICE=mps cargo run --features tch     # force MPS
BURN_DEVICE=cuda cargo run --features tch    # CUDA GPU 0
BURN_DEVICE=cuda:1 cargo run --features tch  # CUDA GPU 1
```

Other backends (wgpu, metal) already select the best GPU by default; ndarray is CPU-only.

## Models

| Directory                  | Model                               | Source              |
| -------------------------- | ----------------------------------- | ------------------- |
| `albert/`                  | ALBERT                              | HuggingFace         |
| `alexnet/`                 | AlexNet                             | ONNX Model Zoo      |
| `all-minilm-l6-v2/`       | all-MiniLM-L6-v2                    | HuggingFace         |
| `arcface/`                 | ArcFace (LResNet100E-IR)            | ONNX Model Zoo      |
| `clip-vit-b-32-text/`     | CLIP ViT-B-32 (text)                | HuggingFace         |
| `clip-vit-b-32-vision/`   | CLIP ViT-B-32 (vision)              | HuggingFace         |
| `densenet121/`             | DenseNet-121                        | ONNX Model Zoo      |
| `depth-anything-v2/`      | Depth-Anything-v2-Small             | HuggingFace         |
| `depth-pro/`              | Apple Depth Pro                     | Apple / HuggingFace |
| `inception-v1/`           | Inception v1 (GoogLeNet)            | ONNX Model Zoo      |
| `inception-v2/`           | Inception v2                        | ONNX Model Zoo      |
| `mediapipe-face-detector/` | MediaPipe Face Detector (BlazeFace) | Google MediaPipe    |
| `mobilenet-v2/`           | MobileNet v2                        | ONNX Model Zoo      |
| `modernbert-base/`        | ModernBERT-base                     | HuggingFace         |
| `qwen/`                   | Qwen 1.5/2.5/3 (0.5B-0.6B)         | HuggingFace         |
| `resnet50/`               | ResNet-50                           | ONNX Model Zoo      |
| `rf-detr/`                | RF-DETR Small                       | Roboflow            |
| `rtmw3d/`                 | RTMW3D-x                            | MMPose              |
| `shufflenet/`             | ShuffleNet v2                       | ONNX Model Zoo      |
| `silero-vad/`             | Silero VAD                          | Silero              |
| `smollm/`                 | SmolLM / SmolLM2 (135M)             | HuggingFace         |
| `squeezenet/`             | SqueezeNet 1.0                      | ONNX Model Zoo      |
| `stable-diffusion-xl/`    | Stable Diffusion XL 1.0 (UNet)      | Stability AI        |
| `vgg19/`                  | VGG-19                              | ONNX Model Zoo      |
| `yolo/`                   | YOLO (v5/v8/v10/v11/v12/v26)        | Ultralytics         |
| `zfnet512/`               | ZFNet-512                           | ONNX Model Zoo      |
