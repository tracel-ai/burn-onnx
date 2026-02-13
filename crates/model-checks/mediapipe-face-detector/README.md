# MediaPipe Face Detector (BlazeFace Short Range)

Model check for the MediaPipe BlazeFace face detection model, originally reported
in [tracel-ai/burn#1370](https://github.com/tracel-ai/burn/issues/1370).

## Model

- **Source**: Google MediaPipe (TFLite, converted to ONNX via tf2onnx)
- **Input**: `[1, 128, 128, 3]` (NHWC image)
- **Outputs**:
  - `regressors`: `[1, 896, 16]` (bounding box + keypoint regressions)
  - `classifiers`: `[1, 896, 1]` (face confidence scores)

## Usage

```bash
# Download and convert model
uv run get_model.py

# Build and run
cargo run --release
```
