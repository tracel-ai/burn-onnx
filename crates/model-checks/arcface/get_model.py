#!/usr/bin/env -S uv run --python 3.11 --script

# /// script
# python = "3.11"
# dependencies = [
#   "onnx>=1.17.0",
#   "onnxruntime>=1.18.0",
#   "numpy",
#   "torch",
# ]
# ///

"""
Download and prepare the ArcFace (LResNet100E-IR) model for testing with burn-onnx.

This is an opset-8 ONNX model from the ONNX Model Zoo where weights appear as
graph inputs with matching initializers (the older ONNX pattern). The opset
upgrade to 16 preserves this pattern, which exercises the PReLU slope-as-graph-input
code path.

See: https://github.com/tracel-ai/burn-onnx/issues/305
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir, process_model

MODEL_URL = "https://tract-ci-builds.s3.amazonaws.com/model/arcfaceresnet100-8.onnx"


def download_model(output_path):
    """Download the ArcFace ONNX model."""
    print(f"Downloading ArcFace (LResNet100E-IR) model...")
    print(f"  URL: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, output_path)
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {file_size:.1f} MB")


def generate_test_data(model_path, output_path):
    """Generate test input/output data and save as PyTorch tensors."""
    import onnxruntime as ort
    import torch

    print("Generating test data...")

    session = ort.InferenceSession(str(model_path))

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  Model inputs:")
    for inp in inputs:
        print(f"    {inp.name}: {inp.shape} ({inp.type})")
    print(f"  Model outputs:")
    for out in outputs:
        print(f"    {out.name}: {out.shape} ({out.type})")

    # Create reproducible test input (batch=1, 3 channels, 112x112 face image)
    np.random.seed(42)
    input_info = inputs[0]
    input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    test_input = np.random.rand(*input_shape).astype(np.float32)

    # Run inference to get reference outputs
    feed = {input_info.name: test_input}
    results = session.run(None, feed)

    # Save as PyTorch tensors
    output_names = [out.name for out in outputs]
    test_data = {input_info.name: torch.from_numpy(test_input)}
    for name, result in zip(output_names, results):
        test_data[name] = torch.from_numpy(result)
        print(f"  {name} shape: {result.shape}")

    torch.save(test_data, output_path)
    print(f"  Test data saved to: {output_path}")


def main():
    print("=" * 60)
    print("ArcFace (LResNet100E-IR, ResNet-100)")
    print("=" * 60)
    print()

    artifacts_dir = get_artifacts_dir("arcface")

    raw_path = artifacts_dir / "arcfaceresnet100-8.onnx"
    onnx_path = artifacts_dir / "arcface.onnx"
    test_data_path = artifacts_dir / "test_data.pt"

    # Check if we already have everything
    if onnx_path.exists() and test_data_path.exists():
        print(f"All files already exist:")
        print(f"  Model: {onnx_path}")
        print(f"  Test data: {test_data_path}")
        print("\nTo re-download, delete the artifacts directory and run again.")
        return

    # Step 1: Download model
    if not raw_path.exists():
        print("Step 1: Downloading model...")
        download_model(raw_path)
    else:
        print(f"Step 1: Raw model already exists at {raw_path}")
    print()

    # Step 2: Process model (upgrade opset, shape inference)
    if not onnx_path.exists():
        print("Step 2: Processing model (opset upgrade + shape inference)...")
        process_model(str(raw_path), str(onnx_path))

        # Clean up raw model
        raw_path.unlink()
        print("  Cleaned up raw model file")
    else:
        print(f"Step 2: Processed model already exists at {onnx_path}")
    print()

    # Step 3: Generate test data
    if not test_data_path.exists():
        print("Step 3: Generating test data...")
        generate_test_data(onnx_path, test_data_path)
    else:
        print(f"Step 3: Test data already exists at {test_data_path}")
    print()

    print("=" * 60)
    print("Model preparation completed!")
    print("=" * 60)
    print()
    print("Related issue: https://github.com/tracel-ai/burn-onnx/issues/305")
    print()
    print("Next steps:")
    print("  1. Build the model: cargo build")
    print("  2. Run the test:    cargo run")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
