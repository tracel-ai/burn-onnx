"""Shared utilities for model-checks get_model.py scripts."""

import os
import sys
from pathlib import Path

import onnx
from onnx import shape_inference, version_converter


def get_artifacts_dir(model_name: str) -> Path:
    """Get platform-specific cache directory for model artifacts."""
    env_dir = os.environ.get("BURN_CACHE_DIR")
    if env_dir:
        base = Path(env_dir)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "burn-onnx"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        base = Path(local) / "burn-onnx"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        base = Path(xdg) / "burn-onnx"
    d = base / "model-checks" / model_name
    d.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {d}")
    return d


def process_model(input_path, output_path, target_opset=16):
    """Load, upgrade opset, and apply shape inference to model."""
    print(f"Loading model from {input_path}...")
    model = onnx.load(input_path)

    # Check and upgrade opset if needed
    current_opset = model.opset_import[0].version
    if current_opset < target_opset:
        print(f"Upgrading opset from {current_opset} to {target_opset}...")
        model = version_converter.convert_version(model, target_opset)

    # Apply shape inference
    print("Applying shape inference...")
    model = shape_inference.infer_shapes(model)

    # Save processed model
    onnx.save(model, output_path)
    print(f"✓ Processed model saved to: {output_path}")

    return model
