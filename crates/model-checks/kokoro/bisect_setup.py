#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime==1.19.2",
#   "numpy",
# ]
# ///

"""
Bisection helper. Modifies the kokoro ONNX graph to expose a set of
intermediate tensors as additional graph outputs, runs ONNX Runtime once,
and writes:
  - kokoro-v1-debug.onnx     (the model with extra outputs)
  - bisect_taps.json         (the names + per-tensor stats from ORT)

The Rust side then loads kokoro-v1-debug.onnx and compares its forward()
return values against bisect_taps.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir


def select_taps(model):
    """Pick strategic intermediate tensors to expose."""
    nodes = list(model.graph.node)
    name_to_node = {n.name: n for n in nodes}

    taps = []

    # All LSTM outputs (the first / hidden output).
    for n in nodes:
        if n.op_type == "LSTM":
            taps.append(("LSTM", n.output[0], n.name))

    # All ConvTranspose outputs.
    for n in nodes:
        if n.op_type == "ConvTranspose":
            taps.append(("ConvTranspose", n.output[0], n.name))

    # The single STFT output.
    for n in nodes:
        if n.op_type == "STFT":
            taps.append(("STFT", n.output[0], n.name))

    # Drill-in: every Conv inside the ups.0..ups.1 generator stack.
    by_index = {i: n for i, n in enumerate(nodes)}
    for i in range(1855, 2145):
        n = by_index[i]
        if n.op_type in ("Conv", "LeakyRelu", "Sin"):
            taps.append((n.op_type, n.output[0], n.name or f"{n.op_type}_{i}"))

    # Second drill-in: find where drift first enters. Sparse taps through the
    # decoder/generator pre-ups.0 region. Tap every ReduceMean and Div output
    # (the AdaIN normalization chokepoints), plus Conv and Add outputs every
    # ~25 nodes for broad coverage.
    last_index_by_op = {}
    for i in range(1300, 1855):
        n = by_index[i]
        if n.op_type in ("ReduceMean", "Div", "Sqrt", "Conv") and n.output:
            prev = last_index_by_op.get(n.op_type, -100)
            # Skip if too close to previous of same type (keep sparse)
            if i - prev >= 20 or n.op_type in ("Div", "Sqrt"):
                taps.append((n.op_type, n.output[0], n.name or f"{n.op_type}_{i}"))
                last_index_by_op[n.op_type] = i

    # Pre-noise_convs phase/magnitude chain: this is where manual atan2 lives.
    # Sign-of-near-zero on `real` flips a quadrant in this chain, which
    # propagates phase by ±π even when the rest of the chain is precise.
    phase_chain_targets = [
        "/decoder/decoder/generator/Gather_4",   # real
        "/decoder/decoder/generator/Gather_5",   # imag
        "/decoder/decoder/generator/Pow",        # real^2
        "/decoder/decoder/generator/Pow_1",      # imag^2
        "/decoder/decoder/generator/Add",        # real^2 + imag^2
        "/decoder/decoder/generator/Sqrt",       # magnitude
        "/decoder/decoder/generator/Div",        # imag/real
        "/decoder/decoder/generator/Atan",       # atan(imag/real)
        "/decoder/decoder/generator/Greater",    # imag > 0
        "/decoder/decoder/generator/Less",       # real < 0
        "/decoder/decoder/generator/Where",      # quadrant 2/3 select
        "/decoder/decoder/generator/Where_1",    # final phase
        "/decoder/decoder/generator/Concat_3",   # [magnitude, phase]
    ]
    name_to_node_dict = {n.name: n for n in nodes}
    for name in phase_chain_targets:
        node = name_to_node_dict.get(name)
        if node is None:
            print(f"  warning: phase chain node not found: {name}")
            continue
        taps.append((node.op_type, node.output[0], name))

    # iSTFT block waypoints.
    istft_targets = [
        "/decoder/decoder/generator/istft/stft/Mul",       # real channel
        "/decoder/decoder/generator/istft/stft/Mul_1",     # imag channel
        "/decoder/decoder/generator/istft/stft/Concat",    # combined complex
        "/decoder/decoder/generator/istft/stft/ConvTranspose",  # overlap-add audio
        "/decoder/decoder/generator/istft/stft/Squeeze_1", # ScatterND data
        "/decoder/decoder/generator/istft/stft/NonZero",   # window-sum mask
        "/decoder/decoder/generator/istft/stft/Gather",    # numerator gather
        "/decoder/decoder/generator/istft/stft/Gather_1",  # denominator gather
        "/decoder/decoder/generator/istft/stft/Div",       # per-sample normalization
        "/decoder/decoder/generator/istft/stft/ScatterND", # post-scatter audio
        "/decoder/decoder/generator/istft/stft/Mul_4",     # final scale
        "/decoder/decoder/generator/istft/stft/Slice_3",   # post first trim
        "/decoder/decoder/generator/istft/stft/Slice_4",   # final audio
    ]
    for name in istft_targets:
        node = name_to_node.get(name)
        if node is None:
            print(f"  warning: iSTFT node not found: {name}")
            continue
        taps.append((node.op_type, node.output[0], name))

    # The graph's actual output (audio).
    for o in model.graph.output:
        taps.append(("graph_output", o.name, "<graph output>"))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in taps:
        if t[1] in seen:
            continue
        seen.add(t[1])
        unique.append(t)
    return unique


def main():
    artifacts = get_artifacts_dir("kokoro")
    src = artifacts / "kokoro-v1.0.onnx"
    if not src.exists():
        print(f"missing model: {src}; run get_model.py first")
        sys.exit(1)

    print(f"Loading {src}...")
    model = onnx.load(str(src))

    print("Running ONNX shape inference to recover tap dtypes...")
    inferred = onnx.shape_inference.infer_shapes(model)
    name_to_vi = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        name_to_vi[vi.name] = vi

    taps = select_taps(model)
    print(f"Selected {len(taps)} taps")

    # Add intermediate value_infos as graph outputs.
    existing_output_names = {o.name for o in model.graph.output}
    new_outputs = []
    for op_type, tensor_name, node_name in taps:
        if tensor_name in existing_output_names:
            continue
        vi = name_to_vi.get(tensor_name)
        if vi is None:
            print(f"  warning: no inferred type for {tensor_name}, skipping")
            continue
        new_outputs.append(vi)
    model.graph.output.extend(new_outputs)
    print(f"Added {len(new_outputs)} new graph outputs (existing: {len(existing_output_names)})")

    debug_path = artifacts / "kokoro-v1-debug.onnx"
    onnx.save(model, str(debug_path))
    print(f"Wrote {debug_path} ({debug_path.stat().st_size / 1e6:.1f} MB)")

    # Run ORT once with the same fixed seed inputs as get_model.py.
    print("Creating ORT session...")
    sess = ort.InferenceSession(str(debug_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    inner = rng.integers(low=1, high=178, size=8, dtype=np.int64)
    tokens = np.concatenate([[0], inner, [0]]).reshape(1, -1).astype(np.int64)
    style = (rng.standard_normal((1, 256)) * 0.1).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)

    print("Running ORT with intermediate taps...")
    out_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(out_names, {"tokens": tokens, "style": style, "speed": speed})
    by_name = dict(zip(out_names, outputs))

    # Exposing intermediate outputs can inhibit ORT fusions and change rounding.
    # Measure that effect on the same inputs before treating this graph's taps as
    # an explanation of the uninstrumented model's final audio.
    del sess
    original_session = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    original_audio = original_session.run(None, {"tokens": tokens, "style": style, "speed": speed})[0].reshape(-1)
    debug_audio = by_name[model.graph.output[0].name].reshape(-1)
    delta = np.abs(debug_audio.astype(np.float64) - original_audio.astype(np.float64))
    graph_comparison = {
        "original_peak": float(np.abs(original_audio).max()),
        "instrumented_peak": float(np.abs(debug_audio).max()),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "pearson": float(np.corrcoef(original_audio, debug_audio)[0, 1]),
    }
    del original_session
    print("Original vs instrumented ORT audio:", graph_comparison)
    if graph_comparison["max_abs_error"] > 0:
        print("WARNING: instrumentation changed ORT audio; tap statistics describe the instrumented graph only.")

    summary = []
    for (op_type, tensor_name, node_name) in taps:
        val = by_name.get(tensor_name)
        if val is None:
            continue
        if not isinstance(val, np.ndarray):
            continue
        v = val.astype(np.float64) if np.issubdtype(val.dtype, np.floating) else val.astype(np.int64).astype(np.float64)
        flat = v.reshape(-1)
        stats = {
            "op_type": op_type,
            "tensor": tensor_name,
            "node": node_name,
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "min": float(flat.min()) if flat.size else 0.0,
            "max": float(flat.max()) if flat.size else 0.0,
            "mean": float(flat.mean()) if flat.size else 0.0,
            "std": float(flat.std()) if flat.size else 0.0,
            "abs_max": float(np.abs(flat).max()) if flat.size else 0.0,
            "abs_mean": float(np.abs(flat).mean()) if flat.size else 0.0,
        }
        # For small tensors, save full values; for large, save head + sampled.
        if val.size <= 64:
            stats["values"] = val.flatten().tolist()
        else:
            stats["head"] = val.flatten()[:32].tolist()
        summary.append(stats)

    out_json = artifacts / "bisect_taps.json"
    with open(out_json, "w") as f:
        json.dump({"reference_graph": "instrumented", "original_graph_comparison": graph_comparison, "taps": summary}, f, indent=2)
    print(f"Wrote {out_json}")

    # Also dump full STFT real/imag values for element-wise comparison.
    full_dumps = {}
    targets_for_full = [
        "/decoder/decoder/generator/STFT_output_0",
        "/decoder/decoder/generator/Gather_4_output_0",
        "/decoder/decoder/generator/Gather_5_output_0",
        "/decoder/decoder/generator/Atan_output_0",
        "/decoder/decoder/generator/Where_1_output_0",
    ]
    for name in targets_for_full:
        if name in by_name:
            full_dumps[name] = {
                "shape": list(by_name[name].shape),
                "values": by_name[name].flatten().astype(np.float64).tolist(),
            }
    with open(artifacts / "ort_full_taps.json", "w") as f:
        json.dump(full_dumps, f)
    print(f"Wrote ort_full_taps.json with {len(full_dumps)} full-value dumps")

    print()
    print("Per-tap stats from ORT:")
    print(f"{'#':>3} {'op_type':16} {'shape':25} abs_max     abs_mean   tensor")
    for i, s in enumerate(summary):
        shape_s = "x".join(str(d) for d in s["shape"])
        print(
            f"{i:>3} {s['op_type']:16} {shape_s:25} "
            f"{s['abs_max']:10.4f}  {s['abs_mean']:10.4f}  {s['tensor'][-50:]}"
        )


if __name__ == "__main__":
    main()
