# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Small, full-spectrum, windowed float64 STFT checked against ONNX."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
root = Path(__file__).parent
model = h.make_model(h.make_graph(
    [h.make_node("STFT", ["signal", "step", "window", "length"], ["output"], onesided=0)],
    "small_stft_f64", [h.make_tensor_value_info("signal", T.DOUBLE, [1, 32])],
    [h.make_tensor_value_info("output", T.DOUBLE, [1, 3, 20, 2])],
    [nh.from_array(np.array(5, np.int64), "step"),
     nh.from_array(np.array(20, np.int64), "length"),
     nh.from_array(np.linspace(0.5, 1.5, 20, dtype=np.float64), "window")],
), opset_imports=[h.make_opsetid("", 17)])
onnx.checker.check_model(model)
onnx.save(model, root / "stft_small_f64.onnx")
# The reference implements only the spec's trailing real-component dimension;
# Burn also accepts rank-two real signals from PyTorch exporters.
reference_model = deepcopy(model)
reference_model.graph.input[0].type.tensor_type.shape.dim.add().dim_value = 1
reference = ReferenceEvaluator(reference_model).run(None, {"signal": np.ones((1, 32, 1), np.float64)})[0]
reference.astype("<f8").tofile(root / "stft_small_f64_expected.bin")
# Constant real frames have exactly one nonzero bin, independently of FFT implementation.
constant_model = onnx.load(root / "stft_non_pow2.onnx")
constant = ReferenceEvaluator(constant_model).run(None, {"signal": np.ones((1, 32, 1), np.float32)})[0]
expected = np.zeros((1, 3, 11, 2), np.float32)
expected[:, :, 0, 0] = 20
np.testing.assert_allclose(constant, expected, atol=1e-12, rtol=0)
