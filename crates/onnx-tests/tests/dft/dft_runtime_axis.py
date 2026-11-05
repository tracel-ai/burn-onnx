#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/dft/dft_runtime_axis.onnx
#
# Real DFT (opset 20) whose `axis` is a graph input rather than a constant, as in the
# ONNX backend tests `test_dft` / `test_dft_axis`. Both signal axes have
# non-power-of-two lengths (3 and 5), so burn's FFT takes its Bluestein path.
#
# Input x: [1, 3, 5, 1], axis: int64 scalar
# full: two-sided spectrum, same shape as x with last dim 2
# half: onesided spectrum, axis length n -> n/2 + 1
# The test runs the model with axis=1 and with axis=-2 (counts from the end of the
# rank-4 input, i.e. axis 2).

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 1])
    axis = helper.make_tensor_value_info("axis", TensorProto.INT64, [])
    full = helper.make_tensor_value_info("full", TensorProto.FLOAT, [1, "h", "w", 2])
    half = helper.make_tensor_value_info("half", TensorProto.FLOAT, [1, "hh", "hw", 2])

    nodes = [
        helper.make_node("DFT", ["x", "", "axis"], ["full"]),
        helper.make_node("DFT", ["x", "", "axis"], ["half"], onesided=1),
    ]
    graph = helper.make_graph(nodes, "dft_runtime_axis_model", [x, axis], [full, half])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "dft_runtime_axis.onnx")

    np.random.seed(42)
    test_input = np.random.randn(1, 3, 5, 1).astype(np.float32)
    print(f"x: {test_input.reshape(-1).tolist()}")

    session = ReferenceEvaluator(model)
    for a in [1, -2]:
        full_out, half_out = session.run(
            None, {"x": test_input, "axis": np.array(a, dtype=np.int64)}
        )
        print(f"axis={a} full {full_out.shape}: {full_out.tolist()}")
        print(f"axis={a} half {half_out.shape}: {half_out.tolist()}")


if __name__ == "__main__":
    main()
