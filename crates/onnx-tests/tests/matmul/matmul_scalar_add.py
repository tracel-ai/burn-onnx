#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/matmul/matmul_scalar_add.onnx
# Tests that MatMul followed by Add with scalar bias is NOT fused into Linear.
# Regression test for https://github.com/tracel-ai/burn-onnx/issues/308

import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator
import numpy as np


def main():
    np.random.seed(42)

    in_features = 4
    out_features = 8

    # Weight matrix [in_features, out_features]
    weight_data = np.random.randn(in_features, out_features).astype(np.float32)
    # Scalar bias [1] (not [out_features])
    scalar_bias_data = np.array([0.5], dtype=np.float32)

    # Graph inputs
    x_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, in_features])

    # Constants
    weight_init = numpy_helper.from_array(weight_data, name="weight")
    bias_init = numpy_helper.from_array(scalar_bias_data, name="scalar_bias")

    # MatMul: x @ weight -> [2, out_features]
    matmul_node = helper.make_node(
        "MatMul", inputs=["x", "weight"], outputs=["matmul_out"], name="matmul"
    )

    # Add: matmul_out + scalar_bias (broadcasts [1] to [2, out_features])
    add_node = helper.make_node(
        "Add",
        inputs=["matmul_out", "scalar_bias"],
        outputs=["output"],
        name="add_scalar",
    )

    # Output
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, out_features]
    )

    graph = helper.make_graph(
        [matmul_node, add_node],
        "matmul_scalar_add",
        [x_input],
        [output_info],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8

    file_name = "matmul_scalar_add.onnx"
    onnx.save(model, file_name)
    print(f"Saved {file_name}")

    # Compute expected output
    x = np.arange(2 * in_features, dtype=np.float32).reshape(2, in_features)
    sess = ReferenceEvaluator(model)
    (output,) = sess.run(None, {"x": x})

    print(f"\nInput x:\n{x}")
    print(f"Weight shape: {weight_data.shape}")
    print(f"Scalar bias: {scalar_bias_data}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    print(f"\nFor Rust test:")
    print(f"  output: {output.flatten().tolist()}")


if __name__ == "__main__":
    main()
