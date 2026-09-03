#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/max/max_shape_tensor.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    np.random.seed(42)

    # Create a graph that tests Shape-Tensor and Tensor-Shape operations
    # Input tensors
    input_tensor = helper.make_tensor_value_info(
        "input_tensor", TensorProto.FLOAT, [5, 7, 9]
    )
    input_tensor_1d = helper.make_tensor_value_info("input_1d", TensorProto.INT64, [3])

    # Shape node - extract shape of input
    shape_node = helper.make_node("Shape", inputs=["input_tensor"], outputs=["shape"])

    # Max tensor from shape Max(Shape, Tensor)
    max_shape_tensor = helper.make_node(
        "Max", inputs=["shape", "input_1d"], outputs=["max_shape_tensor"]
    )

    # Max shape from tensor Max(Tensor, Shape)
    max_tensor_shape = helper.make_node(
        "Max", inputs=["input_1d", "shape"], outputs=["max_tensor_shape"]
    )

    # Outputs
    output1 = helper.make_tensor_value_info("max_shape_tensor", TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info("max_tensor_shape", TensorProto.INT64, [3])

    # Create the graph
    graph_def = helper.make_graph(
        [shape_node, max_shape_tensor, max_tensor_shape],
        "max_shape_tensor_test",
        [input_tensor, input_tensor_1d],
        [output1, output2],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    # Save the model
    onnx_name = "max_shape_tensor.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    # Test the model with sample data
    test_input = np.random.randn(5, 7, 9).astype(np.float32)
    test_1d = np.array([2, 30, 4], dtype=np.int64)

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test 1d tensor: {test_1d}")

    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input_tensor": test_input, "input_1d": test_1d})

    max_shape_tensor, max_tensor_shape = outputs

    print(f"\nTest output max_shape_tensor: {repr(max_shape_tensor)}")
    print(f"Test output max_tensor_shape: {repr(max_tensor_shape)}")

    # Verify results are the same
    assert np.array_equal(max_shape_tensor, max_tensor_shape), (
        "Max results should be the same"
    )


if __name__ == "__main__":
    main()
