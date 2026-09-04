#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/max/max_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    np.random.seed(42)

    # Create a graph that tests both Shape-Scalar and Shape-Shape operations
    # Input tensors
    input_tensor1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [10, 8, 6]
    )
    input_tensor2 = helper.make_tensor_value_info(
        "input2", TensorProto.FLOAT, [2, 30, 4]
    )

    # Shape nodes - extract shapes of inputs
    shape_node1 = helper.make_node("Shape", inputs=["input1"], outputs=["shape1"])

    shape_node2 = helper.make_node("Shape", inputs=["input2"], outputs=["shape2"])

    # Constant scalar value
    scalar_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scalar"],
        value=helper.make_tensor(
            name="const_tensor", data_type=TensorProto.INT64, dims=[], vals=[7]
        ),
    )

    # Max scalar from shape Max(Shape, Scalar)
    max_shape_scalar_node = helper.make_node(
        "Max", inputs=["shape1", "scalar"], outputs=["max_shape_scalar"]
    )

    # Max shape from scalar Max(Scalar, Shape)
    max_scalar_shape_node = helper.make_node(
        "Max", inputs=["scalar", "shape1"], outputs=["max_scalar_shape"]
    )

    # Max shape from shape Max(Shape, Shape)
    max_shapes_node = helper.make_node(
        "Max", inputs=["shape1", "shape2"], outputs=["max_shape_shape"]
    )

    # Outputs - shape arrays
    output1 = helper.make_tensor_value_info("max_shape_scalar", TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info("max_scalar_shape", TensorProto.INT64, [3])
    output3 = helper.make_tensor_value_info("max_shape_shape", TensorProto.INT64, [3])

    # Create the graph
    graph_def = helper.make_graph(
        [
            shape_node1,
            shape_node2,
            scalar_const,
            max_shape_scalar_node,
            max_scalar_shape_node,
            max_shapes_node,
        ],
        "max_shape_test",
        [input_tensor1, input_tensor2],
        [output1, output2, output3],
    )

    # Create the model
    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    # Save the model
    onnx_name = "max_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    # Test the model with sample data
    test_input1 = np.random.randn(10, 8, 6).astype(np.float32)
    test_input2 = np.random.randn(2, 30, 4).astype(np.float32)

    print(f"\nTest input1 shape: {test_input1.shape}")
    print(f"Test input2 shape: {test_input2.shape}")

    # Run the model using ReferenceEvaluator
    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input1": test_input1, "input2": test_input2})

    max_shape_scalar, max_scalar_shape, max_shape_shape = outputs

    print(f"\nTest output max_shape_scalar: {repr(max_shape_scalar)}")
    print(f"Test output max_scalar_shape: {repr(max_scalar_shape)}")
    print(f"Test output max_shape_shape: {repr(max_shape_shape)}")

    assert np.array_equal(max_shape_scalar, max_scalar_shape), (
        "Max results should be the same"
    )
    np.testing.assert_array_equal(max_scalar_shape, [10, 8, 7])
    np.testing.assert_array_equal(max_shape_shape, [10, 30, 6])


if __name__ == "__main__":
    main()
