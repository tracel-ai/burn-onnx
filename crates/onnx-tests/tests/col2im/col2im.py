#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: col2im.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Col2Im: input [N, C * prod(block_shape), L], image_shape, block_shape -> output [N, C, *image_shape]
    # Using: N=1, C=1, image_shape=[5,5], block_shape=[1,5]
    # With these settings: L = ((5-1)/1+1) * ((5-5)/1+1) = 5 * 1 = 5
    # col_channels = C * prod(block_shape) = 1 * 1 * 5 = 5
    # Input shape: [1, 5, 5]
    input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5, 5])
    image_shape_tensor = onnx.helper.make_tensor_value_info(
        "image_shape", TensorProto.INT64, [2]
    )
    block_shape_tensor = onnx.helper.make_tensor_value_info(
        "block_shape", TensorProto.INT64, [2]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1, 5, 5]
    )

    # Create the Col2Im node
    col2im_node = onnx.helper.make_node(
        "Col2Im",
        inputs=["input", "image_shape", "block_shape"],
        outputs=["output"],
        name="Col2ImNode",
    )

    # Create constant initializers for image_shape and block_shape
    image_shape_init = onnx.numpy_helper.from_array(
        np.array([5, 5], dtype=np.int64), name="image_shape"
    )
    block_shape_init = onnx.numpy_helper.from_array(
        np.array([1, 5], dtype=np.int64), name="block_shape"
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [col2im_node],
        "Col2ImModel",
        [input_tensor],
        [output_tensor],
        initializer=[image_shape_init, block_shape_init],
    )

    # Create the model (opset 18 required for Col2Im)
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 18)],
        graph=graph,
        producer_name="ONNX_Generator",
    )

    return model


def main():
    print("Building Col2Im model...")

    # Build model
    np.random.seed(42)
    test_input = np.random.randn(1, 5, 5).astype(np.float32).round(2)
    onnx_model = build_model()
    file_name = "col2im.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    (test_output,) = session.run(None, {"input": test_input})
    print(f"Test output:\n{repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")


if __name__ == "__main__":
    main()
