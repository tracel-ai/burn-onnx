#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/mod/mod_shape.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

# ONNX opset version to use for model generation
OPSET_VERSION = 16


def main():
    # Two input tensors whose shapes feed into Mod(Shape, Shape) and
    # Mod(Shape, scalar). This mirrors what expanded attention models do when
    # validating head counts.
    input_tensor1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [12, 8, 6]
    )
    input_tensor2 = helper.make_tensor_value_info(
        "input2", TensorProto.FLOAT, [4, 2, 3]
    )

    shape_node1 = helper.make_node("Shape", inputs=["input1"], outputs=["shape1"])
    shape_node2 = helper.make_node("Shape", inputs=["input2"], outputs=["shape2"])

    scalar_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["scalar"],
        value=helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.INT64,
            dims=[],
            vals=[5],
        ),
    )

    # Shape % Shape
    mod_shapes_node = helper.make_node(
        "Mod", inputs=["shape1", "shape2"], outputs=["shape_mod_shape"]
    )

    # Shape % scalar
    mod_scalar_node = helper.make_node(
        "Mod", inputs=["shape1", "scalar"], outputs=["shape_mod_scalar"]
    )

    output1 = helper.make_tensor_value_info("shape_mod_shape", TensorProto.INT64, [3])
    output2 = helper.make_tensor_value_info(
        "shape_mod_scalar", TensorProto.INT64, [3]
    )

    graph_def = helper.make_graph(
        [shape_node1, shape_node2, scalar_const, mod_shapes_node, mod_scalar_node],
        "shape_mod_test",
        [input_tensor1, input_tensor2],
        [output1, output2],
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="onnx-tests",
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)],
    )

    onnx_name = "mod_shape.onnx"
    onnx.save(model_def, onnx_name)
    print("Finished exporting model to {}".format(onnx_name))

    test_input1 = np.random.randn(12, 8, 6).astype(np.float32)
    test_input2 = np.random.randn(4, 2, 3).astype(np.float32)

    print(f"\nTest input1 shape: {test_input1.shape}")
    print(f"Test input2 shape: {test_input2.shape}")

    session = ReferenceEvaluator(onnx_name, verbose=0)
    outputs = session.run(None, {"input1": test_input1, "input2": test_input2})

    shape_mod_shape, shape_mod_scalar = outputs

    print(f"\nTest output shape_mod_shape: {repr(shape_mod_shape)}")
    print(f"Test output shape_mod_scalar: {repr(shape_mod_scalar)}")


if __name__ == "__main__":
    main()
