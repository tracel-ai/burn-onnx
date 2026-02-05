#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: expand_with_where_shape.onnx
# Tests that Expand can determine rank from Where's static shape output.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 17


def main():
    # Reproduces the pattern: Where selects between constant shapes, result feeds Expand
    # Constant([3]) -> ConstantOfShape -> ones [3]
    # ones * -1 -> [-1, -1, -1]
    # Equal([2,3,4], [-1,-1,-1]) -> [False, False, False]
    # Where(equal_out, ones, [2,3,4]) -> [2, 3, 4]
    # Expand(input, [2, 3, 4]) -> output

    const_3 = helper.make_node(
        "Constant",
        [],
        ["const_3"],
        value=numpy_helper.from_array(np.array([3], dtype=np.int64), name="c3"),
    )
    cos_node = helper.make_node(
        "ConstantOfShape",
        ["const_3"],
        ["ones"],
        value=numpy_helper.from_array(np.array([1], dtype=np.int64), name="fill"),
    )
    const_neg1 = helper.make_node(
        "Constant",
        [],
        ["neg1"],
        value=numpy_helper.from_array(np.int64(-1), name="neg1"),
    )
    mul_node = helper.make_node("Mul", ["ones", "neg1"], ["neg_ones"])

    const_shape = helper.make_node(
        "Constant",
        [],
        ["target_shape"],
        value=numpy_helper.from_array(np.array([2, 3, 4], dtype=np.int64), name="ts"),
    )
    equal_node = helper.make_node("Equal", ["target_shape", "neg_ones"], ["eq_out"])

    const_shape2 = helper.make_node(
        "Constant",
        [],
        ["target_shape2"],
        value=numpy_helper.from_array(np.array([2, 3, 4], dtype=np.int64), name="ts2"),
    )
    where_node = helper.make_node(
        "Where", ["eq_out", "ones", "target_shape2"], ["expand_shape"]
    )
    expand_node = helper.make_node("Expand", ["input", "expand_shape"], ["output"])

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, None])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, None])

    graph = helper.make_graph(
        [
            const_3,
            cos_node,
            const_neg1,
            mul_node,
            const_shape,
            equal_node,
            const_shape2,
            where_node,
            expand_node,
        ],
        "expand_where_test",
        [inp],
        [out],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx_name = "expand_with_where_shape.onnx"
    onnx.save(model, onnx_name)
    print(f"Model exported to {onnx_name}")

    # Test
    test_input = np.ones([1, 1, 4], dtype=np.float32)
    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"input": test_input})
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs[0].shape}")
    assert outputs[0].shape == (2, 3, 4), f"Expected (2, 3, 4), got {outputs[0].shape}"
    print("Test passed!")


if __name__ == "__main__":
    main()
