#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: onnx-tests/tests/reduce/reduce_max.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    input_shape = [1, 1, 2, 4]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)

    # ReduceMax, keepdims=0, axes=None (reduce all)
    node1 = helper.make_node("ReduceMax", ["x"], ["out1"], keepdims=0)
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [])

    # ReduceMax, keepdims=0, axes=[] (reduce all)
    node2 = helper.make_node("ReduceMax", ["x"], ["out2"], keepdims=0)
    # Add empty axes attribute (onnx helper can't create it from [])
    node2.attribute.append(onnx.AttributeProto(name="axes", type=7))  # INTS type
    out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [])

    # ReduceMax, keepdims=1, axes=[1]
    node3 = helper.make_node("ReduceMax", ["x"], ["out3"], axes=[1], keepdims=1)
    out3 = helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1, 1, 2, 4])

    # ReduceMax, keepdims=1, axes=[-1]
    node4 = helper.make_node("ReduceMax", ["x"], ["out4"], axes=[-1], keepdims=1)
    out4 = helper.make_tensor_value_info("out4", TensorProto.FLOAT, [1, 1, 2, 1])

    # ReduceMax, keepdims=0, axes=[0]
    node5 = helper.make_node("ReduceMax", ["x"], ["out5"], axes=[0], keepdims=0)
    out5 = helper.make_tensor_value_info("out5", TensorProto.FLOAT, [1, 2, 4])

    # ReduceMax, keepdims=0, axes=[0, 2]
    node6 = helper.make_node("ReduceMax", ["x"], ["out6"], axes=[0, 2], keepdims=0)
    out6 = helper.make_tensor_value_info("out6", TensorProto.FLOAT, [1, 4])

    graph = helper.make_graph(
        [node1, node2, node3, node4, node5, node6],
        "reduce_max_test",
        [x],
        [out1, out2, out3, out4, out5, out6],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)])

    onnx_name = "reduce_max.onnx"
    onnx.save(model, onnx_name)
    print(f"Finished exporting model to {onnx_name}")

    # Test
    test_input = np.array([[[[1.0, 4.0, 9.0, 25.0], [2.0, 5.0, 10.0, 26.0]]]]).astype(np.float32)
    session = ReferenceEvaluator(onnx_name)
    outputs = session.run(None, {"x": test_input})
    print(f"Test input data:\n{test_input}")
    for i, o in enumerate(outputs):
        print(f"Output {i}: {o}")


if __name__ == "__main__":
    main()
