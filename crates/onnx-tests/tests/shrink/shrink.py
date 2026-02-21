#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: shrink.onnx

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 9


def main():
    node0 = helper.make_node("Shrink", ["inp_0"], ["out_0"], lambd=1.5)
    node1 = helper.make_node("Shrink", ["inp_1"], ["out_1"], lambd=1.5, bias=1.5)

    inp_no_bias = helper.make_tensor_value_info("inp_0", TensorProto.FLOAT, [5, 5])
    inp_with_bias = helper.make_tensor_value_info("inp_1", TensorProto.FLOAT, [5, 5])

    out_no_bias = helper.make_tensor_value_info("out_0", TensorProto.FLOAT, [5, 5])
    out_with_bias = helper.make_tensor_value_info("out_1", TensorProto.FLOAT, [5, 5])

    graph = helper.make_graph(
        [node0, node1],
        "torch_jit",
        [inp_no_bias, inp_with_bias],
        [out_no_bias, out_with_bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "shrink.onnx")
    print("Finished exporting model to shrink.onnx")

    # Create input data and run inference
    input_arr = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-3.0, -2.5, -0.5, 0.5, 3.0],
            [-1.5, 0.0, 1.5, 2.5, 3.5],
            [-2.2, -1.1, 0.0, 1.1, 2.2],
            [-4.0, -2.0, 0.0, 2.0, 4.0],
        ],
        dtype=np.float32,
    )

    print("\nInput:")
    print(input_arr)

    # Run inference using ReferenceEvaluator
    ref_eval = ReferenceEvaluator(model)
    outputs = list(ref_eval.run(None, {"inp_0": input_arr, "inp_1": input_arr}))

    print("\nOutput (no bias):")
    print(outputs[0])
    print("\nOutput (with bias):")
    print(outputs[1])


if __name__ == "__main__":
    main()
