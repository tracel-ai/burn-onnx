#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: global_avr_pool_3d.onnx
#
# GlobalAveragePool on a rank 5 (N x C x D x H x W) input.

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 16


def main():
    node = helper.make_node("GlobalAveragePool", ["input"], ["output"])

    graph = helper.make_graph(
        [node],
        "global_avr_pool_3d",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 2, 3, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 1, 1, 1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "global_avr_pool_3d.onnx")

    x = np.arange(2 * 3 * 2 * 3 * 4, dtype=np.float32).reshape(2, 3, 2, 3, 4)
    (out,) = ReferenceEvaluator(model).run(None, {"input": x})
    print(f"output shape: {out.shape}")
    print(f"output: {out.reshape(2, 3).tolist()}")


if __name__ == "__main__":
    main()
