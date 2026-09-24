#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: space_to_depth_multi.onnx
#
# SpaceToDepth with two input channels, so the order of the output channel
# groups is observable.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 1, 2])
    node = helper.make_node("SpaceToDepth", ["x"], ["y"], blocksize=2)
    graph = helper.make_graph([node], "space_to_depth_multi", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "space_to_depth_multi.onnx")

    data = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4)
    [out] = ReferenceEvaluator(model).run(None, {"x": data})
    print(f"y: {out.tolist()}")


if __name__ == "__main__":
    main()
