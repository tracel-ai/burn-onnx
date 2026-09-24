#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: depth_to_space_multi.onnx
#
# DepthToSpace in both modes with more than one output channel, where DCR and
# CRD order the input channels differently.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 1, 2])
    dcr = helper.make_tensor_value_info("dcr", TensorProto.FLOAT, [1, 2, 2, 4])
    crd = helper.make_tensor_value_info("crd", TensorProto.FLOAT, [1, 2, 2, 4])
    nodes = [
        helper.make_node("DepthToSpace", ["x"], ["dcr"], blocksize=2, mode="DCR"),
        helper.make_node("DepthToSpace", ["x"], ["crd"], blocksize=2, mode="CRD"),
    ]
    graph = helper.make_graph(nodes, "depth_to_space_multi", [x], [dcr, crd])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "depth_to_space_multi.onnx")

    data = np.arange(16, dtype=np.float32).reshape(1, 8, 1, 2)
    dcr_out, crd_out = ReferenceEvaluator(model).run(None, {"x": data})
    print(f"dcr: {dcr_out.tolist()}")
    print(f"crd: {crd_out.tolist()}")


if __name__ == "__main__":
    main()
