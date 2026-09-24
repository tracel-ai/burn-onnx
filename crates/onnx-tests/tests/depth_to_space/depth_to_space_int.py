#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: depth_to_space_int.onnx
#
# DepthToSpace on int64 data in both modes (burn's PixelShuffle is float-only),
# with more than one output channel so DCR and CRD differ.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.INT64, [1, 8, 1, 2])
    dcr = helper.make_tensor_value_info("dcr", TensorProto.INT64, [1, 2, 2, 4])
    crd = helper.make_tensor_value_info("crd", TensorProto.INT64, [1, 2, 2, 4])
    nodes = [
        helper.make_node("DepthToSpace", ["x"], ["dcr"], blocksize=2, mode="DCR"),
        helper.make_node("DepthToSpace", ["x"], ["crd"], blocksize=2, mode="CRD"),
    ]
    graph = helper.make_graph(nodes, "depth_to_space_int", [x], [dcr, crd])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "depth_to_space_int.onnx")

    data = np.arange(16, dtype=np.int64).reshape(1, 8, 1, 2)
    dcr_out, crd_out = ReferenceEvaluator(model).run(None, {"x": data})
    print(f"dcr: {dcr_out.tolist()}")
    print(f"crd: {crd_out.tolist()}")


if __name__ == "__main__":
    main()
