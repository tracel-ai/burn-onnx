#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: mod_int_fmod.onnx
#
# Integer Mod with fmod=1 (truncated, sign of the dividend) over every sign
# combination, against a tensor and against a scalar divisor, a scalar dividend,
# and a pair that broadcasts ([2, 1] against [3]).

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    x = helper.make_tensor_value_info("x", TensorProto.INT64, [8])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [8])
    d = helper.make_tensor_value_info("d", TensorProto.INT64, [])
    z = helper.make_tensor_value_info("z", TensorProto.INT64, [8])
    zs = helper.make_tensor_value_info("zs", TensorProto.INT64, [8])
    a = helper.make_tensor_value_info("a", TensorProto.INT64, [2, 1])
    b = helper.make_tensor_value_info("b", TensorProto.INT64, [3])
    zb = helper.make_tensor_value_info("zb", TensorProto.INT64, [2, 3])
    zd = helper.make_tensor_value_info("zd", TensorProto.INT64, [8])
    nodes = [
        helper.make_node("Mod", ["x", "y"], ["z"], fmod=1),
        helper.make_node("Mod", ["x", "d"], ["zs"], fmod=1),
        helper.make_node("Mod", ["a", "b"], ["zb"], fmod=1),
        helper.make_node("Mod", ["d", "y"], ["zd"], fmod=1),
    ]
    graph = helper.make_graph(nodes, "mod_int_fmod", [x, y, d, a, b], [z, zs, zb, zd])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "mod_int_fmod.onnx")

    xs = np.array([7, -7, 7, -7, 6, -6, 0, 5], dtype=np.int64)
    ys = np.array([3, 3, -3, -3, 3, -3, 4, 7], dtype=np.int64)
    ds = np.array(-4, dtype=np.int64)
    a_s = np.array([[7], [-7]], dtype=np.int64)
    bs = np.array([3, -3, 4], dtype=np.int64)
    z_out, zs_out, zb_out, zd_out = ReferenceEvaluator(model).run(
        None, {"x": xs, "y": ys, "d": ds, "a": a_s, "b": bs}
    )
    print(f"z: {z_out.tolist()}")
    print(f"zs: {zs_out.tolist()}")
    print(f"zb: {zb_out.tolist()}")
    print(f"zd: {zd_out.tolist()}")


if __name__ == "__main__":
    main()
