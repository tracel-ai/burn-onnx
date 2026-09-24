#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate model: maxpool2d_indices_same.onnx
#
# SAME_UPPER padding with an even kernel pads unevenly (0 before, 1 after).
# MaxPool with its Indices output under that padding, on a static input and on
# one whose spatial size is only known at run time (column-major indices), and a
# Conv with a runtime weight and SAME_UPPER padding on the dynamic input.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    f = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("xs", f, [1, 2, 3, 4]),
        helper.make_tensor_value_info("xd", f, ["N", 2, "H", "W"]),
        helper.make_tensor_value_info("w", f, [2, 2, 2, 2]),
    ]
    outputs = [
        helper.make_tensor_value_info("ys", f, [1, 2, 3, 4]),
        helper.make_tensor_value_info("is", TensorProto.INT64, [1, 2, 3, 4]),
        helper.make_tensor_value_info("yd", f, ["N", 2, "H", "W"]),
        helper.make_tensor_value_info("id", TensorProto.INT64, ["N", 2, "H", "W"]),
        helper.make_tensor_value_info("conv", f, ["N", 2, "H", "W"]),
    ]
    pool = dict(kernel_shape=[2, 2], auto_pad="SAME_UPPER")
    nodes = [
        helper.make_node("MaxPool", ["xs"], ["ys", "is"], **pool),
        helper.make_node("MaxPool", ["xd"], ["yd", "id"], storage_order=1, **pool),
        helper.make_node("Conv", ["xd", "w"], ["conv"], kernel_shape=[2, 2], auto_pad="SAME_UPPER"),
    ]
    graph = helper.make_graph(nodes, "maxpool2d_indices_same", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "maxpool2d_indices_same.onnx")

    np.random.seed(42)
    x = np.random.permutation(24).astype(np.float32).reshape(1, 2, 3, 4)
    w = (np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) * 0.1 - 0.7)
    ys, _, _, _, conv = ReferenceEvaluator(model).run(None, {"xs": x, "xd": x, "w": w})

    # onnx.reference computes wrong indices under auto_pad SAME (and ignores
    # storage_order), so the expected indices come from the definition: SAME_UPPER
    # with a 2x2 kernel pads 0 before and 1 after each axis.
    n, c, h, w_ = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), constant_values=-np.inf)
    row_major = np.zeros_like(x, dtype=np.int64)
    col_major = np.zeros_like(x, dtype=np.int64)
    for b in range(n):
        for ch in range(c):
            plane = (b * c + ch) * h * w_
            for i in range(h):
                for j in range(w_):
                    window = padded[b, ch, i : i + 2, j : j + 2]
                    di, dj = np.unravel_index(np.argmax(window), window.shape)
                    r, q = i + di, j + dj
                    row_major[b, ch, i, j] = plane + r * w_ + q
                    col_major[b, ch, i, j] = plane + q * h + r
    print(f"x: {x.astype(int).tolist()}")
    print(f"ys: {ys.astype(int).tolist()}")
    print(f"is: {row_major.tolist()}")
    print(f"id: {col_major.tolist()}")
    print(f"conv: {np.round(conv, 5).tolist()}")


if __name__ == "__main__":
    main()
