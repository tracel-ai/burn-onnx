#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate model: maxpool2d_indices_ceil.onnx
#
# MaxPool with Indices and ceil_mode=1 where ONNX drops the last window because it
# would start inside the trailing padding: symmetric pads (5 wide, kernel 2,
# stride 2, pads 1/1 -> 3, not 4) and asymmetric pads (5 wide, kernel 3, stride 3,
# pads 0/2 -> 2, not 3). Expected values and indices follow the definition.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def out_len(size, begin, end, kernel, stride):
    n = -(-(size + begin + end - kernel) // stride) + 1
    return n - 1 if (n - 1) * stride >= size + begin else n


def pool(x, kernel, stride, pads):
    top, left, bottom, right = pads
    _, _, h, w = x.shape
    oh, ow = out_len(h, top, bottom, kernel, stride), out_len(w, left, right, kernel, stride)
    values = np.zeros((1, 1, oh, ow), np.float32)
    indices = np.zeros((1, 1, oh, ow), np.int64)
    for i in range(oh):
        for j in range(ow):
            best = None
            for di in range(kernel):
                for dj in range(kernel):
                    r, c = i * stride - top + di, j * stride - left + dj
                    if 0 <= r < h and 0 <= c < w and (best is None or x[0, 0, r, c] > best[0]):
                        best = (x[0, 0, r, c], r * w + c)
            values[0, 0, i, j], indices[0, 0, i, j] = best
    return values, indices


def main():
    f = TensorProto.FLOAT
    configs = {
        "sym": dict(kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1]),
        "asym": dict(kernel_shape=[3, 3], strides=[3, 3], pads=[0, 0, 2, 2]),
    }
    nodes = [
        helper.make_node("MaxPool", ["x"], [f"y_{n}", f"i_{n}"], ceil_mode=1, **a)
        for n, a in configs.items()
    ]
    outputs = []
    for n in configs:
        outputs.append(helper.make_tensor_value_info(f"y_{n}", f, None))
        outputs.append(helper.make_tensor_value_info(f"i_{n}", TensorProto.INT64, None))
    graph = helper.make_graph(
        nodes,
        "maxpool2d_indices_ceil",
        [helper.make_tensor_value_info("x", f, [1, 1, 5, 5])],
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.save(model, "maxpool2d_indices_ceil.onnx")

    x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)[:, :, ::-1, :].copy()
    reference = ReferenceEvaluator(model).run(None, {"x": x})
    for n, a in configs.items():
        values, indices = pool(x, a["kernel_shape"][0], a["strides"][0], a["pads"])
        ref_values = reference[2 * list(configs).index(n)]
        np.testing.assert_array_equal(values, ref_values)
        print(f"{n}: values {values.tolist()} indices {indices.tolist()}")


if __name__ == "__main__":
    main()
