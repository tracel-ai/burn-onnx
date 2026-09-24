#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
#   "onnxruntime",
# ]
# ///

# used to generate model: maxpool1d_indices.onnx
#
# 1D MaxPool with the optional Indices output over a batch of two with two
# channels, so the per-plane offsets are exercised:
# - sym: symmetric pads (1/1), kernel 2, stride 2, storage_order=1 (same as row
#   major for a single spatial axis)
# - asym: asymmetric pads (0/2), kernel 3, stride 1
# - ceil: ceil_mode=1 where ONNX drops the last window because it would start
#   inside the trailing padding (7 long, kernel 2, stride 2, pads 1/1 -> 4, not 5)
# - dil: ceil_mode=1 with asymmetric pads (0/1) and dilation 2 where the last,
#   partial window is kept (7 long, effective kernel 3, stride 2 -> 4; floor gives 3)
# - same: auto_pad=SAME_UPPER on an input whose length is only known at run time
#   (kernel 2 pads 0 before, 1 after)
# - uint8: the sym configuration on a uint8 input
# - int8: kernel 3 on an int8 input, without Indices
# Expected values and indices follow the definition and are checked against
# onnxruntime. onnx.reference is not used: it misreads 1D pads (5 outputs for
# asym instead of 7) and fails on 1D SAME padding.

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def out_len(size, begin, end, kernel, stride, dilation, ceil):
    span = size + begin + end - ((kernel - 1) * dilation + 1)
    n = (-(-span // stride) if ceil else span // stride) + 1
    return n - 1 if ceil and (n - 1) * stride >= size + begin else n


def pool(x, kernel, stride, begin, end, dilation=1, ceil=False):
    n, c, length = x.shape
    out = out_len(length, begin, end, kernel, stride, dilation, ceil)
    values = np.zeros((n, c, out), x.dtype)
    indices = np.zeros((n, c, out), np.int64)
    for b in range(n):
        for ch in range(c):
            plane = (b * c + ch) * length
            for i in range(out):
                best = None
                for k in range(kernel):
                    p = i * stride - begin + k * dilation
                    if 0 <= p < length and (best is None or x[b, ch, p] > best[0]):
                        best = (x[b, ch, p], plane + p)
                values[b, ch, i], indices[b, ch, i] = best
    return values, indices


def main():
    f = TensorProto.FLOAT
    sym = dict(kernel_shape=[2], strides=[2], pads=[1, 1])
    configs = {
        "sym": dict(storage_order=1, **sym),
        "asym": dict(kernel_shape=[3], strides=[1], pads=[0, 2]),
        "ceil": dict(ceil_mode=1, **sym),
        "dil": dict(kernel_shape=[2], strides=[2], pads=[0, 1], dilations=[2], ceil_mode=1),
    }
    nodes = [
        helper.make_node("MaxPool", ["x"], [f"y_{n}", f"i_{n}"], **a) for n, a in configs.items()
    ]
    nodes += [
        helper.make_node(
            "MaxPool", ["xd"], ["y_same", "i_same"], kernel_shape=[2], auto_pad="SAME_UPPER"
        ),
        helper.make_node("MaxPool", ["xu"], ["y_uint8", "i_uint8"], **sym),
        helper.make_node("MaxPool", ["xi"], ["y_int8"], kernel_shape=[3]),
    ]
    shapes = {
        "sym": [2, 2, 4],
        "asym": [2, 2, 7],
        "ceil": [2, 2, 4],
        "dil": [2, 2, 4],
        "same": ["N", 2, "L"],
    }
    outputs = []
    for n, shape in shapes.items():
        outputs.append(helper.make_tensor_value_info(f"y_{n}", f, shape))
        outputs.append(helper.make_tensor_value_info(f"i_{n}", TensorProto.INT64, shape))
    outputs += [
        helper.make_tensor_value_info("y_uint8", TensorProto.UINT8, [2, 2, 4]),
        helper.make_tensor_value_info("i_uint8", TensorProto.INT64, [2, 2, 4]),
        helper.make_tensor_value_info("y_int8", TensorProto.INT8, [2, 2, 5]),
    ]
    inputs = [
        helper.make_tensor_value_info("x", f, [2, 2, 7]),
        helper.make_tensor_value_info("xd", f, ["N", 2, "L"]),
        helper.make_tensor_value_info("xu", TensorProto.UINT8, [2, 2, 7]),
        helper.make_tensor_value_info("xi", TensorProto.INT8, [2, 2, 7]),
    ]
    graph = helper.make_graph(nodes, "maxpool1d_indices", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "maxpool1d_indices.onnx")

    np.random.seed(42)
    x = np.random.permutation(28).astype(np.float32).reshape(2, 2, 7)
    xu = x.astype(np.uint8)
    xi = (x - 14).astype(np.int8)
    feeds = {"x": x, "xd": x, "xu": xu, "xi": xi}
    reference = ort.InferenceSession("maxpool1d_indices.onnx").run(None, feeds)
    expected = {
        "sym": pool(x, 2, 2, 1, 1),
        "asym": pool(x, 3, 1, 0, 2),
        "ceil": pool(x, 2, 2, 1, 1, ceil=True),
        "dil": pool(x, 2, 2, 0, 1, dilation=2, ceil=True),
        "same": pool(x, 2, 1, 0, 1),
        "uint8": pool(xu, 2, 2, 1, 1),
    }
    print(f"x: {x.astype(int).tolist()}")
    for idx, (n, (values, indices)) in enumerate(expected.items()):
        np.testing.assert_array_equal(values, reference[2 * idx])
        np.testing.assert_array_equal(indices, reference[2 * idx + 1])
        print(f"{n}: values {values.astype(int).tolist()} indices {indices.tolist()}")
    int8_values, _ = pool(xi, 3, 1, 0, 0)
    np.testing.assert_array_equal(int8_values, reference[-1])
    print(f"int8: values {int8_values.tolist()}")


if __name__ == "__main__":
    main()
