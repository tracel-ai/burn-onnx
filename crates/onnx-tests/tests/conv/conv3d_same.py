# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""SAME_UPPER/LOWER, stride two and dilation, using explicit-pad reference graphs."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
root = Path(__file__).parent
kernel, strides, dilations = [2, 3, 2], [2, 2, 2], [2, 1, 2]
nodes = [h.make_node("Conv", ["x", "w"], [name], kernel_shape=kernel,
                     strides=strides, dilations=dilations, auto_pad=mode)
         for name, mode in [("upper", "SAME_UPPER"), ("lower", "SAME_LOWER")]]
model = h.make_model(h.make_graph(nodes, "conv3d_same", [h.make_tensor_value_info("x", T.FLOAT, [1, 1, 4, 5, 6])],
    [h.make_tensor_value_info(n, T.FLOAT, [1, 1, "O1", "O2", "O3"]) for n in ["upper", "lower"]],
    [nh.from_array(np.arange(12, dtype=np.float32).reshape(1, 1, 2, 3, 2) / 12, "w")]),
    opset_imports=[h.make_opsetid("", 18)])
onnx.checker.check_model(model)
onnx.save(model, root / "conv3d_same.onnx")
for case, shape in enumerate([(4, 5, 6)]):
    # ReferenceEvaluator has a SAME/stride bug; resolve pads per spec and evaluate
    # the mathematically equivalent explicit-padding Conv instead.
    reference = deepcopy(model)
    total = [max(0, ((n + s - 1) // s - 1) * s + (k - 1) * d + 1 - n)
             for n, k, s, d in zip(shape, kernel, strides, dilations)]
    for node, lower in zip(reference.graph.node, [False, True]):
        attrs = [a for a in node.attribute if a.name != "auto_pad"]
        del node.attribute[:]
        node.attribute.extend(attrs)
        begin = [(p + int(lower)) // 2 for p in total]
        node.attribute.append(h.make_attribute("pads", begin + [p-b for p,b in zip(total, begin)]))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(1, 1, *shape) / 10
    for name, output in zip(["upper", "lower"], ReferenceEvaluator(reference).run(None, {"x": x})):
        output.astype("<f4").tofile(root / f"conv3d_same_{case}_{name}.bin")
