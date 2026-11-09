# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Indices smaller on both non-axis dimensions, including negative axis/indices."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
root = Path(__file__).parent
data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
indices = np.array([[[0, -1], [1, 0]]], dtype=np.int64)
updates = np.array([[[30, -10], [-4, 40]]], dtype=np.float32)
names = ["gather", "none", "add", "max", "min"]
nodes = [h.make_node("GatherElements", ["data", "indices"], ["gather"], axis=-2)]
for reduction in names[1:]:
    nodes.append(h.make_node("ScatterElements", ["data", "indices", "updates"], [reduction],
                             axis=1, reduction=reduction))
model = h.make_model(h.make_graph(nodes, "smaller_indices",
    [h.make_tensor_value_info(n, t, list(v.shape)) for n,t,v in
     [("data", T.FLOAT, data), ("indices", T.INT64, indices), ("updates", T.FLOAT, updates)]],
    [h.make_tensor_value_info(n, T.FLOAT, [1,2,2] if n == "gather" else [2,3,4]) for n in names]),
    opset_imports=[h.make_opsetid("", 18)])
onnx.checker.check_model(model)
onnx.save(model, root / "smaller_indices.onnx")
# ReferenceEvaluator wrongly requires equal non-axis dimensions for GatherElements.
# Slice those dimensions first in the reference graph; this preserves every
# indexed coordinate and leaves the actual test model unchanged.
reference = deepcopy(model)
reference.graph.node[0].input[0] = "gather_data"
reference.graph.node[0].attribute[0].i = 1
reference.graph.input.append(h.make_tensor_value_info("gather_data", T.FLOAT, [1,3,2]))
for name, output in zip(names, ReferenceEvaluator(reference).run(None,
        dict(data=data, indices=indices, updates=updates, gather_data=data[:1, :, :2]))):
    output.astype("<f4").tofile(root / f"smaller_indices_{name}.bin")
