# /// script
# dependencies = ["onnx==1.23.0", "numpy"]
# ///
"""Dilated 1D/2D/3D average pools, explicit pads and dynamic SAME padding."""

from pathlib import Path
from copy import deepcopy
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
root = Path(__file__).parent
for rank in [1, 2, 3]:
    nodes = []
    for include, ceil in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        name = (
            f"{'include' if include else 'exclude'}_pad_{'ceil' if ceil else 'floor'}"
        )
        nodes.append(
            h.make_node(
                "AveragePool",
                ["x"],
                [name],
                kernel_shape=[2] * rank,
                strides=[2] * rank,
                dilations=[2] * rank,
                pads=[1] * (2 * rank),
                count_include_pad=include,
                ceil_mode=ceil,
            )
        )
    for mode in ["SAME_UPPER", "SAME_LOWER"]:
        nodes.append(
            h.make_node(
                "AveragePool",
                ["x"],
                [mode.lower()],
                kernel_shape=[2] * rank,
                strides=[2] * rank,
                dilations=[2] * rank,
                auto_pad=mode,
                count_include_pad=0,
            )
        )
    model = h.make_model(
        h.make_graph(
            nodes,
            "dilated_pool",
            [
                h.make_tensor_value_info(
                    "x", T.FLOAT, [1, 1] + [f"d{i}" for i in range(rank)]
                )
            ],
            [
                h.make_tensor_value_info(n.output[0], T.FLOAT, [1, 1] + [None] * rank)
                for n in nodes
            ],
        ),
        opset_imports=[h.make_opsetid("", 19)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, root / f"avg_pool{rank}d_dilations.onnx")
    for size in [6, 7]:
        x = np.arange(size**rank, dtype=np.float32).reshape([1, 1] + [size] * rank) / 10
        reference = deepcopy(model)
        # Resolve SAME pads explicitly to avoid evaluator-specific auto_pad rules.
        total = max(0, ((size + 1) // 2 - 1) * 2 + 3 - size)
        for node in reference.graph.node[-2:]:
            lower = node.output[0] == "same_lower"
            begin = (total + int(lower)) // 2
            attrs = [a for a in node.attribute if a.name != "auto_pad"]
            del node.attribute[:]
            node.attribute.extend(attrs)
            node.attribute.append(
                h.make_attribute("pads", [begin] * rank + [total - begin] * rank)
            )
        for node, y in zip(nodes, ReferenceEvaluator(reference).run(None, {"x": x})):
            # Paste these reference values directly into the Rust integration test.
            shape = ", ".join(str(d) for d in y.shape)
            values = ", ".join(repr(float(f"{v:.7g}")) for v in y.flat)
            print(f"// rank {rank}, size {size}, {node.output[0]}")
            print(f"(&[{values}], [{shape}]),")
