#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate model: attention_gqa_causal.onnx
#
# Grouped-query attention (4 query heads over 2 K/V heads, each K/V head shared by
# two consecutive query heads) with
# a causal mask over a shorter query than key sequence, which ONNX aligns to the
# upper-left corner, plus an additive mask applied on top of it.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 4, 2, 4])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 3, 4])
    v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 3, 4])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 4])

    node = helper.make_node("Attention", ["q", "k", "v", "mask"], ["y"], is_causal=1)
    graph = helper.make_graph([node], "attention_gqa_causal", [q, k, v, mask], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "attention_gqa_causal.onnx")

    def seq(shape, scale):
        return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale) % 1.7 - 0.8

    feeds = {
        "q": seq([1, 4, 2, 4], 0.37),
        "k": seq([1, 2, 3, 4], 0.23),
        "v": seq([1, 2, 3, 4], 0.41),
        "mask": np.array([[0.0, -0.5, 0.3], [0.2, 0.0, -1.0]], dtype=np.float32),
    }
    [out] = ReferenceEvaluator(model).run(None, feeds)
    print(f"y: {np.round(out, 6).tolist()}")


if __name__ == "__main__":
    main()
