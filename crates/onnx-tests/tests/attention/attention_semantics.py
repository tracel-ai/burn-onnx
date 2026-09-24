#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.23.0",
#   "numpy",
# ]
# ///

# used to generate models: attention_kv_cache_causal.onnx,
# attention_padding_mask_causal.onnx, attention_softcap_bias.onnx
#
# Attention semantics that depend on the current ONNX definition:
# - a causal decode step over a KV cache, where the mask is offset by the cached
#   length (the new query sees every cached key);
# - a [batch, 1, 1, keys] bool padding mask combined with a non-square causal
#   mask, with one batch hiding every key (those rows come out as zeros). The
#   onnx 1.23 reference sizes the causal mask from the attention mask's shape, so
#   a broadcast [.., 1, keys] mask would get a causal mask for a single query;
#   the expected output comes from the same mask materialized over the queries;
# - softcap with a large additive mask, where the softcap comes first.

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def seq(shape, scale):
    return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale) % 1.7 - 0.8


def build(name, inputs, outputs, node_inputs, node_outputs, feeds, reference=None, **attrs):
    node = helper.make_node("Attention", node_inputs, node_outputs, **attrs)
    graph = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, f"{name}.onnx")
    if reference is not None:
        # Same computation with the inputs the reference needs to get it right.
        inputs, feeds = reference
        graph = helper.make_graph(
            [node],
            name,
            [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in outputs],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
        model.ir_version = 10
    for out_name, value in zip(node_outputs, ReferenceEvaluator(model).run(None, feeds)):
        print(f"{name} {out_name}: {np.round(value, 6).tolist()}")


def main():
    f = TensorProto.FLOAT
    build(
        "attention_kv_cache_causal",
        [
            ("q", f, [1, 2, 1, 4]),
            ("k", f, [1, 2, 1, 4]),
            ("v", f, [1, 2, 1, 4]),
            ("past_k", f, [1, 2, 2, 4]),
            ("past_v", f, [1, 2, 2, 4]),
        ],
        [("y", [1, 2, 1, 4]), ("present_k", [1, 2, 3, 4]), ("present_v", [1, 2, 3, 4])],
        ["q", "k", "v", "", "past_k", "past_v"],
        ["y", "present_k", "present_v"],
        {
            "q": seq([1, 2, 1, 4], 0.37),
            "k": seq([1, 2, 1, 4], 0.23),
            "v": seq([1, 2, 1, 4], 0.41),
            "past_k": seq([1, 2, 2, 4], 0.29),
            "past_v": seq([1, 2, 2, 4], 0.53),
        },
        is_causal=1,
    )
    build(
        "attention_padding_mask_causal",
        [
            ("q", f, [2, 2, 2, 4]),
            ("k", f, [2, 2, 3, 4]),
            ("v", f, [2, 2, 3, 4]),
            ("mask", TensorProto.BOOL, [2, 1, 1, 3]),
        ],
        [("y", [2, 2, 2, 4])],
        ["q", "k", "v", "mask"],
        ["y"],
        {
            "q": seq([2, 2, 2, 4], 0.37),
            "k": seq([2, 2, 3, 4], 0.23),
            "v": seq([2, 2, 3, 4], 0.41),
            "mask": np.array([[[[True, True, False]]], [[[False, False, False]]]]),
        },
        reference=(
            [
                ("q", f, [2, 2, 2, 4]),
                ("k", f, [2, 2, 3, 4]),
                ("v", f, [2, 2, 3, 4]),
                ("mask", TensorProto.BOOL, [2, 1, 2, 3]),
            ],
            {
                "q": seq([2, 2, 2, 4], 0.37),
                "k": seq([2, 2, 3, 4], 0.23),
                "v": seq([2, 2, 3, 4], 0.41),
                "mask": np.broadcast_to(
                    np.array([[[[True, True, False]]], [[[False, False, False]]]]),
                    (2, 1, 2, 3),
                ).copy(),
            },
        ),
        is_causal=1,
    )
    build(
        "attention_softcap_bias",
        [
            ("q", f, [1, 1, 2, 4]),
            ("k", f, [1, 1, 3, 4]),
            ("v", f, [1, 1, 3, 4]),
            ("mask", f, [2, 3]),
        ],
        [("y", [1, 1, 2, 4])],
        ["q", "k", "v", "mask"],
        ["y"],
        {
            "q": seq([1, 1, 2, 4], 1.37),
            "k": seq([1, 1, 3, 4], 1.23),
            "v": seq([1, 1, 3, 4], 0.41),
            "mask": np.array([[0.0, 3.0, -2.0], [2.5, 0.0, 1.0]], dtype=np.float32),
        },
        softcap=1.0,
    )


if __name__ == "__main__":
    main()
