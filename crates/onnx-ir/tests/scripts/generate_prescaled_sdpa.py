#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
# ]
# ///

"""
Generate an ONNX model with a pre-scaled SDPA (Scaled Dot-Product Attention) pattern.

This is the pattern emitted by some PyTorch exporters (e.g., RF-DETR) where sqrt(scale)
is applied to Q and K individually BEFORE the QK MatMul, and K's transpose combines
head-split and key-transpose into a single Transpose([0,2,3,1]) op:

    Q -> Reshape -> Transpose([0,2,1,3]) -> Mul(sqrt_scale) --\
                                                                MatMul -> Softmax -> MatMul -> Transpose -> Reshape
    K -> Reshape -> Transpose([0,2,3,1]) -> Mul(sqrt_scale) --/                      V --^
    V -> Reshape -> Transpose([0,2,1,3]) -------------------------------------------|

The simplification pass should coalesce this into a single Attention node.
"""

import onnx
from onnx import helper, TensorProto


def create_prescaled_sdpa_model():
    """Create an ONNX model with a pre-scaled SDPA pattern."""
    B, S, num_heads, head_dim = 1, 4, 2, 4
    hidden = num_heads * head_dim  # 8

    # Graph inputs: Q, K, V already projected [B, S, hidden]
    q_input = helper.make_tensor_value_info("q", TensorProto.FLOAT, [B, S, hidden])
    k_input = helper.make_tensor_value_info("k", TensorProto.FLOAT, [B, S, hidden])
    v_input = helper.make_tensor_value_info("v", TensorProto.FLOAT, [B, S, hidden])

    # Output: [B, S, hidden]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [B, S, hidden])

    # Constants
    reshape_shape = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [4], [B, S, num_heads, head_dim]
    )
    output_shape = helper.make_tensor(
        "output_shape", TensorProto.INT64, [3], [B, S, hidden]
    )
    # head_dim as float for scale computation
    head_dim_tensor = helper.make_tensor(
        "head_dim_val", TensorProto.FLOAT, [1], [float(head_dim)]
    )
    one_tensor = helper.make_tensor(
        "one_val", TensorProto.FLOAT, [1], [1.0]
    )

    nodes = [
        # Reshape Q, K, V: [B, S, hidden] -> [B, S, num_heads, head_dim]
        helper.make_node("Reshape", ["q", "reshape_shape"], ["q_reshaped"], name="reshape_q"),
        helper.make_node("Reshape", ["k", "reshape_shape"], ["k_reshaped"], name="reshape_k"),
        helper.make_node("Reshape", ["v", "reshape_shape"], ["v_reshaped"], name="reshape_v"),

        # V transpose: [B, S, H, D] -> [B, H, S, D] (standard head-split)
        helper.make_node(
            "Transpose", ["v_reshaped"], ["v_transposed"],
            name="transpose_v", perm=[0, 2, 1, 3]
        ),

        # Q transpose: [B, S, H, D] -> [B, H, S, D] (standard head-split)
        helper.make_node(
            "Transpose", ["q_reshaped"], ["q_transposed"],
            name="transpose_q", perm=[0, 2, 1, 3]
        ),

        # K transpose: [B, S, H, D] -> [B, H, D, S] (combined head-split + key-transpose)
        helper.make_node(
            "Transpose", ["k_reshaped"], ["k_transposed"],
            name="transpose_k", perm=[0, 2, 3, 1]
        ),

        # Compute sqrt_scale = sqrt(1 / sqrt(head_dim))
        # This mirrors the RF-DETR pattern where the scale is computed dynamically
        helper.make_node("Sqrt", ["head_dim_val"], ["sqrt_head_dim"], name="sqrt_hd"),
        helper.make_node("Div", ["one_val", "sqrt_head_dim"], ["inv_sqrt_hd"], name="div_scale"),
        # Duplicate Sqrt (as seen in real models before CSE merges them)
        helper.make_node("Sqrt", ["inv_sqrt_hd"], ["sqrt_scale_q"], name="sqrt_scale_q"),
        helper.make_node("Sqrt", ["inv_sqrt_hd"], ["sqrt_scale_k"], name="sqrt_scale_k"),

        # Pre-scale Q and K
        helper.make_node("Mul", ["q_transposed", "sqrt_scale_q"], ["q_scaled"], name="mul_q"),
        helper.make_node("Mul", ["k_transposed", "sqrt_scale_k"], ["k_scaled"], name="mul_k"),

        # QK MatMul: [B, H, S, D] @ [B, H, D, S] = [B, H, S, S]
        helper.make_node("MatMul", ["q_scaled", "k_scaled"], ["qk"], name="qk_matmul"),

        # Softmax on last dim
        helper.make_node("Softmax", ["qk"], ["attn_weights"], name="softmax", axis=-1),

        # Scores @ V: [B, H, S, S] @ [B, H, S, D] = [B, H, S, D]
        helper.make_node(
            "MatMul", ["attn_weights", "v_transposed"], ["attn_out"],
            name="sv_matmul"
        ),

        # Output transpose: [B, H, S, D] -> [B, S, H, D]
        helper.make_node(
            "Transpose", ["attn_out"], ["out_transposed"],
            name="transpose_out", perm=[0, 2, 1, 3]
        ),

        # Reshape back: [B, S, H, D] -> [B, S, hidden]
        helper.make_node(
            "Reshape", ["out_transposed", "output_shape"], ["output"],
            name="reshape_out"
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "prescaled_sdpa",
        [q_input, k_input, v_input],
        [output],
        initializer=[reshape_shape, output_shape, head_dim_tensor, one_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="onnx-ir-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )

    onnx.checker.check_model(model)
    return model


def main():
    model = create_prescaled_sdpa_model()
    output_path = "../fixtures/prescaled_sdpa.onnx"
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    print(f"\nModel info:")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"  Outputs: {[out.name for out in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    for node in model.graph.node:
        print(f"    - {node.op_type} ({node.name})")


if __name__ == "__main__":
    main()
