#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# Generates `crates/onnx-tests/tests/einsum/einsum_scalar_ellipsis.onnx`.
#
# A scalar operand whose term is a zero-width ellipsis: where no other term has
# an ellipsis, next to a batched operand, and on its own (`...->...`).

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def main():
    nodes = [
        helper.make_node("Einsum", ["scale", "matrix"], ["scaled"], equation="...,ij->...ij"),
        helper.make_node(
            "Einsum", ["scale", "batch"], ["batch_scaled"], equation="...,...ij->...ij"
        ),
        helper.make_node("Einsum", ["scale"], ["identity"], equation="...->..."),
    ]
    graph = helper.make_graph(
        nodes,
        "einsum_scalar_ellipsis",
        [
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("matrix", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("batch", TensorProto.FLOAT, [2, 2, 2]),
        ],
        [
            helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("batch_scaled", TensorProto.FLOAT, [2, 2, 2]),
            helper.make_tensor_value_info("identity", TensorProto.FLOAT, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, "einsum_scalar_ellipsis.onnx")

    inputs = {
        "scale": np.array(2.0, dtype=np.float32),
        "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "batch": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
    }
    scaled, batch_scaled, identity = ReferenceEvaluator(model).run(None, inputs)
    print(f"scaled: {scaled.tolist()}")
    print(f"batch_scaled: {batch_scaled.tolist()}")
    print(f"identity: {identity.tolist()}")


if __name__ == "__main__":
    main()
