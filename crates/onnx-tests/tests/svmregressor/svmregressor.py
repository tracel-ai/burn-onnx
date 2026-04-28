#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
ONNX SVMRegressor operator test model generator.

Generates two SVMRegressor models:
  - svmregressor.onnx        : LINEAR kernel
  - svmregressor_rbf.onnx    : RBF kernel (non-default configuration)

Expected outputs are computed with onnx.reference.ReferenceEvaluator.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def make_model(kernel_type, support_vectors, coefficients, rho,
               kernel_params=None, n_features=2, batch_size=3, name="svmregressor_test"):
    n_supports = len(support_vectors) // n_features
    attrs = dict(
        coefficients=coefficients.tolist(),
        kernel_type=kernel_type,
        n_supports=n_supports,
        rho=rho.tolist(),
        support_vectors=support_vectors.tolist(),
    )
    if kernel_params is not None:
        attrs["kernel_params"] = kernel_params.tolist()

    svm_node = helper.make_node(
        'SVMRegressor', inputs=['X'], outputs=['Y'],
        domain='ai.onnx.ml', **attrs,
    )
    graph = helper.make_graph(
        [svm_node], name,
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [batch_size, n_features])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [batch_size])],
    )
    return helper.make_model(
        graph,
        producer_name='svmregressor-test',
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )


def main():
    np.random.seed(42)
    batch_size = 3
    n_features = 2

    input_data = np.random.randn(batch_size, n_features).astype(np.float32)

    # ── Model 1: LINEAR kernel ────────────────────────────────────────────────
    sv_linear = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).flatten()
    coef_linear = np.array([1.0, -0.5], dtype=np.float32)
    rho_linear = np.array([0.5], dtype=np.float32)

    model_linear = make_model(
        'LINEAR', sv_linear, coef_linear, rho_linear,
        n_features=n_features, batch_size=batch_size,
    )
    onnx.save(model_linear, 'svmregressor.onnx')

    ref_linear = ReferenceEvaluator(model_linear)
    (output_linear,) = ref_linear.run(None, {"X": input_data})
    print(f"LINEAR input:  {input_data.tolist()}")
    print(f"LINEAR output: {output_linear.tolist()}")

    # ── Model 2: RBF kernel ───────────────────────────────────────────────────
    sv_rbf = np.array([[0.5, 1.5], [2.0, 0.5]], dtype=np.float32).flatten()
    coef_rbf = np.array([0.8, -0.3], dtype=np.float32)
    rho_rbf = np.array([0.1], dtype=np.float32)
    # kernel_params = [gamma, coef0, degree] — gamma=0.5 for RBF
    kernel_params_rbf = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    model_rbf = make_model(
        'RBF', sv_rbf, coef_rbf, rho_rbf,
        kernel_params=kernel_params_rbf,
        n_features=n_features, batch_size=batch_size,
        name="svmregressor_rbf_test",
    )
    onnx.save(model_rbf, 'svmregressor_rbf.onnx')

    ref_rbf = ReferenceEvaluator(model_rbf)
    (output_rbf,) = ref_rbf.run(None, {"X": input_data})
    print(f"RBF input:  {input_data.tolist()}")
    print(f"RBF output: {output_rbf.tolist()}")


if __name__ == '__main__':
    main()
