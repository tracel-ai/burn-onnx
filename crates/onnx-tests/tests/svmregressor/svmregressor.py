#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = ["numpy", "onnx"]
# ///
"""
ONNX SVMRegressor operator test model generator.

Generates six SVMRegressor models:
  - svmregressor.onnx              : LINEAR kernel
  - svmregressor_rbf.onnx          : RBF kernel (gamma=0.5)
  - svmregressor_poly.onnx         : POLY kernel (gamma=1, coef0=1, degree=2)
  - svmregressor_sigmoid.onnx      : SIGMOID kernel (gamma=0.5, coef0=0.1)
  - svmregressor_logistic.onnx     : LINEAR kernel + LOGISTIC post-transform
  - svmregressor_softmax_zero.onnx : LINEAR kernel + SOFTMAX_ZERO post-transform

Expected outputs are computed with onnx.reference.ReferenceEvaluator.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


def make_model(kernel_type, support_vectors, coefficients, rho,
               kernel_params=None, post_transform=None, n_features=2,
               batch_size=3, name="svmregressor_test"):
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
    if post_transform is not None:
        attrs["post_transform"] = post_transform

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

    sv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).flatten()
    coef = np.array([1.0, -0.5], dtype=np.float32)
    rho = np.array([0.5], dtype=np.float32)

    # ── Model 1: LINEAR kernel ────────────────────────────────────────────────
    model_linear = make_model('LINEAR', sv, coef, rho,
                              n_features=n_features, batch_size=batch_size)
    onnx.save(model_linear, 'svmregressor.onnx')
    (output_linear,) = ReferenceEvaluator(model_linear).run(None, {"X": input_data})
    print(f"LINEAR output: {output_linear.tolist()}")

    # ── Model 2: RBF kernel ───────────────────────────────────────────────────
    sv_rbf = np.array([[0.5, 1.5], [2.0, 0.5]], dtype=np.float32).flatten()
    coef_rbf = np.array([0.8, -0.3], dtype=np.float32)
    rho_rbf = np.array([0.1], dtype=np.float32)
    kp_rbf = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    model_rbf = make_model('RBF', sv_rbf, coef_rbf, rho_rbf,
                           kernel_params=kp_rbf, n_features=n_features,
                           batch_size=batch_size, name="svmregressor_rbf_test")
    onnx.save(model_rbf, 'svmregressor_rbf.onnx')
    (output_rbf,) = ReferenceEvaluator(model_rbf).run(None, {"X": input_data})
    print(f"RBF output: {output_rbf.tolist()}")

    # ── Model 3: POLY kernel (gamma=1, coef0=1, degree=2) ────────────────────
    kp_poly = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    model_poly = make_model('POLY', sv, coef, rho,
                            kernel_params=kp_poly, n_features=n_features,
                            batch_size=batch_size, name="svmregressor_poly_test")
    onnx.save(model_poly, 'svmregressor_poly.onnx')
    (output_poly,) = ReferenceEvaluator(model_poly).run(None, {"X": input_data})
    print(f"POLY output: {output_poly.tolist()}")

    # ── Model 4: SIGMOID kernel (gamma=0.5, coef0=0.1) ───────────────────────
    kp_sigmoid = np.array([0.5, 0.1, 0.0], dtype=np.float32)
    model_sigmoid = make_model('SIGMOID', sv, coef, rho,
                               kernel_params=kp_sigmoid, n_features=n_features,
                               batch_size=batch_size, name="svmregressor_sigmoid_test")
    onnx.save(model_sigmoid, 'svmregressor_sigmoid.onnx')
    (output_sigmoid,) = ReferenceEvaluator(model_sigmoid).run(None, {"X": input_data})
    print(f"SIGMOID output: {output_sigmoid.tolist()}")

    # ── Model 5: LINEAR + LOGISTIC post-transform ────────────────────────────
    model_logistic = make_model('LINEAR', sv, coef, rho,
                                post_transform='LOGISTIC',
                                n_features=n_features, batch_size=batch_size,
                                name="svmregressor_logistic_test")
    onnx.save(model_logistic, 'svmregressor_logistic.onnx')
    # ReferenceEvaluator doesn't implement LOGISTIC; compute sigmoid of LINEAR output manually.
    output_logistic = (1.0 / (1.0 + np.exp(-output_linear))).flatten()
    print(f"LOGISTIC output: {output_logistic.tolist()}")

    # ── Model 6: LINEAR + SOFTMAX_ZERO post-transform ────────────────────────
    # SOFTMAX_ZERO appends an implicit zero class, so for single-target output y
    # the result is exp(y) / (exp(y) + 1) = sigmoid(y). Hand-compute since
    # ReferenceEvaluator doesn't implement SOFTMAX_ZERO either.
    model_softmax_zero = make_model('LINEAR', sv, coef, rho,
                                    post_transform='SOFTMAX_ZERO',
                                    n_features=n_features, batch_size=batch_size,
                                    name="svmregressor_softmax_zero_test")
    onnx.save(model_softmax_zero, 'svmregressor_softmax_zero.onnx')
    output_softmax_zero = (np.exp(output_linear) / (np.exp(output_linear) + 1.0)).flatten()
    print(f"SOFTMAX_ZERO output: {output_softmax_zero.tolist()}")


if __name__ == '__main__':
    main()
