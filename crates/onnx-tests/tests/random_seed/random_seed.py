# /// script
# dependencies = ["onnx==1.23.0", "numpy"]
# ///
"""Seed reproducibility is backend-local; reference RNG sequences need not match Burn."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
root = Path(__file__).parent
for op in ["RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike", "Bernoulli", "Multinomial"]:
    name = "seed_" + op.lower()
    has_input = op not in ["RandomNormal", "RandomUniform"]
    dtype = T.INT64 if op in ["Bernoulli", "Multinomial"] else T.DOUBLE
    attrs = {"dtype": dtype}
    if not has_input:
        attrs["shape"] = [2, 64]
    if op == "Multinomial":
        attrs["sample_size"] = 64
    nodes = [h.make_node(op, ["x"] if has_input else [], [n], seed=seed, **attrs)
             for n,seed in [("a",42.5),("b",42.5),("c",42.75)]]
    model = h.make_model(h.make_graph(nodes, name,
        [h.make_tensor_value_info("x", T.FLOAT, [2,64])] if has_input else [],
        [h.make_tensor_value_info(n, dtype, [2,64]) for n in ["a","b","c"]]),
        opset_imports=[h.make_opsetid("", 22)])
    onnx.checker.check_model(model)
    onnx.save(model, root / f"{name}.onnx")
    # Check the reference contract (shape/dtype). Exact samples are deliberately
    # not persisted: ONNX does not prescribe a cross-backend RNG algorithm.
    if op == "Multinomial":
        # ReferenceEvaluator does not implement Multinomial. Rust tests verify
        # its reproducibility and category range directly.
        continue
    reference = ReferenceEvaluator(model).run(None, {"x": np.full([2,64], .5, dtype=np.float32)} if has_input else {})
    assert all(x.shape == (2,64) for x in reference)
    assert all(x.dtype == (np.int64 if dtype == T.INT64 else np.float64) for x in reference)

for op in ["RandomNormalLike", "RandomUniformLike"]:
    name = "inherit_" + op.lower()
    model = h.make_model(h.make_graph(
        [h.make_node(op, ["x"], ["y"], seed=42.5)], name,
        [h.make_tensor_value_info("x", T.DOUBLE, [2,64])],
        [h.make_tensor_value_info("y", T.DOUBLE, [2,64])]),
        opset_imports=[h.make_opsetid("",22)])
    onnx.checker.check_model(model)
    assert ReferenceEvaluator(model).run(None, {"x": np.zeros([2,64], dtype=np.float64)})[0].dtype == np.float64
    onnx.save(model, root / f"{name}.onnx")
