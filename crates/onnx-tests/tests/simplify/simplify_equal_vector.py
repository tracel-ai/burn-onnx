# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""A folded one-element comparison must remain a tensor mask."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
model = helper.make_model(helper.make_graph([
    helper.make_node("Add", ["one", "one"], ["a"]),
    helper.make_node("Equal", ["a", "b"], ["mask"]),
    helper.make_node("Where", ["mask", "x", "y"], ["out"]),
], "where_folded_equal", [
    helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
    helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [3])], [
    numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
    numpy_helper.from_array(np.array([2], dtype=np.int64), "b"),
]), opset_imports=[helper.make_opsetid("", 18)])
onnx.checker.check_model(model)
x = np.array([1, 2, 3], dtype=np.float32)
y = np.array([4, 5, 6], dtype=np.float32)
np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {"x": x, "y": y})[0], x)
onnx.save(model, Path(__file__).with_suffix(".onnx"))
