# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Legacy ImageScaler; reference computed with equivalent standard Mul/Add ops."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper
from onnx.reference import ReferenceEvaluator

np.random.seed(42)
shape = [2, 3, 2, 2]
x = np.arange(24, dtype=np.float32).reshape(shape) - 12
for name, attrs in [("image_scaler", dict(scale=0.5, bias=[1., -2., 3.])), ("image_scaler_default", {})]:
    info = lambda n: h.make_tensor_value_info(n, T.FLOAT, shape)
    graph = h.make_graph([h.make_node("ImageScaler", ["x"], ["y"], **attrs)], name, [info("x")], [info("y")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 1)])
    onnx.save(model, Path(__file__).with_name(name + ".onnx"))
    scale = numpy_helper.from_array(np.array(attrs.get("scale", 1.), dtype=np.float32), "scale")
    bias = numpy_helper.from_array(np.array(attrs.get("bias", [0., 0., 0.]), dtype=np.float32).reshape(1,3,1,1), "bias")
    ref = h.make_model(h.make_graph([h.make_node("Mul", ["x", "scale"], ["scaled"]), h.make_node("Add", ["scaled", "bias"], ["y"])], "reference", [info("x")], [info("y")], [scale, bias]), opset_imports=[h.make_opsetid("", 13)])
    print(name, ReferenceEvaluator(ref).run(None, {"x": x})[0].flatten().tolist())
