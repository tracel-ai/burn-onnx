# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
def branch(name, input_name, dims):
    return h.make_graph([h.make_node("Identity", [input_name], [name])], name, [], [h.make_tensor_value_info(name,T.FLOAT,dims)])
m = h.make_model(h.make_graph([
 h.make_node("If",["cond"],["selected"],then_branch=branch("a","x",[1,3]),else_branch=branch("b","y",[3])),
 h.make_node("MatMul",["selected","w"],["out"]),
],"alignment",[h.make_tensor_value_info("cond",T.BOOL,[]),h.make_tensor_value_info("x",T.FLOAT,[1,3]),h.make_tensor_value_info("y",T.FLOAT,[3]),h.make_tensor_value_info("w",T.FLOAT,[3,2])],[h.make_tensor_value_info("out",T.FLOAT,[None,2])]),opset_imports=[h.make_opsetid("",18)])
onnx.checker.check_model(m)
for cond in (True,False):
    out = ReferenceEvaluator(m).run(None,dict(cond=np.array(cond),x=np.array([[1,2,3]],dtype=np.float32),y=np.array([4,5,6],dtype=np.float32),w=np.ones((3,2),dtype=np.float32)))[0]
    np.testing.assert_array_equal(out.ravel(),[6,6] if cond else [15,15])
onnx.save(m,Path(__file__).with_suffix(".onnx"))
