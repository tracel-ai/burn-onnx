# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
def info(name, dtype, shape): return h.make_tensor_value_info(name,dtype,shape)
def save(name, nodes, inputs, outputs, feeds, initializers=(), opset=21):
    m=h.make_model(h.make_graph(nodes,name,inputs,outputs,list(initializers)),opset_imports=[h.make_opsetid("",opset)])
    onnx.checker.check_model(m)
    print(name, [v.tolist() for v in ReferenceEvaluator(m).run(None,feeds)])
    onnx.save(m,Path(__file__).with_name(name+".onnx"))
save("dequantize_half_center",[h.make_node("DequantizeLinear",["x","scale","zero"],["y"])],
 [info("x",T.UINT16,[1]),info("scale",T.FLOAT16,[]),info("zero",T.UINT16,[])],[info("y",T.FLOAT16,[1])],
 dict(x=np.array([2049],np.uint16),scale=np.array(1,np.float16),zero=np.array(2048,np.uint16)))
save("dynamic_scalar",[h.make_node("DynamicQuantizeLinear",["x"],["y","scale","zero"])],[info("x",T.FLOAT,[])],
 [info("y",T.UINT8,[]),info("scale",T.FLOAT,[]),info("zero",T.UINT8,[])],dict(x=np.array(2,np.float32)))
save("qmatmul_shared",[h.make_node("QLinearMatMul",["a","scale","zero","a","scale","zero","scale","zero"],["y"])],
 [info("a",T.UINT8,[1,2,2]),info("scale",T.FLOAT,[1]),info("zero",T.UINT8,[1])],[info("y",T.UINT8,[1,2,2])],
 dict(a=np.array([[[1,2],[3,4]]],np.uint8),scale=np.array([1],np.float32),zero=np.array([0],np.uint8)))
save("dynamic_matmul",[
 h.make_node("DynamicQuantizeLinear",["x"],["a","as","az"]),
 h.make_node("QLinearMatMul",["a","as","az","b","bs","bz","bs","bz"],["y"]),
 h.make_node("QLinearMatMul",["a","as","bz","b","bs","bz","bs","bz"],["host_zero"]),
 h.make_node("QLinearMatMul",["a","bs","az","b","bs","bz","bs","bz"],["host_scale"]),
 ],[info("x",T.FLOAT,[1,2,2])],[info("y",T.UINT8,[1,2,2]),info("host_zero",T.UINT8,[1,2,2]),info("host_scale",T.UINT8,[1,2,2])],dict(x=np.array([[[0,1],[2,3]]],np.float32)),
 [nh.from_array(np.array([[[1,0],[0,1]]],np.uint8),"b"),nh.from_array(np.array(1,np.float32),"bs"),nh.from_array(np.array(0,np.uint8),"bz")])
