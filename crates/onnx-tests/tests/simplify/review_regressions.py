# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Runtime shape and comparison-domain regressions, checked against ONNX."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
def info(name, dtype, shape): return h.make_tensor_value_info(name,dtype,shape)
def save(name,nodes,inputs,outputs,feeds,initializers=()):
    model=h.make_model(h.make_graph(nodes,name,inputs,outputs,list(initializers)),opset_imports=[h.make_opsetid("",18)])
    onnx.checker.check_model(model)
    # Burn accepts singleton tensors for Range bounds; the reference requires scalars.
    reference_feeds = {k: v.reshape(()) for k, v in feeds.items()} if name == "simplify_range_name" else feeds
    print(name,[v.tolist() for v in ReferenceEvaluator(model).run(None,reference_feeds)])
    onnx.save(model,Path(__file__).with_name(name+".onnx"))
save("simplify_runtime_size",[h.make_node("Size",["x"],["count"]),h.make_node("Equal",["count","six"],["matches"])],
 [info("x",T.FLOAT,[2,3])],[info("count",T.INT64,[]),info("matches",T.BOOL,[])],dict(x=np.ones((4,3),np.float32)),[nh.from_array(np.array(6,np.int64),"six")])
save("simplify_equal_shapes",[
 h.make_node("Shape",["x"],["sx"]),h.make_node("Shape",["sx"],["rankx"]),
 h.make_node("Shape",["y"],["sy"]),h.make_node("Shape",["sy"],["ranky"]),
 h.make_node("Equal",["rankx","ranky"],["same"]),h.make_node("Equal",["rankx","one"],["different"])],
 [info("x",T.FLOAT,[2,3]),info("y",T.FLOAT,[4,3])],[info("same",T.BOOL,[1]),info("different",T.BOOL,[1])],
 dict(x=np.ones((2,3),np.float32),y=np.ones((4,3),np.float32)),[nh.from_array(np.array([1],np.int64),"one")])
save("simplify_range_name",[h.make_node("Range",["zero","start","step"],["out"])],
 [info("start",T.INT64,[1])],[info("out",T.INT64,[None])],dict(start=np.array([4],np.int64)),
 [nh.from_array(np.array(0,np.int64),"zero"),nh.from_array(np.array(1,np.int64),"step")])
