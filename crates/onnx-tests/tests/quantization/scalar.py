# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
info=lambda n,t,s:h.make_tensor_value_info(n,t,s)
model=h.make_model(h.make_graph([h.make_node('QuantizeLinear',['x','scale','zero'],['q']), h.make_node('DequantizeLinear',['q','scale','zero'],['y'])], 'scalar', [info('x',T.FLOAT,[2,3]),info('scale',T.FLOAT,[]),info('zero',T.UINT8,[])], [info('q',T.UINT8,[2,3]),info('y',T.FLOAT,[2,3])]),opset_imports=[h.make_opsetid('',21)])
onnx.checker.check_model(model)
onnx.save(model,Path(__file__).with_suffix('.onnx'))
x=np.array([[-100, -1.25, -.25],[.25,1.25,1000]],np.float32)
print(ReferenceEvaluator(model).run(None,dict(x=x,scale=np.array(.5,np.float32),zero=np.array(4,np.uint8))))
