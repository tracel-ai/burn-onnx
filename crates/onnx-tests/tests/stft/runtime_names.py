# /// script
# dependencies = ["onnx==1.19.0", "numpy"]
# ///
"""Rename runtime FFT inputs to names also used by codegen temporaries."""
from pathlib import Path
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
root=Path(__file__).parent.parent
def renamed(source, dest, mapping, feeds):
    model=onnx.load(root/source)
    for node in model.graph.node:
        for names in (node.input,node.output):
            for i,name in enumerate(names): names[i]=mapping.get(name,name)
    for value in list(model.graph.input)+list(model.graph.output): value.name=mapping.get(value.name,value.name)
    onnx.checker.check_model(model)
    outputs=ReferenceEvaluator(model).run(None, feeds)
    print(dest,[(list(x.shape),float(x.sum())) for x in outputs])
    onnx.save(model,root/dest)
renamed("dft/dft_runtime_axis.onnx","dft/dft_axis_name.onnx",{"x":"axis","axis":"dim"},{"axis":np.ones((1,3,5,1),np.float32),"dim":np.array(1,np.int64)})
for param in ("frame_step","frame_length"):
    mapping={"signal":"x",param:"signal"}
    renamed("stft/stft_runtime_params.onnx",f"stft/stft_{param}_name.onnx",mapping,{mapping.get(k,k):v for k,v in dict(signal=np.ones((1,32,1),np.float32),frame_step=np.array(4,np.int64),frame_length=np.array(12,np.int64)).items()})
