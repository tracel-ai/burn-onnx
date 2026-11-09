# /// script
# dependencies = ["onnx==1.23.0", "numpy"]
# ///
"""External KV caches: per-batch lengths, short masks, GQA and causal alignment."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from onnx.reference import ReferenceEvaluator
np.random.seed(42)
root = Path(__file__).parent
feeds = {"q": np.arange(16, dtype=np.float32).reshape(2,2,2,2)/10,
         "k": np.arange(16, dtype=np.float32).reshape(2,1,4,2)/10,
         "v": np.arange(16, dtype=np.float32).reshape(2,1,4,2)/5,
         "lengths": np.array([3,2], dtype=np.int64)}
mask = np.array([[[[True,True,False],[True,True,True]]],
                 [[[False,False,False],[True,False,True]]]], dtype=bool)
for suffix, causal, qk, masked in [("nativecausal",1,False,False), ("plain3d",0,False,False), ("plain",0,False,False), ("causal",1,False,True), ("qk",1,True,True)]:
    data = dict(feeds)
    if suffix == "plain3d":
        for n in ["q", "k", "v"]:
            a = data[n]
            data[n] = a.transpose(0,2,1,3).reshape(a.shape[0], a.shape[2], -1)
    if masked: data["mask"] = mask
    outputs = ["y", "", "", "qk"] if qk else ["y"]
    graph = h.make_graph([h.make_node("Attention", ["q","k","v","mask" if masked else "","","","lengths"],
        outputs, is_causal=causal, qk_matmul_output_mode=3, **({"q_num_heads":2,"kv_num_heads":1} if suffix=="plain3d" else {}))], "lengths",
        [h.make_tensor_value_info(n, T.INT64 if n=="lengths" else T.BOOL if n=="mask" else T.FLOAT, list(a.shape)) for n,a in data.items()],
        [h.make_tensor_value_info(n, T.FLOAT, [2,2,4] if suffix=="plain3d" else [2,2,2,4] if n=="qk" else [2,2,2,2]) for n in outputs if n])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("",24)])
    onnx.checker.check_model(model)
    onnx.save(model, root/f"attention_lengths_{suffix}.onnx")
    for case, lengths in enumerate([[3,2],[0,2]]):
        data["lengths"] = np.array(lengths,dtype=np.int64)
        for name,output in zip([n for n in outputs if n], ReferenceEvaluator(model).run(None,data)):
            # Rust compares the rank-3 model after restoring its rank-4 layout.
            if suffix != "plain3d":
                output.astype("<f4").tofile(root/f"attention_lengths_{suffix}_{case}_{name}.bin")

shape_model = h.make_model(h.make_graph([
    h.make_node("Shape", ["k"], ["lengths"], start=2, end=3),
    h.make_node("Attention", ["q","k","v","","","","lengths"], ["y"]),
], "shape_lengths", [
    h.make_tensor_value_info("q", T.FLOAT, [1,2,2,2]),
    h.make_tensor_value_info("k", T.FLOAT, [1,1,"capacity",2]),
    h.make_tensor_value_info("v", T.FLOAT, [1,1,"capacity",2]),
], [h.make_tensor_value_info("y", T.FLOAT, [1,2,2,2])]), opset_imports=[h.make_opsetid("",24)])
onnx.checker.check_model(shape_model)
onnx.save(shape_model, root/"attention_lengths_shape.onnx")
shape_feeds = {n: a[:1] for n,a in feeds.items() if n != "lengths"}
ReferenceEvaluator(shape_model).run(None, shape_feeds)[0].astype("<f4").tofile(root/"attention_lengths_shape_y.bin")
