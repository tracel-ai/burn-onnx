#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate models: imputer.onnx, imputer_per_feature.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 1


def main():
    np.random.seed(42)

    # Test case 1: single imputed value (replace -999.0 with 0.0)
    # Note: ONNX Imputer doesn't handle NaN directly; it replaces specific values.
    # We use -999.0 as a sentinel for missing values.
    input_data = np.array([[1.0, -999.0, 3.0], [4.0, 5.0, -999.0]], dtype=np.float32)
    node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_floats=[0.0],
        replaced_value_float=-999.0,
    )
    
    # Create graph
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    
    graph = helper.make_graph(
        [node],
        "imputer_test",
        [input_tensor],
        [output_tensor],
    )
    
    model = helper.make_model(
        graph, opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),  # Default domain opset
        ]
    )
    
    onnx.save(model, "imputer.onnx")
    print("Finished exporting model to imputer.onnx")

    sess = ReferenceEvaluator(model)
    result = sess.run(None, {"input": input_data})

    print("\nInput:")
    print(input_data)
    print("\nOutput (-999.0 replaced with 0.0):")
    print(result[0])

    # Test case 2: per-feature imputed values
    # Replace -999.0 in each column with its own value [10, 20, 30].
    per_feature_input = np.array(
        [
            [-999.0, 2.0, -999.0],
            [4.0, -999.0, 6.0],
        ],
        dtype=np.float32,
    )

    per_feature_node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_floats=[10.0, 20.0, 30.0],
        replaced_value_float=-999.0,
    )

    per_feature_graph = helper.make_graph(
        [per_feature_node],
        "imputer_per_feature_test",
        [input_tensor],
        [output_tensor],
    )

    per_feature_model = helper.make_model(
        per_feature_graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),  # Default domain opset
        ],
    )

    onnx.save(per_feature_model, "imputer_per_feature.onnx")
    print("Finished exporting model to imputer_per_feature.onnx")

    per_feature_sess = ReferenceEvaluator(per_feature_model)
    per_feature_result = per_feature_sess.run(None, {"input": per_feature_input})

    print("\nPer-feature input:")
    print(per_feature_input)
    print("\nPer-feature output (-999.0 replaced by [10, 20, 30] per column):")
    print(per_feature_result[0])


if __name__ == "__main__":
    main()
