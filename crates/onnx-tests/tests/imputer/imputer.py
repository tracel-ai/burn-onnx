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
    # -999.0 is used as an explicit sentinel for missing values.
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

    # Test case 3: integer input — replace sentinel -1 with 0
    int_input_data = np.array([[1, -1, 3], [4, 5, -1]], dtype=np.int64)
    int_node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_int64s=[0],
        replaced_value_int64=-1,
    )

    int_input_tensor = helper.make_tensor_value_info("input", TensorProto.INT64, [2, 3])
    int_output_tensor = helper.make_tensor_value_info("output", TensorProto.INT64, [2, 3])

    int_graph = helper.make_graph(
        [int_node],
        "imputer_int_test",
        [int_input_tensor],
        [int_output_tensor],
    )

    int_model = helper.make_model(
        int_graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),
        ],
    )

    onnx.save(int_model, "imputer_int.onnx")
    print("Finished exporting model to imputer_int.onnx")

    int_sess = ReferenceEvaluator(int_model)
    int_result = int_sess.run(None, {"input": int_input_data})

    print("\nInteger input:")
    print(int_input_data)
    print("\nInteger output (-1 replaced with 0):")
    print(int_result[0])

    # Test case 4: float input with explicit NaN sentinel (replaced_value_float=NaN)
    nan_input_data = np.array([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]], dtype=np.float32)
    nan_node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_floats=[0.0],
        replaced_value_float=float("nan"),
    )

    nan_graph = helper.make_graph(
        [nan_node],
        "imputer_nan_test",
        [input_tensor],
        [output_tensor],
    )

    nan_model = helper.make_model(
        nan_graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),
        ],
    )

    onnx.save(nan_model, "imputer_nan.onnx")
    print("Finished exporting model to imputer_nan.onnx")

    nan_sess = ReferenceEvaluator(nan_model)
    nan_result = nan_sess.run(None, {"input": nan_input_data})

    print("\nNaN input:")
    print(nan_input_data)
    print("\nNaN output (NaN replaced with 0.0):")
    print(nan_result[0])

    # Test case 5: float input with DEFAULT NaN sentinel (no replaced_value_float attribute)
    # This tests the case where replaced_value_float is omitted entirely and defaults to NaN
    nan_default_input_data = np.array([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]], dtype=np.float32)
    nan_default_node = helper.make_node(
        "Imputer",
        ["input"],
        ["output"],
        domain="ai.onnx.ml",
        imputed_value_floats=[0.0],
        # Note: replaced_value_float is NOT set, so it defaults to NaN per ONNX spec
    )

    nan_default_graph = helper.make_graph(
        [nan_default_node],
        "imputer_nan_default_test",
        [input_tensor],
        [output_tensor],
    )

    nan_default_model = helper.make_model(
        nan_default_graph,
        opset_imports=[
            helper.make_operatorsetid("ai.onnx.ml", OPSET_VERSION),
            helper.make_operatorsetid("", 17),
        ],
    )

    onnx.save(nan_default_model, "imputer_nan_default.onnx")
    print("Finished exporting model to imputer_nan_default.onnx")

    nan_default_sess = ReferenceEvaluator(nan_default_model)
    nan_default_result = nan_default_sess.run(None, {"input": nan_default_input_data})

    print("\nNaN default input:")
    print(nan_default_input_data)
    print("\nNaN default output (NaN replaced with 0.0, replaced_value_float omitted):")
    print(nan_default_result[0])


if __name__ == "__main__":
    main()
