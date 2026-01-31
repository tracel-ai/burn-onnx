#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: if_linear.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Identity", ["x"], ["x_input"])

    else_branch_init_else_weights = numpy_helper.from_array(
        np.array(
            [
                -0.06899978220462799,
                0.13831374049186707,
                0.14408420026302338,
                0.15694919228553772,
                -0.008251483552157879,
                -0.07299817353487015,
                -0.13720250129699707,
                0.14989738166332245,
                -0.09566367417573929,
                0.09063049405813217,
                0.036029260605573654,
                -0.01815488189458847,
                -0.026097942143678665,
                -0.21963348984718323,
                0.028441324830055237,
                0.15849481523036957,
                -0.058875035494565964,
                -0.10504671186208725,
                -0.11666909605264664,
                0.14694112539291382,
                -0.10624774545431137,
                0.011162863112986088,
                0.0679384395480156,
                -0.0014343969523906708,
                0.03689006716012955,
                -0.06988617032766342,
                -0.055100709199905396,
                0.022783836349844933,
                0.18131650984287262,
                -0.03965723142027855,
            ],
            dtype=np.float32,
        ).reshape([5, 6]),
        name="else_weights",
    )
    else_branch_init_else_bias = numpy_helper.from_array(
        np.array(
            [
                -0.016322925686836243,
                -0.010074645280838013,
                -0.013432850129902363,
                0.013715281151235104,
                0.0026333953719586134,
                0.000791763246525079,
            ],
            dtype=np.float32,
        ).reshape([6]),
        name="else_bias",
    )
    else_branch_init_else_mul_const = numpy_helper.from_array(
        np.array([1.5], dtype=np.float32).reshape([1]), name="else_mul_const"
    )
    else_branch_node0 = helper.make_node(
        "MatMul", ["x_input", "else_weights"], ["else_matmul_out"]
    )
    else_branch_node1 = helper.make_node(
        "Add", ["else_matmul_out", "else_bias"], ["else_linear_out"]
    )
    else_branch_node2 = helper.make_node(
        "Mul", ["else_linear_out", "else_mul_const"], ["else_output"]
    )
    else_branch_inp_x_input = helper.make_tensor_value_info(
        "x_input", TensorProto.FLOAT, [2, 5]
    )
    else_branch_out_else_output = helper.make_tensor_value_info(
        "else_output", TensorProto.FLOAT, [2, 6]
    )
    else_branch = helper.make_graph(
        [else_branch_node0, else_branch_node1, else_branch_node2],
        "else_branch",
        [else_branch_inp_x_input],
        [else_branch_out_else_output],
        initializer=[
            else_branch_init_else_weights,
            else_branch_init_else_bias,
            else_branch_init_else_mul_const,
        ],
    )

    then_branch_init_then_weights = numpy_helper.from_array(
        np.array(
            [
                0.03471628576517105,
                0.05935513600707054,
                -0.2465275526046753,
                -0.029306570068001747,
                -0.3127596974372864,
                0.11997383832931519,
                0.02726096846163273,
                -0.04698772355914116,
                0.03495538607239723,
                -0.05127568170428276,
                0.12402128428220749,
                -0.22674822807312012,
                0.04250297695398331,
                -0.029285266995429993,
                -0.11492715030908585,
                -0.04403650760650635,
                0.0018438417464494705,
                -0.16522078216075897,
                0.17950880527496338,
                -0.19747814536094666,
                -0.12243866920471191,
                -0.019051147624850273,
                0.17673379182815552,
                -0.017315372824668884,
                0.23524987697601318,
                -0.02315344288945198,
                -0.009921828284859657,
                -0.06540118902921677,
                -0.16283731162548065,
                0.2198699563741684,
                0.017742447555065155,
                -0.06281713396310806,
                -0.15383853018283844,
                -0.033122655004262924,
                0.04432637616991997,
                0.04890937730669975,
                -0.03400649130344391,
                -0.004104148130863905,
                -0.06391217559576035,
                -0.03479693830013275,
            ],
            dtype=np.float32,
        ).reshape([5, 8]),
        name="then_weights",
    )
    then_branch_init_then_bias = numpy_helper.from_array(
        np.array(
            [
                0.01438705250620842,
                0.005427639465779066,
                0.006323283538222313,
                -0.007757253013551235,
                0.014736952260136604,
                0.00670194486156106,
                -0.0089835524559021,
                0.013574095442891121,
            ],
            dtype=np.float32,
        ).reshape([8]),
        name="then_bias",
    )
    then_branch_init_then_add_const = numpy_helper.from_array(
        np.array([0.5], dtype=np.float32).reshape([1]), name="then_add_const"
    )
    then_branch_node0 = helper.make_node(
        "MatMul", ["x_input", "then_weights"], ["then_matmul_out"]
    )
    then_branch_node1 = helper.make_node(
        "Add", ["then_matmul_out", "then_bias"], ["then_linear_out"]
    )
    then_branch_node2 = helper.make_node(
        "Add", ["then_linear_out", "then_add_const"], ["then_output"]
    )
    then_branch_inp_x_input = helper.make_tensor_value_info(
        "x_input", TensorProto.FLOAT, [2, 5]
    )
    then_branch_out_then_output = helper.make_tensor_value_info(
        "then_output", TensorProto.FLOAT, [2, 8]
    )
    then_branch = helper.make_graph(
        [then_branch_node0, then_branch_node1, then_branch_node2],
        "then_branch",
        [then_branch_inp_x_input],
        [then_branch_out_then_output],
        initializer=[
            then_branch_init_then_weights,
            then_branch_init_then_bias,
            then_branch_init_then_add_const,
        ],
    )
    node1 = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        else_branch=else_branch,
        then_branch=then_branch,
    )

    inp_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])
    inp_condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])

    out_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, ""])

    graph = helper.make_graph(
        [node0, node1], "main_graph", [inp_x, inp_condition], [out_output]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "if_linear.onnx")
    print(f"Finished exporting model to if_linear.onnx")


if __name__ == "__main__":
    main()
