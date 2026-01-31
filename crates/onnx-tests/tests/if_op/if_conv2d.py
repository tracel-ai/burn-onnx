#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: if_conv2d.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OPSET_VERSION = 16


def main():
    node0 = helper.make_node("Identity", ["x"], ["x_input"])

    else_branch_init_else_conv_weights = numpy_helper.from_array(
        np.array(
            [
                -0.026230990886688232,
                0.03264722228050232,
                -0.01752324216067791,
                -0.12008970975875854,
                0.011772572994232178,
                -0.1440521478652954,
            ],
            dtype=np.float32,
        ).reshape([3, 2, 1, 1]),
        name="else_conv_weights",
    )
    else_branch_init_else_conv_bias = numpy_helper.from_array(
        np.array(
            [-0.0013980440562590957, 0.0043389578349888325, 0.0020434586331248283],
            dtype=np.float32,
        ).reshape([3]),
        name="else_conv_bias",
    )
    else_branch_init_else_mul_const = numpy_helper.from_array(
        np.array([2.0], dtype=np.float32).reshape([1]), name="else_mul_const"
    )
    else_branch_node0 = helper.make_node(
        "Conv",
        ["x_input", "else_conv_weights", "else_conv_bias"],
        ["else_conv_out"],
        kernel_shape=[1, 1],
    )
    else_branch_node1 = helper.make_node(
        "Mul", ["else_conv_out", "else_mul_const"], ["else_output"]
    )
    else_branch_inp_x_input = helper.make_tensor_value_info(
        "x_input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    else_branch_out_else_output = helper.make_tensor_value_info(
        "else_output", TensorProto.FLOAT, [1, 3, 4, 4]
    )
    else_branch = helper.make_graph(
        [else_branch_node0, else_branch_node1],
        "else_branch",
        [else_branch_inp_x_input],
        [else_branch_out_else_output],
        initializer=[
            else_branch_init_else_conv_weights,
            else_branch_init_else_conv_bias,
            else_branch_init_else_mul_const,
        ],
    )

    then_branch_init_then_conv_weights = numpy_helper.from_array(
        np.array(
            [
                0.0062792650423944,
                0.01018186379224062,
                -0.017946559935808182,
                0.03686149790883064,
                0.18356488645076752,
                0.007340160198509693,
                -0.01889890804886818,
                0.051749616861343384,
                0.0045648240484297276,
                -0.04718387871980667,
                0.18461477756500244,
                -0.12731988728046417,
                0.12483429908752441,
                -0.01058033388108015,
                -0.08489411324262619,
                0.06900661438703537,
                0.027210945263504982,
                -0.16210947930812836,
                0.04117472097277641,
                0.03595687821507454,
                -0.0833130031824112,
                -0.06289147585630417,
                0.01756666973233223,
                -0.08888985961675644,
                -0.05388952046632767,
                -0.09861009567975998,
                -0.12425316870212555,
                0.11475975811481476,
                0.06318313628435135,
                0.26006075739860535,
                -0.038811277598142624,
                0.0124992486089468,
                0.026229411363601685,
                -0.13714466989040375,
                -0.145949125289917,
                0.04517635330557823,
                0.06306454539299011,
                -0.14728516340255737,
                0.018353404477238655,
                0.09785288572311401,
                0.10049381107091904,
                -0.10394078493118286,
                0.1610475480556488,
                -0.11356496810913086,
                0.05412856489419937,
                0.023206369951367378,
                0.056992705911397934,
                -0.24426808953285217,
                0.17524705827236176,
                -0.02186218462884426,
                -0.07496672123670578,
                0.059531357139348984,
                0.049336984753608704,
                0.08814554661512375,
                0.045388754457235336,
                0.011366221122443676,
                -0.06369312852621078,
                0.0370362363755703,
                0.09913959354162216,
                -0.11782006174325943,
                -0.049742069095373154,
                -0.0863153487443924,
                0.14342281222343445,
                -0.09808763116598129,
                -0.1623149961233139,
                -0.07513117790222168,
                0.11502490192651749,
                -0.017736829817295074,
                -0.055597275495529175,
                0.1255975216627121,
                0.11963093280792236,
                0.03184151649475098,
            ],
            dtype=np.float32,
        ).reshape([4, 2, 3, 3]),
        name="then_conv_weights",
    )
    then_branch_init_then_conv_bias = numpy_helper.from_array(
        np.array(
            [
                -0.002083738800138235,
                -0.023250119760632515,
                0.0020702241454273462,
                -0.0026169251650571823,
            ],
            dtype=np.float32,
        ).reshape([4]),
        name="then_conv_bias",
    )
    then_branch_init_then_add_const = numpy_helper.from_array(
        np.array([1.0], dtype=np.float32).reshape([1]), name="then_add_const"
    )
    then_branch_node0 = helper.make_node(
        "Conv",
        ["x_input", "then_conv_weights", "then_conv_bias"],
        ["then_conv_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    then_branch_node1 = helper.make_node(
        "Add", ["then_conv_out", "then_add_const"], ["then_output"]
    )
    then_branch_inp_x_input = helper.make_tensor_value_info(
        "x_input", TensorProto.FLOAT, [1, 2, 4, 4]
    )
    then_branch_out_then_output = helper.make_tensor_value_info(
        "then_output", TensorProto.FLOAT, [1, 4, 4, 4]
    )
    then_branch = helper.make_graph(
        [then_branch_node0, then_branch_node1],
        "then_branch",
        [then_branch_inp_x_input],
        [then_branch_out_then_output],
        initializer=[
            then_branch_init_then_conv_weights,
            then_branch_init_then_conv_bias,
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

    inp_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    inp_condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])

    out_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, "", 4, 4]
    )

    graph = helper.make_graph(
        [node0, node1], "main_graph", [inp_x, inp_condition], [out_output]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "if_conv2d.onnx")
    print(f"Finished exporting model to if_conv2d.onnx")


if __name__ == "__main__":
    main()
