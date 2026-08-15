//! Runs the imported model and checks it against the reference values printed
//! by `src/model/generate_model.py`.

use burn::prelude::*;
use burn::tensor::Tolerance;
use custom_op_hooks::custom_model::Model;

fn main() {
    let device = Device::default();

    // from_file loads the weights produced at build time. Model::new would
    // leave them zeroed: the MatMul weights are real parameters (only the
    // ChannelScale constant was inlined by its hook).
    let weights = concat!(env!("OUT_DIR"), "/model/custom_model.bpk");
    let model: Model = Model::from_file(weights, &device);

    let input = Tensor::<2>::from_floats([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]], &device);
    let output = model.forward(input);

    println!("output: {output}");

    // Reference values from generate_model.py (numpy).
    let expected = TensorData::from([
        [1.0f32, 0.273_183_23, 0.006_790_343],
        [1.376_118e-9, 0.346_480_58, 1.997_496_6],
    ]);
    output
        .to_data()
        .assert_approx_eq::<f32>(&expected, Tolerance::default());

    println!("matches the ONNX reference");
}
