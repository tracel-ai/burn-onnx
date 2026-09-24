// Import the shared macro
use crate::include_models;
include_models!(lp_pool1d, lp_pool1d_opset1, lp_pool2d, lp_pool2d_opset1);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData, Tolerance};

    #[test]
    fn lp_pool1d() {
        let device = Default::default();
        let model: lp_pool1d::Model = lp_pool1d::Model::new(&device);

        let input = Tensor::<3>::from_floats([[[-1.0, 2.0, -3.0, 4.0, -5.0]]], &device);
        let output = model.forward(input);

        let expected = TensorData::from([[[2.0800838f32, 4.6260653, 5.738794]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(0.01, 0.001));
    }

    // Opset 1 carries `p` as a FLOAT, here a fractional 1.5. Input from
    // np.random.seed(42), expected output from onnx.reference.ReferenceEvaluator.
    #[test]
    fn lp_pool1d_opset1_float_p() {
        let device = Default::default();
        let model: lp_pool1d_opset1::Model = lp_pool1d_opset1::Model::default();

        let input = Tensor::<3>::from_floats(
            [[[
                0.49671414,
                -0.1382643,
                0.64768857,
                1.5230298,
                -0.23415338,
                -0.23413695,
            ]]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[[0.54422724f32, 1.792981, 0.37168226]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }

    #[test]
    fn lp_pool2d() {
        let device = Default::default();
        let model: lp_pool2d::Model = lp_pool2d::Model::new(&device);

        let input = Tensor::<4>::from_floats(
            [[[
                [1.0, -2.0, 3.0, -4.0],
                [5.0, -6.0, 7.0, -8.0],
                [9.0, -10.0, 11.0, -12.0],
                [13.0, -14.0, 15.0, -16.0],
            ]]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[[
            [7.047299f32, 10.537283],
            [12.74452, 17.246693],
            [18.823858, 24.257643],
            [17.032236, 21.697657],
        ]]]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::rel_abs(0.01, 0.001));
    }

    // Opset 1 carries `p` as a FLOAT, here a fractional 1.5. Input from
    // np.random.seed(42), expected output from onnx.reference.ReferenceEvaluator.
    #[test]
    fn lp_pool2d_opset1_float_p() {
        let device = Default::default();
        let model: lp_pool2d_opset1::Model = lp_pool2d_opset1::Model::default();

        let input = Tensor::<4>::from_floats(
            [[[
                [0.49671414, -0.1382643, 0.64768857, 1.5230298],
                [-0.23415338, -0.23413695, 1.5792128, 0.7674347],
                [-0.46947438, 0.54256004, -0.46341768, -0.46572974],
                [0.24196227, -1.9132802, -1.7249179, -0.5622875],
            ]]],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([[[[0.73340786f32, 2.9464645], [2.2994246, 2.2256594]]]]);
        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
