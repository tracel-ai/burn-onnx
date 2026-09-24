use crate::include_models;
include_models!(global_max_pool);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn global_max_pool() {
        let device = Default::default();
        let model: global_max_pool::Model = global_max_pool::Model::new(&device);

        let x3 = Tensor::<3>::from_floats(
            [[
                [0.49671414, -0.1382643, 0.64768857, 1.5230298],
                [-0.23415338, -0.23413695, 1.5792128, 0.7674347],
            ]],
            &device,
        );
        let x4 = Tensor::<4>::from_floats(
            [
                [
                    [
                        [-0.46947438, 0.54256004, -0.46341768],
                        [-0.46572974, 0.24196227, -1.9132802],
                    ],
                    [
                        [-1.7249179, -0.5622875, -1.0128311],
                        [0.31424734, -0.9080241, -1.4123037],
                    ],
                ],
                [
                    [
                        [1.4656488, -0.2257763, 0.0675282],
                        [-1.4247482, -0.54438275, 0.11092259],
                    ],
                    [
                        [-1.1509936, 0.37569803, -0.6006387],
                        [-0.29169375, -0.60170662, 1.8522782],
                    ],
                ],
            ],
            &device,
        );

        let (y3, y4) = model.forward(x3, x4);

        y3.to_data()
            .assert_eq(&TensorData::from([[[1.5230298f32], [1.5792128]]]), true);
        y4.to_data().assert_eq(
            &TensorData::from([
                [[[0.54256004f32]], [[0.31424734]]],
                [[[1.4656488]], [[1.8522782]]],
            ]),
            true,
        );
    }
}
