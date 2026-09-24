use crate::include_models;
include_models!(
    split,
    split_uneven,
    split_axis1,
    split_runtime_sizes,
    split_zero_size
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Device, Shape, Tensor, TensorData};

    #[test]
    fn split() {
        let device = Default::default();
        let model = split::Model::new(&device);
        let shape = [5, 2];
        let input = Tensor::ones(shape, &device);

        let (tensor_1, tensor_2, tensor_3) = model.forward(input);

        assert_eq!(tensor_1.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_2.shape(), Shape::from([2, 2]));
        assert_eq!(tensor_3.shape(), Shape::from([1, 2]));
    }

    #[test]
    fn split_uneven() {
        // num_outputs=3 on dim=10 (not evenly divisible)
        // ONNX spec: ceil(10/3) = 4, splits are [4, 4, 2]
        let device = Default::default();
        let model = split_uneven::Model::new(&device);

        // np.random.seed(42); np.random.randn(10, 3).astype(np.float32)
        let input = Tensor::<2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857],
                [1.5230298, -0.23415338, -0.23413695],
                [1.5792128, 0.7674347, -0.46947438],
                [0.54256004, -0.46341768, -0.46572974],
                [0.24196227, -1.9132802, -1.7249179],
                [-0.5622875, -1.0128311, 0.31424734],
                [-0.9080241, -1.4123037, 1.4656488],
                [-0.2257763, 0.0675282, -1.4247482],
                [-0.54438275, 0.11092259, -1.1509936],
                [0.37569803, -0.6006387, -0.29169375],
            ],
            &device,
        );
        let (y0, y1, y2) = model.forward(input);

        assert_eq!(y0.shape(), Shape::from([4, 3]));
        assert_eq!(y1.shape(), Shape::from([4, 3]));
        assert_eq!(y2.shape(), Shape::from([2, 3]));

        let expected_y0 = TensorData::from([
            [0.49671414f32, -0.1382643, 0.64768857],
            [1.5230298, -0.23415338, -0.23413695],
            [1.5792128, 0.7674347, -0.46947438],
            [0.54256004, -0.46341768, -0.46572974],
        ]);
        let expected_y2 = TensorData::from([
            [-0.54438275f32, 0.11092259, -1.1509936],
            [0.37569803, -0.6006387, -0.29169375],
        ]);

        y0.to_data().assert_eq(&expected_y0, true);
        y2.to_data().assert_eq(&expected_y2, true);
    }

    #[test]
    fn split_axis1() {
        // Explicit split sizes [2, 3] on axis=1
        let device = Default::default();
        let model = split_axis1::Model::new(&device);

        // np.random.seed(42); np.random.randn(3, 5).astype(np.float32)
        let input = Tensor::<2>::from_floats(
            [
                [0.49671414, -0.1382643, 0.64768857, 1.5230298, -0.23415338],
                [-0.23413695, 1.5792128, 0.7674347, -0.46947438, 0.54256004],
                [-0.46341768, -0.46572974, 0.24196227, -1.9132802, -1.7249179],
            ],
            &device,
        );
        let (y0, y1) = model.forward(input);

        assert_eq!(y0.shape(), Shape::from([3, 2]));
        assert_eq!(y1.shape(), Shape::from([3, 3]));

        let expected_y0 = TensorData::from([
            [0.49671414f32, -0.1382643],
            [-0.23413695, 1.5792128],
            [-0.46341768, -0.46572974],
        ]);
        let expected_y1 = TensorData::from([
            [0.64768857f32, 1.5230298, -0.23415338],
            [0.7674347, -0.46947438, 0.54256004],
            [0.24196227, -1.9132802, -1.7249179],
        ]);

        y0.to_data().assert_eq(&expected_y0, true);
        y1.to_data().assert_eq(&expected_y1, true);
    }

    #[test]
    fn split_runtime_sizes() {
        let device = Default::default();
        let model = split_runtime_sizes::Model::new(&device);
        let input = || Tensor::<1>::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let sizes = |s: [i64; 2]| Tensor::<1, burn::tensor::Int>::from_ints(s, &device);

        let (a, b) = model.forward(input(), sizes([2, 4]));
        a.to_data()
            .assert_eq(&TensorData::from([0.0f32, 1.0]), true);
        b.to_data()
            .assert_eq(&TensorData::from([2.0f32, 3.0, 4.0, 5.0]), true);

        let (a, b) = model.forward(input(), sizes([5, 1]));
        a.to_data()
            .assert_eq(&TensorData::from([0.0f32, 1.0, 2.0, 3.0, 4.0]), true);
        b.to_data().assert_eq(&TensorData::from([5.0f32]), true);
    }

    #[test]
    fn split_zero_size() {
        // Expected values from split_zero_size.py (onnx ReferenceEvaluator).
        let device = Default::default();
        let model = split_zero_size::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/split_zero_size.bpk"),
            &device,
        );
        let x = Tensor::<2>::from_data(
            TensorData::new((0..15).map(|v| v as f32).collect::<Vec<_>>(), [5, 3]),
            &device,
        );
        let runtime_sizes = Tensor::<1, burn::tensor::Int>::from_ints([0, 4, 1], &device);

        let (s0, s1, s2, r0, r1, r2, c0, c1, c2, c3) = model.forward(x, runtime_sizes);

        // Rows [start, end) of x, which holds 0..15 in row-major order.
        let rows = |start: usize, end: usize| {
            let values: Vec<f32> = (start * 3..end * 3).map(|v| v as f32).collect();
            TensorData::new(values, [end - start, 3])
        };
        let column = |c: usize| {
            let values: Vec<f32> = (0..5).map(|r| (r * 3 + c) as f32).collect();
            TensorData::new(values, [5, 1])
        };

        s0.to_data().assert_eq(&rows(0, 3), true);
        assert_eq!(s1.shape(), Shape::from([0, 3]));
        s2.to_data().assert_eq(&rows(3, 5), true);
        assert_eq!(r0.shape(), Shape::from([0, 3]));
        r1.to_data().assert_eq(&rows(0, 4), true);
        r2.to_data().assert_eq(&rows(4, 5), true);
        c0.to_data().assert_eq(&column(0), true);
        c1.to_data().assert_eq(&column(1), true);
        c2.to_data().assert_eq(&column(2), true);
        assert_eq!(c3.shape(), Shape::from([5, 0]));
    }
}
