use crate::include_models;

include_models!(
    gathernd_bool,
    gathernd,
    gathernd_partial,
    gathernd_3d,
    gathernd_batch1,
    gathernd_neg_idx
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor, TensorData};

    #[test]
    fn gathernd_full_index() {
        // Spec Example 1: data [2,2], indices [2,2] -> output [2]
        let model: gathernd::Model = gathernd::Model::default();
        let device = Default::default();

        let data = Tensor::<2>::from_floats([[0., 1.], [2., 3.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([0f32, 3.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_partial_index() {
        // Spec Example 2: data [2,2], indices [2,1] -> output [2,2]
        let model: gathernd_partial::Model = gathernd_partial::Model::default();
        let device = Default::default();

        let data = Tensor::<2>::from_floats([[0., 1.], [2., 3.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [0., 1.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_3d_data() {
        // Spec Example 3: data [2,2,2], indices [2,2] -> output [2,2]
        let model: gathernd_3d::Model = gathernd_3d::Model::default();
        let device = Default::default();

        let data = Tensor::<3>::from_floats([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [4., 5.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_batch_dims() {
        // Spec Example 5: batch_dims=1, data [2,2,2], indices [2,1] -> output [2,2]
        let model: gathernd_batch1::Model = gathernd_batch1::Model::default();
        let device = Default::default();

        let data = Tensor::<3>::from_floats([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [4., 5.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_negative_indices() {
        // data [3,2], indices [2,1] with -1 index -> output [2,2]
        let model: gathernd_neg_idx::Model = gathernd_neg_idx::Model::default();
        let device = Default::default();

        let data = Tensor::<2>::from_floats([[0., 1.], [2., 3.], [4., 5.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[4f32, 5.], [0., 1.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_bool() {
        // Expected values from gathernd_bool.py (ONNX reference): single elements with a
        // negative index, and rows of 3D data (index length 1 < rank) with batch_dims=1
        let device = Default::default();
        let model = gathernd_bool::Model::from_file(
            concat!(env!("OUT_DIR"), "/model/gathernd_bool.bpk"),
            &device,
        );
        let data = burn::tensor::Tensor::<2, burn::tensor::Bool>::from_bool(
            TensorData::from([[true, false, true], [false, true, false]]),
            &device,
        );
        let indices = burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
            TensorData::from([[0i64, 1], [1, 1], [-1, 0]]),
            &device,
        );

        let data_3d = burn::tensor::Tensor::<3, burn::tensor::Bool>::from_bool(
            TensorData::from([
                [[true, false, true], [false, false, true]],
                [[false, true, true], [true, true, false]],
            ]),
            &device,
        );
        let indices_3d = burn::tensor::Tensor::<2, burn::tensor::Int>::from_data(
            TensorData::from([[1i64], [0]]),
            &device,
        );

        let (output, output_slices) = model.forward(data, indices, data_3d, indices_3d);

        output
            .to_data()
            .assert_eq(&TensorData::from([false, true, false]), false);
        output_slices.to_data().assert_eq(
            &TensorData::from([[false, false, true], [false, true, true]]),
            false,
        );
    }
}
