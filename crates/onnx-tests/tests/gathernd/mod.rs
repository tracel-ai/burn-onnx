use crate::include_models;

include_models!(
    gathernd,
    gathernd_partial,
    gathernd_3d,
    gathernd_batch1,
    gathernd_neg_idx
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn gathernd_full_index() {
        // Spec Example 1: data [2,2], indices [2,2] -> output [2]
        let model: gathernd::Model<TestBackend> = gathernd::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 2>::from_floats([[0., 1.], [2., 3.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([0f32, 3.]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_partial_index() {
        // Spec Example 2: data [2,2], indices [2,1] -> output [2,2]
        let model: gathernd_partial::Model<TestBackend> = gathernd_partial::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 2>::from_floats([[0., 1.], [2., 3.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [0., 1.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_3d_data() {
        // Spec Example 3: data [2,2,2], indices [2,2] -> output [2,2]
        let model: gathernd_3d::Model<TestBackend> = gathernd_3d::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
            &device,
        );
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [4., 5.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_batch_dims() {
        // Spec Example 5: batch_dims=1, data [2,2,2], indices [2,1] -> output [2,2]
        let model: gathernd_batch1::Model<TestBackend> = gathernd_batch1::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 3>::from_floats(
            [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
            &device,
        );
        let output = model.forward(data);

        let expected = TensorData::from([[2f32, 3.], [4., 5.]]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gathernd_negative_indices() {
        // data [3,2], indices [2,1] with -1 index -> output [2,2]
        let model: gathernd_neg_idx::Model<TestBackend> = gathernd_neg_idx::Model::default();
        let device = Default::default();

        let data = Tensor::<TestBackend, 2>::from_floats([[0., 1.], [2., 3.], [4., 5.]], &device);
        let output = model.forward(data);

        let expected = TensorData::from([[4f32, 5.], [0., 1.]]);
        assert_eq!(output.to_data(), expected);
    }
}
