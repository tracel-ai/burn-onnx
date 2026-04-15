use crate::include_models;
include_models!(
    einsum,
    einsum_ellipsis,
    einsum_implicit,
    einsum_outer_int,
    einsum_reduction,
    einsum_sam,
    einsum_scalar,
    einsum_scalar_scalar,
    einsum_shadow_rhs
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn einsum_matmul_and_batch_matmul() {
        let model: einsum::Model<TestBackend> = einsum::Model::default();
        let device = Default::default();

        let a =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
        let b = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            &device,
        );
        let c = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
            &device,
        );
        let d = Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &device,
        );

        let (matmul_result, batch_result) = model.forward(a, b, c, d);

        let expected_matmul = TensorData::from([
            [11.0f32, 14.0, 17.0, 20.0],
            [23.0, 30.0, 37.0, 44.0],
            [35.0, 46.0, 57.0, 68.0],
        ]);
        matmul_result.to_data().assert_eq(&expected_matmul, true);

        let expected_batch =
            TensorData::from([[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 12.0], [14.0, 16.0]]]);
        batch_result.to_data().assert_eq(&expected_batch, true);
    }

    #[test]
    fn einsum_sam_pattern() {
        let model: einsum_sam::Model<TestBackend> = einsum_sam::Model::default();
        let device = Default::default();

        let r_q = Tensor::<TestBackend, 4>::from_floats(
            [[[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]],
            &device,
        );
        let r_h = Tensor::<TestBackend, 3>::from_floats(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device,
        );

        let output = model.forward(r_q, r_h);

        assert_eq!(output.dims(), [1, 2, 1, 2]);

        let expected = TensorData::from([[[[1.0f32, 4.0]], [[8.0, 11.0]]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn einsum_outer_product_int() {
        let model: einsum_outer_int::Model<TestBackend> = einsum_outer_int::Model::default();
        let device = Default::default();

        let a = Tensor::<TestBackend, 1, Int>::from_ints([1, 2, 3], &device);
        let b = Tensor::<TestBackend, 1, Int>::from_ints([4, 5, 6], &device);

        let output = model.forward(a, b);

        let expected = TensorData::from([[4i32, 5, 6], [8, 10, 12], [12, 15, 18]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn einsum_scalar_operands() {
        let model: einsum_scalar::Model<TestBackend> = einsum_scalar::Model::default();
        let device = Default::default();

        let matrix = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let (lhs_scaled, rhs_scaled) = model.forward(2.0, matrix.clone(), 3.0);

        let expected_lhs = TensorData::from([[2.0f32, 4.0], [6.0, 8.0]]);
        lhs_scaled.to_data().assert_eq(&expected_lhs, true);

        let expected_rhs = TensorData::from([[3.0f32, 6.0], [9.0, 12.0]]);
        rhs_scaled.to_data().assert_eq(&expected_rhs, true);
    }

    #[test]
    fn einsum_scalar_scalar_operand() {
        let model: einsum_scalar_scalar::Model<TestBackend> =
            einsum_scalar_scalar::Model::default();
        let output = model.forward(2.0, 3.0);
        assert_eq!(output, 6.0f32);
    }

    #[test]
    fn einsum_implicit_form() {
        let model: einsum_implicit::Model<TestBackend> = einsum_implicit::Model::default();
        let device = Default::default();

        let a =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
        let b = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let output = model.forward(a, b);

        // "ij,jk" is equivalent to "ij,jk->ik" (standard matmul)
        let expected = TensorData::from([[7.0f32, 10.0], [15.0, 22.0], [23.0, 34.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn einsum_one_sided_reduction() {
        let model: einsum_reduction::Model<TestBackend> = einsum_reduction::Model::default();
        let device = Default::default();

        let a = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let b = Tensor::<TestBackend, 2>::from_floats([[10.0, 20.0, 30.0]], &device);

        let output = model.forward(a, b);

        // "ij,kl->il": output[i,l] = sum_j(a[i,j]) * sum_k(b[k,l])
        // sum_j(a[0,:]) = 3, sum_j(a[1,:]) = 7
        // sum_k(b[:,0]) = 10, sum_k(b[:,1]) = 20, sum_k(b[:,2]) = 30
        let expected = TensorData::from([[30.0f32, 60.0, 90.0], [70.0, 140.0, 210.0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn einsum_ellipsis_batch_matmul() {
        let model: einsum_ellipsis::Model<TestBackend> = einsum_ellipsis::Model::default();
        let device = Default::default();

        // "...ij,...jk->...ik" with rank 4 (2 batch dims)
        // Use identity-like matrices for easy verification
        let a = Tensor::<TestBackend, 4>::from_floats(
            [[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]],
            &device,
        );
        let b = Tensor::<TestBackend, 4>::from_floats(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
            &device,
        );

        let output = model.forward(a, b);
        assert_eq!(output.dims(), [1, 2, 2, 2]);

        let expected =
            TensorData::from([[[[1.0f32, 2.0], [3.0, 4.0]], [[10.0, 12.0], [14.0, 16.0]]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn einsum_shadow_rhs_reuses_original_input() {
        let model: einsum_shadow_rhs::Model<TestBackend> = einsum_shadow_rhs::Model::default();
        let device = Default::default();

        let lhs =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let einsum_rhs = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
            &device,
        );

        let output = model.forward(lhs, einsum_rhs);

        let expected = TensorData::from([[432.0f32, 864.0, 1296.0], [765.0, 1530.0, 2295.0]]);
        output.to_data().assert_eq(&expected, true);
    }
}
