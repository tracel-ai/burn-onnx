// Tests that verify simplified and unsimplified models produce identical outputs.
//
// Each ONNX model here is purpose-built to exercise a specific simplification
// pattern. Both simplified and unsimplified codegen are tested to ensure they
// produce the same results.

use crate::include_simplified_models;

include_simplified_models!(
    simplify_shape_folding,
    simplify_gather_on_shape,
    simplify_slice_on_shape,
    simplify_concat_shapes,
    simplify_reshape_from_shape,
    simplify_binary_ops_on_shape,
    simplify_cast_shape,
    simplify_where_on_shapes,
    simplify_expand_from_shape,
    simplify_constant_of_shape_opt,
    simplify_gather_shape_chain,
    simplify_permute_via_shape_gather
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::TestBackend;

    #[test]
    fn shape_folding() {
        let device = Default::default();
        let s = simplified::simplify_shape_folding::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_shape_folding::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(s.forward(input.clone()), u.forward(input));
    }

    #[test]
    fn gather_on_shape() {
        let device = Default::default();
        let s = simplified::simplify_gather_on_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_gather_on_shape::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(s.forward(input.clone()), u.forward(input));
    }

    #[test]
    fn slice_on_shape() {
        let device = Default::default();
        let s = simplified::simplify_slice_on_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_slice_on_shape::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 4>::ones([2, 3, 4, 5], &device);
        assert_eq!(s.forward(input.clone()), u.forward(input));
    }

    #[test]
    fn concat_shapes() {
        let device = Default::default();
        let s = simplified::simplify_concat_shapes::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_concat_shapes::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 2>::ones([2, 3], &device);
        let y = Tensor::<TestBackend, 3>::ones([4, 5, 6], &device);
        assert_eq!(s.forward(x.clone(), y.clone()), u.forward(x, y));
    }

    #[test]
    fn reshape_from_shape() {
        let device = Default::default();
        let s = simplified::simplify_reshape_from_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_reshape_from_shape::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 1>::from_floats(
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            &device,
        );
        let shape_source = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        assert_eq!(
            s.forward(x.clone(), shape_source.clone()).to_data(),
            u.forward(x, shape_source).to_data()
        );
    }

    #[test]
    fn binary_ops_on_shape() {
        let device = Default::default();
        let s = simplified::simplify_binary_ops_on_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_binary_ops_on_shape::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(s.forward(input.clone()), u.forward(input));
    }

    #[test]
    fn cast_shape() {
        let device = Default::default();
        let s = simplified::simplify_cast_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_cast_shape::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(
            s.forward(input.clone()).to_data(),
            u.forward(input).to_data()
        );
    }

    #[test]
    fn where_on_shapes() {
        let device = Default::default();
        let s = simplified::simplify_where_on_shapes::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_where_on_shapes::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let y = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);
        assert_eq!(s.forward(true, x.clone(), y.clone()), u.forward(true, x, y));
    }

    #[test]
    fn expand_from_shape() {
        let device = Default::default();
        let s = simplified::simplify_expand_from_shape::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_expand_from_shape::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 3>::ones([1, 1, 4], &device);
        let shape_source = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(
            s.forward(x.clone(), shape_source.clone()).to_data(),
            u.forward(x, shape_source).to_data()
        );
    }

    /// Chained Shape->Gather where first Gather's result is the second Gather's index.
    /// Tests that value_store propagation enables multi-iteration constant folding.
    #[test]
    fn gather_shape_chain() {
        let device = Default::default();
        let s = simplified::simplify_gather_shape_chain::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_gather_shape_chain::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 3>::ones([3, 1, 4], &device);
        let y = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);
        assert_eq!(s.forward(x.clone(), y.clone()), u.forward(x, y));
    }

    /// Shape->Gather->Unsqueeze->Concat->Reshape that transposes last two dims.
    /// Should simplify to a Transpose.
    #[test]
    fn permute_via_shape_gather() {
        let device = Default::default();
        let s = simplified::simplify_permute_via_shape_gather::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_permute_via_shape_gather::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 4>::ones([2, 3, 4, 5], &device);
        assert_eq!(
            s.forward(input.clone()).to_data(),
            u.forward(input).to_data()
        );
    }

    #[test]
    fn constant_of_shape_opt() {
        let device = Default::default();
        let s = simplified::simplify_constant_of_shape_opt::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_constant_of_shape_opt::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 2>::ones([2, 3], &device);
        assert_eq!(
            s.forward(input.clone()).to_data(),
            u.forward(input).to_data()
        );
    }
}
