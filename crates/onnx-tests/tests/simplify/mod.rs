// Tests that verify simplified and unsimplified models produce identical outputs
// but generate different code (proving simplification is effective).
//
// Each ONNX model here is purpose-built to exercise a specific simplification
// pattern. Both simplified and unsimplified codegen are tested to ensure they
// produce the same results, while insta snapshots verify the generated code
// actually differs and matches expected simplification output.

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

/// Extract the `forward` method body from generated source code.
///
/// Strips the surrounding struct/impl boilerplate and file-system-dependent paths
/// so snapshots are stable across machines and build directories.
#[cfg(test)]
fn extract_forward(source: &str) -> &str {
    // Find the forward method signature
    let start = source
        .find("pub fn forward(")
        .expect("no forward method found");
    // The forward method is the last item in the impl block, ended by closing braces
    &source[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::TestBackend;

    // -- Output equality tests --

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

    #[test]
    fn gather_shape_chain() {
        let device = Default::default();
        let s = simplified::simplify_gather_shape_chain::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_gather_shape_chain::Model::<TestBackend>::new(&device);
        let x = Tensor::<TestBackend, 3>::ones([3, 1, 4], &device);
        let y = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);
        assert_eq!(s.forward(x.clone(), y.clone()), u.forward(x, y));
    }

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

    // -- Codegen snapshot tests --
    // Verify that simplification actually changes the generated code and
    // snapshot the simplified forward() method for regression tracking.
    //
    // Only models where simplification produces different codegen are tested here.
    // Models that exercise patterns not yet optimized (e.g., standalone Shape without
    // Gather/Slice, binary ops on shapes without constant folding) are covered by
    // the output equality tests above.

    fn assert_codegen_differs(simplified: &str, unsimplified: &str, model: &str) {
        let s = extract_forward(simplified);
        let u = extract_forward(unsimplified);
        assert_ne!(
            s, u,
            "simplified and unsimplified codegen should differ for {model}"
        );
    }

    #[test]
    fn codegen_gather_on_shape() {
        let s = simplified_source::simplify_gather_on_shape();
        let u = unsimplified_source::simplify_gather_on_shape();
        assert_codegen_differs(s, u, "gather_on_shape");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, x: Tensor<B, 3>) -> i64 {
                let gather1_out1 = 3i64;
                gather1_out1
            }
        }
        ");
    }

    #[test]
    fn codegen_slice_on_shape() {
        let s = simplified_source::simplify_slice_on_shape();
        let u = unsimplified_source::simplify_slice_on_shape();
        assert_codegen_differs(s, u, "slice_on_shape");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, x: Tensor<B, 4>) -> [i64; 2] {
                let slice1_out1: [i64; 2] = [3i64, 4i64];
                slice1_out1
            }
        }
        ");
    }

    #[test]
    fn codegen_gather_shape_chain() {
        let s = simplified_source::simplify_gather_shape_chain();
        let u = unsimplified_source::simplify_gather_shape_chain();
        assert_codegen_differs(s, u, "gather_shape_chain");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, x: Tensor<B, 3>, y: Tensor<B, 3>) -> i64 {
                let gather2_out1 = 6i64;
                gather2_out1
            }
        }
        ");
    }

    #[test]
    fn codegen_permute_via_shape_gather() {
        let s = simplified_source::simplify_permute_via_shape_gather();
        let u = unsimplified_source::simplify_permute_via_shape_gather();
        assert_codegen_differs(s, u, "permute_via_shape_gather");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                let reshape1_out1 = input.permute([0, 1, 3, 2]);
                reshape1_out1
            }
        }
        ");
    }
}
