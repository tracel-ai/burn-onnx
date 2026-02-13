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
    simplify_permute_via_shape_gather,
    simplify_sdpa_coalesce,
    simplify_constant_fold
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

    #[test]
    fn sdpa_coalesce() {
        use burn::tensor::{Tolerance, ops::FloatElem};
        type FT = FloatElem<TestBackend>;
        let device = Default::default();
        let s = simplified::simplify_sdpa_coalesce::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_sdpa_coalesce::Model::<TestBackend>::new(&device);
        // Use distinct tensors to catch Q/K/V ordering bugs
        let q = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 1.1, 1.2],
                ],
            ]],
            &device,
        );
        let k = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [0.1, 0.3, 0.5, 0.7],
                    [0.2, 0.4, 0.6, 0.8],
                    [0.9, 1.1, 1.3, 1.5],
                ],
                [
                    [1.0, 1.2, 1.4, 1.6],
                    [1.1, 1.3, 1.5, 1.7],
                    [1.8, 2.0, 2.2, 2.4],
                ],
            ]],
            &device,
        );
        let v = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [2.0, 0.5, 1.0, 3.0],
                    [1.5, 2.5, 0.5, 1.0],
                    [0.5, 1.5, 2.5, 0.5],
                ],
                [
                    [3.0, 1.0, 2.0, 0.5],
                    [2.5, 0.5, 1.5, 2.0],
                    [1.0, 3.0, 0.5, 1.5],
                ],
            ]],
            &device,
        );
        let s_out = s.forward(q.clone(), k.clone(), v.clone());
        let u_out = u.forward(q, k, v);
        s_out
            .to_data()
            .assert_approx_eq::<FT>(&u_out.to_data(), Tolerance::default());
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
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(&self, x: Tensor<B, 3>) -> i64 {
                let shape1_out1: [i64; 3] = {
                    let axes = &x.dims()[0..3];
                    let mut output = [0i64; 3];
                    for i in 0..3 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let constant1_out1 = 1i64;
                let actual_idx = if constant1_out1 < 0 {
                    (shape1_out1.len() as i64 + constant1_out1) as usize
                } else {
                    constant1_out1 as usize
                };
                let gather1_out1 = shape1_out1[actual_idx] as i64;
                gather1_out1
            }
        }
        ");
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
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(&self, x: Tensor<B, 4>) -> [i64; 2] {
                let shape1_out1: [i64; 4] = {
                    let axes = &x.dims()[0..4];
                    let mut output = [0i64; 4];
                    for i in 0..4 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let slice1_out1: [i64; 2] = shape1_out1[1..3].try_into().unwrap();
                slice1_out1
            }
        }
        ");
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
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(&self, x: Tensor<B, 3>, y: Tensor<B, 3>) -> i64 {
                let shape1_out1: [i64; 3] = {
                    let axes = &x.dims()[0..3];
                    let mut output = [0i64; 3];
                    for i in 0..3 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let constant1_out1 = 1i64;
                let actual_idx = if constant1_out1 < 0 {
                    (shape1_out1.len() as i64 + constant1_out1) as usize
                } else {
                    constant1_out1 as usize
                };
                let gather1_out1 = shape1_out1[actual_idx] as i64;
                let shape2_out1: [i64; 3] = {
                    let axes = &y.dims()[0..3];
                    let mut output = [0i64; 3];
                    for i in 0..3 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let actual_idx = if gather1_out1 < 0 {
                    (shape2_out1.len() as i64 + gather1_out1) as usize
                } else {
                    gather1_out1 as usize
                };
                let gather2_out1 = shape2_out1[actual_idx] as i64;
                gather2_out1
            }
        }
        ");
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
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                let shape1_out1: [i64; 4] = {
                    let axes = &input.clone().dims()[0..4];
                    let mut output = [0i64; 4];
                    for i in 0..4 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let shape2_out1: [i64; 4] = {
                    let axes = &input.clone().dims()[0..4];
                    let mut output = [0i64; 4];
                    for i in 0..4 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let shape3_out1: [i64; 4] = {
                    let axes = &input.clone().dims()[0..4];
                    let mut output = [0i64; 4];
                    for i in 0..4 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let shape4_out1: [i64; 4] = {
                    let axes = &input.clone().dims()[0..4];
                    let mut output = [0i64; 4];
                    for i in 0..4 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let constant1_out1 = 0i64;
                let constant2_out1 = 1i64;
                let constant3_out1 = 3i64;
                let constant4_out1 = 2i64;
                let actual_idx = if constant1_out1 < 0 {
                    (shape1_out1.len() as i64 + constant1_out1) as usize
                } else {
                    constant1_out1 as usize
                };
                let gather1_out1 = shape1_out1[actual_idx] as i64;
                let actual_idx = if constant2_out1 < 0 {
                    (shape2_out1.len() as i64 + constant2_out1) as usize
                } else {
                    constant2_out1 as usize
                };
                let gather2_out1 = shape2_out1[actual_idx] as i64;
                let actual_idx = if constant3_out1 < 0 {
                    (shape3_out1.len() as i64 + constant3_out1) as usize
                } else {
                    constant3_out1 as usize
                };
                let gather3_out1 = shape3_out1[actual_idx] as i64;
                let actual_idx = if constant4_out1 < 0 {
                    (shape4_out1.len() as i64 + constant4_out1) as usize
                } else {
                    constant4_out1 as usize
                };
                let gather4_out1 = shape4_out1[actual_idx] as i64;
                let unsqueeze1_out1 = [gather1_out1];
                let unsqueeze2_out1 = [gather2_out1];
                let unsqueeze3_out1 = [gather3_out1];
                let unsqueeze4_out1 = [gather4_out1];
                let concat1_out1: [i64; 4usize] = [
                    &unsqueeze1_out1[..],
                    &unsqueeze2_out1[..],
                    &unsqueeze3_out1[..],
                    &unsqueeze4_out1[..],
                ]
                    .concat()
                    .try_into()
                    .unwrap();
                let reshape1_out1 = input.reshape(concat1_out1);
                reshape1_out1
            }
        }
        ");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
                let reshape1_out1 = input.permute([0, 1, 3, 2]);
                reshape1_out1
            }
        }
        ");
    }

    #[test]
    fn constant_fold() {
        let device = Default::default();
        let s = simplified::simplify_constant_fold::Model::<TestBackend>::new(&device);
        let u = unsimplified::simplify_constant_fold::Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        assert_eq!(s.forward(input.clone()), u.forward(input));
    }

    #[test]
    fn codegen_constant_fold() {
        let s = simplified_source::simplify_constant_fold();
        let u = unsimplified_source::simplify_constant_fold();
        assert_codegen_differs(s, u, "constant_fold");
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(&self, x: Tensor<B, 3>) -> i64 {
                let shape1_out1: [i64; 3] = {
                    let axes = &x.clone().dims()[0..3];
                    let mut output = [0i64; 3];
                    for i in 0..3 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let shape2_out1: [i64; 3] = {
                    let axes = &x.dims()[0..3];
                    let mut output = [0i64; 3];
                    for i in 0..3 {
                        output[i] = axes[i] as i64;
                    }
                    output
                };
                let constant1_out1 = 1i64;
                let constant2_out1 = 2i64;
                let actual_idx = if constant1_out1 < 0 {
                    (shape1_out1.len() as i64 + constant1_out1) as usize
                } else {
                    constant1_out1 as usize
                };
                let gather1_out1 = shape1_out1[actual_idx] as i64;
                let actual_idx = if constant2_out1 < 0 {
                    (shape2_out1.len() as i64 + constant2_out1) as usize
                } else {
                    constant2_out1 as usize
                };
                let gather2_out1 = shape2_out1[actual_idx] as i64;
                let mul1_out1 = gather1_out1 * gather2_out1;
                mul1_out1
            }
        }
        ");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(&self, x: Tensor<B, 3>) -> i64 {
                let mul1_out1 = 12i64;
                mul1_out1
            }
        }
        ");
    }

    #[test]
    fn codegen_sdpa_coalesce() {
        let s = simplified_source::simplify_sdpa_coalesce();
        let u = unsimplified_source::simplify_sdpa_coalesce();
        assert_codegen_differs(s, u, "sdpa_coalesce");
        insta::assert_snapshot!(extract_forward(u), @r"
        pub fn forward(
                &self,
                q: Tensor<B, 4>,
                k: Tensor<B, 4>,
                v: Tensor<B, 4>,
            ) -> Tensor<B, 4> {
                let transpose1_out1 = k.permute([0, 1, 3, 2]);
                let matmul1_out1 = q.matmul(transpose1_out1);
                let constant1_out1 = 2f32;
                let div1_out1 = matmul1_out1.div_scalar(constant1_out1);
                let softmax1_out1 = burn::tensor::activation::softmax(div1_out1, 3);
                let matmul2_out1 = softmax1_out1.matmul(v);
                matmul2_out1
            }
        }
        ");
        insta::assert_snapshot!(extract_forward(s), @r"
        pub fn forward(
                &self,
                q: Tensor<B, 4>,
                k: Tensor<B, 4>,
                v: Tensor<B, 4>,
            ) -> Tensor<B, 4> {
                let (matmul2_out1,) = {
                    let q = q;
                    let k = k;
                    let v = v;
                    let matmul2_out1 = burn::tensor::module::attention(
                        q,
                        k,
                        v,
                        None,
                        None,
                        burn::tensor::ops::AttentionOptions {
                            scale: Some(0.5f64),
                            softcap: None,
                            is_causal: false,
                        },
                    );
                    (matmul2_out1,)
                };
                matmul2_out1
            }
        }
        ");
    }
}
