use super::prelude::*;

impl NodeCodegen for onnx_ir::matmulinteger::MatMulIntegerNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let lhs = scope.arg(self.inputs.first().unwrap());
        let rhs = scope.arg(self.inputs.get(1).unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // MatMulInteger output dtype (always I32 per ONNX spec).
        // Cast all operands to this dtype to ensure type compatibility,
        // since inputs may be U8/I8 and zero points may differ.
        let output_dtype = self.outputs.first().unwrap().ty.elem_type().to_tokens();

        // Get ranks for handling broadcasting
        let lhs_rank = match &self.inputs.first().unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for lhs"),
        };
        let rhs_rank = match &self.inputs.get(1).unwrap().ty {
            onnx_ir::ir::ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for rhs"),
        };

        // Centered inputs: cast to output dtype, subtract zero-points if provided.
        // When zero-points are absent (default 0), just cast without subtraction.
        let lhs_centered = if let Some(zp_input) = self.inputs.get(2) {
            let zp = scope.arg(zp_input);
            let zp_expr = if lhs_rank > 1 {
                quote! { (#zp).cast(#output_dtype).unsqueeze::<#lhs_rank>() }
            } else {
                quote! { (#zp).cast(#output_dtype) }
            };
            quote! { (#lhs).cast(#output_dtype).sub(#zp_expr) }
        } else {
            quote! { (#lhs).cast(#output_dtype) }
        };

        let rhs_centered = if let Some(zp_input) = self.inputs.get(3) {
            let zp = scope.arg(zp_input);
            let zp_expr = if rhs_rank > 1 {
                quote! { (#zp).cast(#output_dtype).unsqueeze::<#rhs_rank>() }
            } else {
                quote! { (#zp).cast(#output_dtype) }
            };
            quote! { (#rhs).cast(#output_dtype).sub(#zp_expr) }
        } else {
            quote! { (#rhs).cast(#output_dtype) }
        };

        // Handle rank differences for matmul broadcasting
        match lhs_rank.cmp(&rhs_rank) {
            std::cmp::Ordering::Greater => {
                let num_unsqueezes = lhs_rank - rhs_rank;

                if rhs_rank == 1 {
                    // Matrix-vector product: expand vector to match matrix rank
                    let squeeze_dim = lhs_rank - 1;
                    let out_rank = lhs_rank - 1;

                    // Build unsqueeze dimensions: [-1, 0, 0, ...]
                    let mut unsqueeze_dims = vec![-1isize];
                    if num_unsqueezes > 1 {
                        unsqueeze_dims.extend(std::iter::repeat_n(0isize, num_unsqueezes - 1));
                    }

                    quote! {
                        let #output = (#lhs_centered).matmul((#rhs_centered).unsqueeze_dims(&[#(#unsqueeze_dims),*])).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = lhs_rank;
                    quote! {
                        let #output = (#lhs_centered).matmul((#rhs_centered).unsqueeze::<#target_rank>());
                    }
                }
            }
            std::cmp::Ordering::Less => {
                if lhs_rank == 1 {
                    // Vector-matrix product: expand vector to match matrix rank
                    let squeeze_dim = rhs_rank - 2;
                    let out_rank = rhs_rank - 1;
                    let target_rank = rhs_rank;
                    quote! {
                        let #output = (#lhs_centered).unsqueeze::<#target_rank>().matmul(#rhs_centered).squeeze_dim::<#out_rank>(#squeeze_dim);
                    }
                } else {
                    // General tensor broadcasting: add leading dimensions
                    let target_rank = rhs_rank;
                    quote! {
                        let #output = (#lhs_centered).unsqueeze::<#target_rank>().matmul(#rhs_centered);
                    }
                }
            }
            std::cmp::Ordering::Equal => {
                quote! {
                    let #output = (#lhs_centered).matmul(#rhs_centered);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::matmulinteger::MatMulIntegerNodeBuilder;

    #[test]
    fn test_matmul_integer_same_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint1")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2, Int>, b: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .matmul((b).cast(burn::tensor::DType::I32));
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_with_zero_points() {
        let node = MatMulIntegerNodeBuilder::new("mmint2")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .input_tensor("a_zero_point", 2, DType::I32)
            .input_tensor("b_zero_point", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            b: Tensor<B, 2, Int>,
            a_zero_point: Tensor<B, 2, Int>,
            b_zero_point: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = ((a)
                .cast(burn::tensor::DType::I32)
                .sub((a_zero_point).cast(burn::tensor::DType::I32).unsqueeze::<2usize>()))
                .matmul(
                    (b)
                        .cast(burn::tensor::DType::I32)
                        .sub((b_zero_point).cast(burn::tensor::DType::I32).unsqueeze::<2usize>()),
                );
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_lhs_zero_point_only() {
        let node = MatMulIntegerNodeBuilder::new("mmint3")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .input_tensor("a_zero_point", 2, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 2, Int>,
            b: Tensor<B, 2, Int>,
            a_zero_point: Tensor<B, 2, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = ((a)
                .cast(burn::tensor::DType::I32)
                .sub((a_zero_point).cast(burn::tensor::DType::I32).unsqueeze::<2usize>()))
                .matmul((b).cast(burn::tensor::DType::I32));
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_lhs_greater_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint4")
            .input_tensor("a", 3, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3, Int>, b: Tensor<B, 2, Int>) -> Tensor<B, 3, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .matmul(((b).cast(burn::tensor::DType::I32)).unsqueeze::<3usize>());
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_rhs_greater_rank() {
        let node = MatMulIntegerNodeBuilder::new("mmint5")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 3, DType::I32)
            .output_tensor("output", 3, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2, Int>, b: Tensor<B, 3, Int>) -> Tensor<B, 3, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .unsqueeze::<3usize>()
                .matmul((b).cast(burn::tensor::DType::I32));
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_matrix_vector() {
        let node = MatMulIntegerNodeBuilder::new("mmint6")
            .input_tensor("a", 2, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 2, Int>, b: Tensor<B, 1, Int>) -> Tensor<B, 1, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .matmul(((b).cast(burn::tensor::DType::I32)).unsqueeze_dims(&[-1isize]))
                .squeeze_dim::<1usize>(1usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_vector_matrix() {
        let node = MatMulIntegerNodeBuilder::new("mmint7")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 2, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 1, Int>, b: Tensor<B, 2, Int>) -> Tensor<B, 1, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .unsqueeze::<2usize>()
                .matmul((b).cast(burn::tensor::DType::I32))
                .squeeze_dim::<1usize>(0usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_3d_vector() {
        let node = MatMulIntegerNodeBuilder::new("mmint8")
            .input_tensor("a", 3, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .output_tensor("output", 2, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, a: Tensor<B, 3, Int>, b: Tensor<B, 1, Int>) -> Tensor<B, 2, Int> {
            let output = ((a).cast(burn::tensor::DType::I32))
                .matmul(((b).cast(burn::tensor::DType::I32)).unsqueeze_dims(&[-1isize, 0isize]))
                .squeeze_dim::<2usize>(2usize);
            output
        }
        ");
    }

    #[test]
    fn test_matmul_integer_zero_points_scalar_rank1() {
        let node = MatMulIntegerNodeBuilder::new("mmint9")
            .input_tensor("a", 1, DType::I32)
            .input_tensor("b", 1, DType::I32)
            .input_tensor("a_zero_point", 1, DType::I32)
            .input_tensor("b_zero_point", 1, DType::I32)
            .output_tensor("output", 1, DType::I32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            a: Tensor<B, 1, Int>,
            b: Tensor<B, 1, Int>,
            a_zero_point: Tensor<B, 1, Int>,
            b_zero_point: Tensor<B, 1, Int>,
        ) -> Tensor<B, 1, Int> {
            let output = ((a)
                .cast(burn::tensor::DType::I32)
                .sub((a_zero_point).cast(burn::tensor::DType::I32)))
                .matmul(
                    (b)
                        .cast(burn::tensor::DType::I32)
                        .sub((b_zero_point).cast(burn::tensor::DType::I32)),
                );
            output
        }
        ");
    }
}
