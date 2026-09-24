use super::prelude::*;
use crate::burn::argument_helpers::{elem_cast_tokens, tensor_type_tokens};

fn compile_error_tokens(message: impl Into<String>) -> TokenStream {
    let message = message.into();
    quote! {
        compile_error!(#message);
    }
}

/// Lift a native scalar operand onto the device as the shape-`[1]` tensor Burn's
/// einsum expects for an empty term.
fn scalar_native_to_tensor(expr: TokenStream, dtype: DType) -> TokenStream {
    let dtype_tokens = dtype.to_tokens();

    // Promote through a wide host literal and let Burn cast to the requested dtype.
    if matches!(dtype, DType::F16 | DType::BF16) {
        quote! {
            Tensor::<1>::from_data(
                burn::tensor::TensorData::from([(#expr).to_f64()]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F32) {
        quote! {
            Tensor::<1>::from_data(
                burn::tensor::TensorData::from([f64::from(#expr)]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if matches!(dtype, DType::F64) {
        quote! {
            Tensor::<1>::from_data(
                burn::tensor::TensorData::from([#expr]),
                (&self.device, #dtype_tokens)
            )
        }
    } else if dtype.is_int() || dtype.is_uint() {
        quote! {
            Tensor::<1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::from([#expr as i64]),
                (&self.device, #dtype_tokens)
            )
        }
    } else {
        compile_error_tokens(format!("Einsum does not support scalar dtype {:?}", dtype))
    }
}

/// The equation as Burn's einsum needs it. An ONNX scalar operand may carry a
/// zero-width `...` term, but Burn passes scalars as shape `[1]`, where an ellipsis
/// would claim that axis, so the `...` is dropped from scalar terms (and from the
/// output when no other term keeps one).
fn burn_equation(equation: &str, inputs: &[Argument]) -> String {
    let (terms, output) = match equation.split_once("->") {
        Some((terms, output)) => (terms, Some(output)),
        None => (equation, None),
    };
    let terms: Vec<String> = terms
        .split(',')
        .zip(inputs)
        .map(|(term, input)| {
            if input.ty.is_scalar() {
                term.replace("...", "")
            } else {
                term.to_string()
            }
        })
        .collect();
    let mut rewritten = terms.join(",");
    if let Some(output) = output {
        rewritten.push_str("->");
        if terms.iter().any(|term| term.contains("...")) {
            rewritten.push_str(output);
        } else {
            rewritten.push_str(&output.replace("...", ""));
        }
    }
    rewritten
}

impl NodeCodegen for onnx_ir::node::einsum::EinsumNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let [output_arg] = self.outputs.as_slice() else {
            return compile_error_tokens(format!(
                "Einsum node '{}' expects exactly 1 output, got {}",
                self.name,
                self.outputs.len()
            ));
        };
        let output = arg_to_ident(output_arg);
        let equation = burn_equation(&self.config.equation, &self.inputs);

        let mut operands = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let value = scope.arg(input);
            let tensor = match &input.ty {
                ArgType::Tensor(_) | ArgType::ScalarTensor(_) => value,
                ArgType::ScalarNative(dtype) => scalar_native_to_tensor(value, *dtype),
                other => {
                    return compile_error_tokens(format!(
                        "Einsum node '{}' requires tensor or scalar inputs, got {:?}",
                        self.name, other
                    ));
                }
            };
            operands.push(quote! { #tensor.into() });
        }

        // Burn represents a scalar result as a shape-[1] tensor.
        let dtype = output_arg.ty.elem_type();
        let (rank, readback) = match &output_arg.ty {
            ArgType::Tensor(tensor) => (tensor.rank, None),
            ArgType::ScalarTensor(_) => (1, None),
            ArgType::ScalarNative(dtype) => (1, Some(elem_cast_tokens(dtype))),
            other => {
                return compile_error_tokens(format!(
                    "Einsum node '{}' does not support output type {:?}",
                    self.name, other
                ));
            }
        };
        let ty = tensor_type_tokens(rank, &dtype);
        let einsum = quote! { Tensor::einsum(#equation, [#(#operands),*]) };

        match readback {
            None => quote! {
                let #output: #ty = #einsum;
            },
            Some(readback) => quote! {
                let #output = {
                    let result: #ty = #einsum;
                    result #readback
                };
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::einsum::{EinsumConfig, EinsumNodeBuilder};

    fn config(equation: &str) -> EinsumConfig {
        EinsumConfig {
            equation: equation.to_string(),
        }
    }

    #[test]
    fn test_einsum_matmul() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 2, DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config("ij,jk->ik"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum("ij,jk->ik", [lhs.into(), rhs.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_int() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 3, DType::I64)
            .input_tensor("rhs", 3, DType::I64)
            .output_tensor("output", 3, DType::I64)
            .config(config("bij,bjk->bik"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<3, Int>, rhs: Tensor<3, Int>) -> Tensor<3, Int> {
            let output: Tensor<3, Int> = Tensor::einsum(
                "bij,bjk->bik",
                [lhs.into(), rhs.into()],
            );
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_single_input() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config("...ii->...i"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, x: Tensor<3>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum("...ii->...i", [x.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_three_inputs() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("a", 2, DType::F32)
            .input_tensor("b", 2, DType::F32)
            .input_tensor("c", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config("ij,jk,kl->il"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, a: Tensor<2>, b: Tensor<2>, c: Tensor<2>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum(
                "ij,jk,kl->il",
                [a.into(), b.into(), c.into()],
            );
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_native_operand() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_scalar("scale", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config(",ij->ij"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, scale: f32, rhs: Tensor<2>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum(
                ",ij->ij",
                [
                    Tensor::<
                        1,
                    >::from_data(
                            burn::tensor::TensorData::from([f64::from(scale)]),
                            (&self.device, burn::tensor::DType::F32),
                        )
                        .into(),
                    rhs.into(),
                ],
            );
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_float16_scalar_native_operand() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_scalar("scale", DType::F16)
            .input_tensor("rhs", 2, DType::F16)
            .output_tensor("output", 2, DType::F16)
            .config(config(",ij->ij"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, scale: half::f16, rhs: Tensor<2>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum(
                ",ij->ij",
                [
                    Tensor::<
                        1,
                    >::from_data(
                            burn::tensor::TensorData::from([(scale).to_f64()]),
                            (&self.device, burn::tensor::DType::F16),
                        )
                        .into(),
                    rhs.into(),
                ],
            );
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_tensor_operand() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 2, DType::F32)
            .input_scalar_tensor("scale", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config("ij,->ij"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<2>, scale: Tensor<1>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum("ij,->ij", [lhs.into(), scale.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_operand_with_ellipsis() {
        // The scalar's `...` is zero-width in ONNX; it and the output's `...` are
        // dropped since no other term has one.
        let node = EinsumNodeBuilder::new("einsum1")
            .input_scalar_tensor("scale", DType::F32)
            .input_tensor("rhs", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config("...,ij->...ij"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, scale: Tensor<1>, rhs: Tensor<2>) -> Tensor<2> {
            let output: Tensor<2> = Tensor::einsum(",ij->ij", [scale.into(), rhs.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_operand_with_ellipsis_kept_elsewhere() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_scalar_tensor("scale", DType::F32)
            .input_tensor("rhs", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config("...,...ij->...ij"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, scale: Tensor<1>, rhs: Tensor<3>) -> Tensor<3> {
            let output: Tensor<3> = Tensor::einsum(",...ij->...ij", [scale.into(), rhs.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_tensor_output() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_tensor("lhs", 1, DType::F32)
            .input_tensor("rhs", 1, DType::F32)
            .output_scalar_tensor("output", DType::F32)
            .config(config("i,i->"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: Tensor<1>, rhs: Tensor<1>) -> Tensor<1> {
            let output: Tensor<1> = Tensor::einsum("i,i->", [lhs.into(), rhs.into()]);
            output
        }
        "#);
    }

    #[test]
    fn test_einsum_scalar_native_output() {
        let node = EinsumNodeBuilder::new("einsum1")
            .input_scalar("lhs", DType::F32)
            .input_scalar("rhs", DType::F32)
            .output_scalar("output", DType::F32)
            .config(config(",->"))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, lhs: f32, rhs: f32) -> f32 {
            let output = {
                let result: Tensor<1> = Tensor::einsum(
                    ",->",
                    [
                        Tensor::<
                            1,
                        >::from_data(
                                burn::tensor::TensorData::from([f64::from(lhs)]),
                                (&self.device, burn::tensor::DType::F32),
                            )
                            .into(),
                        Tensor::<
                            1,
                        >::from_data(
                                burn::tensor::TensorData::from([f64::from(rhs)]),
                                (&self.device, burn::tensor::DType::F32),
                            )
                            .into(),
                    ],
                );
                result.into_scalar::<f32>()
            };
            output
        }
        "#);
    }
}
