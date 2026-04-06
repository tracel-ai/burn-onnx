use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::ir::DType;

/// NodeCodegen for CastLike.
///
/// CastLike casts the first input to the same dtype as the second input.
/// By the time codegen runs, `target_type` (input[1]) has been consumed during
/// type inference and dropped from the node's input list. The target dtype is
/// stored in `config.to`. The generated code is identical to Cast's codegen.
impl NodeCodegen for onnx_ir::cast_like::CastLikeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        match (&input_arg.ty, &output_arg.ty) {
            // -----------------------
            // Scalar -> Scalar
            // -----------------------
            (ArgType::ScalarNative(input_dtype), ArgType::ScalarNative(_output_dtype)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);

                // Check if the cast is a no-op within the same dtype "family"
                let is_noop = input_dtype == &self.config.to
                    || (input_dtype.is_float()
                        && self.config.to.is_float()
                        && input_dtype != &DType::F64
                        && self.config.to != DType::F64)
                    || ((input_dtype.is_int() || input_dtype.is_uint())
                        && (self.config.to.is_int() || self.config.to.is_uint()));

                if is_noop {
                    quote! {
                        let #output = #input;
                    }
                } else {
                    let ty = match self.config.to {
                        DType::F32 | DType::F16 => quote! { f32 },
                        DType::F64 => quote! { f64 },
                        DType::I32 => quote! { i32 },
                        DType::I64 => quote! { i64 },
                        DType::U16 => quote! { u16 },
                        DType::I8 => quote! { i8 },
                        DType::U8 => quote! { u8 },
                        DType::Bool(_) => quote! { bool },
                        _ => panic!("Unsupported DType for CastLike: {:?}", self.config.to),
                    };
                    quote! {
                        let #output = #input as #ty;
                    }
                }
            }

            // -----------------------
            // Tensor -> Tensor (also covers ScalarTensor -> ScalarTensor)
            // -----------------------
            (input_ty, output_ty) if input_ty.is_on_device() && output_ty.is_on_device() => {
                let input = scope.arg(input_arg);
                let output = arg_to_ident(output_arg);

                let input_dtype = input_ty.elem_type();
                let target_dtype = self.config.to;
                let input_kind: TensorKind = input_dtype.into();
                let target_kind: TensorKind = target_dtype.into();
                let dtype_tokens = target_dtype.to_tokens();

                if input_dtype == target_dtype {
                    quote! {
                        let #output = #input;
                    }
                } else if input_kind == target_kind {
                    quote! {
                        let #output = #input.cast(#dtype_tokens);
                    }
                } else if target_kind == TensorKind::Bool {
                    quote! {
                        let #output = #input.bool();
                    }
                } else {
                    let kind_fn = match target_kind {
                        TensorKind::Int => quote! { int() },
                        TensorKind::Float => quote! { float() },
                        TensorKind::Bool => unreachable!(),
                    };
                    quote! {
                        let #output = #input.#kind_fn.cast(#dtype_tokens);
                    }
                }
            }

            // -----------------------
            // Shape -> Shape
            // -----------------------
            (ArgType::Shape(_), ArgType::Shape(_)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);
                quote! {
                    let #output = #input;
                }
            }

            // -----------------------
            // Shape -> Tensor
            // -----------------------
            (ArgType::Shape(input_rank), ArgType::Tensor(_)) => {
                let input = arg_to_ident(input_arg);
                let output = arg_to_ident(output_arg);
                let rank = *input_rank;

                match &self.config.to {
                    dtype if dtype.is_float() => {
                        let dtype_tokens = self.config.to.to_tokens();
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let float_array: [f64; #rank] = shape_array.map(|x| x as f64);
                                Tensor::<B, 1>::from_data(
                                    TensorData::from(float_array),
                                    (&self.device, #dtype_tokens)
                                )
                            };
                        }
                    }
                    dtype if dtype.is_bool() => {
                        let dtype_tokens = self.config.to.to_tokens();
                        quote! {
                            let #output = {
                                let shape_array = #input as [i64; #rank];
                                let bool_array: [bool; #rank] = shape_array.map(|x| x != 0);
                                Tensor::<B, 1, Bool>::from_data(
                                    TensorData::from(bool_array),
                                    (&self.device, #dtype_tokens)
                                )
                            };
                        }
                    }
                    dtype if dtype.is_int() || dtype.is_uint() => {
                        panic!(
                            "CastLike: Shape to Int tensor should be handled as Shape->Shape in onnx-ir"
                        )
                    }
                    _ => panic!("Unsupported DType for CastLike: {:?}", self.config.to),
                }
            }

            _ => panic!(
                "CastLike: unsupported type combination: input={:?}, output={:?}",
                input_arg.ty, output_arg.ty
            ),
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        let input_arg = self.inputs.first().unwrap();
        let output_arg = self.outputs.first().unwrap();

        if let (ArgType::Shape(_), ArgType::Tensor(_)) = (&input_arg.ty, &output_arg.ty) {
            imports.register("burn::tensor::TensorData");
            if self.config.to.is_bool() {
                imports.register("burn::tensor::Bool");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::{BoolStore, DType};
    use insta::assert_snapshot;
    use onnx_ir::cast_like::{CastLikeConfig, CastLikeNode, CastLikeNodeBuilder};

    fn create_cast_like_node_tensor(
        name: &str,
        input_dtype: DType,
        output_dtype: DType,
    ) -> CastLikeNode {
        let config = CastLikeConfig::new(output_dtype, None, None);
        CastLikeNodeBuilder::new(name)
            .input_tensor("input", 2, input_dtype)
            .output_tensor("output", 2, output_dtype)
            .config(config)
            .build()
    }

    fn create_cast_like_node_scalar(
        name: &str,
        input_dtype: DType,
        output_dtype: DType,
    ) -> CastLikeNode {
        let config = CastLikeConfig::new(output_dtype, None, None);
        CastLikeNodeBuilder::new(name)
            .input_scalar("input", input_dtype)
            .output_scalar("output", output_dtype)
            .config(config)
            .build()
    }

    #[test]
    fn test_cast_like_int_to_float() {
        let node = create_cast_like_node_tensor("cast_like1", DType::I32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = input.float().cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_float_to_int() {
        let node = create_cast_like_node_tensor("cast_like1", DType::F32, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Int> {
            let output = input.int().cast(burn::tensor::DType::I32);
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_float_to_bool() {
        let node =
            create_cast_like_node_tensor("cast_like1", DType::F32, DType::Bool(BoolStore::Native));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2, Bool> {
            let output = input.bool();
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_noop_float32_to_float32() {
        let node = create_cast_like_node_tensor("cast_like1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_scalar_int_to_float() {
        let node = create_cast_like_node_scalar("cast_like1", DType::I32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: i32) -> f32 {
            let output = input as f32;
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_float32_to_float16() {
        let node = create_cast_like_node_tensor("cast_like1", DType::F32, DType::F16);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input.cast(burn::tensor::DType::F16);
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_int64_to_int32() {
        let node = create_cast_like_node_tensor("cast_like1", DType::I64, DType::I32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2, Int> {
            let output = input.cast(burn::tensor::DType::I32);
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_scalar_noop() {
        let node = create_cast_like_node_scalar("cast_like1", DType::F32, DType::F32);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: f32) -> f32 {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_shape_to_shape() {
        let config = CastLikeConfig::new(DType::I64, None, None);
        let node = CastLikeNodeBuilder::new("cast_like1")
            .input_shape("input", 3)
            .output_shape("output", 3)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> [i64; 3] {
            let output = input;
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_shape_to_float_tensor() {
        let config = CastLikeConfig::new(DType::F32, None, None);
        let node = CastLikeNodeBuilder::new("cast_like1")
            .input_shape("input", 3)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> Tensor<B, 1> {
            let output = {
                let shape_array = input as [i64; 3usize];
                let float_array: [f64; 3usize] = shape_array.map(|x| x as f64);
                Tensor::<
                    B,
                    1,
                >::from_data(
                    TensorData::from(float_array),
                    (&self.device, burn::tensor::DType::F32),
                )
            };
            output
        }
        ");
    }

    #[test]
    fn test_cast_like_shape_to_bool_tensor() {
        let config = CastLikeConfig::new(DType::Bool(BoolStore::Native), None, None);
        let node = CastLikeNodeBuilder::new("cast_like1")
            .input_shape("input", 3)
            .output_tensor("output", 1, DType::Bool(BoolStore::Native))
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: [i64; 3]) -> Tensor<B, 1, Bool> {
            let output = {
                let shape_array = input as [i64; 3usize];
                let bool_array: [bool; 3usize] = shape_array.map(|x| x != 0);
                Tensor::<
                    B,
                    1,
                    Bool,
                >::from_data(
                    TensorData::from(bool_array),
                    (&self.device, burn::tensor::DType::Bool(burn::tensor::BoolStore::Native)),
                )
            };
            output
        }
        ");
    }
}
