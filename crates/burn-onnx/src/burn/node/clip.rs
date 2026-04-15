use super::prelude::*;

/// Which native scalar type a runtime Clip bound should be cast to.
/// Picked once from the input tensor's dtype so the bound stays in a
/// type that can represent every value the input can hold:
///
/// - `I64` for signed integer inputs (preserves the full i64 range).
/// - `U64` for unsigned integer inputs (preserves the full u64 range,
///   including values above i64::MAX that would wrap through `i64`).
/// - `F64` for float inputs.
#[derive(Clone, Copy)]
enum ClipBoundCast {
    I64,
    U64,
    F64,
}

impl ClipBoundCast {
    fn from_dtype(dtype: burn::tensor::DType) -> Self {
        if dtype.is_uint() {
            Self::U64
        } else if dtype.is_int() {
            Self::I64
        } else {
            Self::F64
        }
    }

    fn tokens(self) -> TokenStream {
        match self {
            Self::I64 => quote! { i64 },
            Self::U64 => quote! { u64 },
            Self::F64 => quote! { f64 },
        }
    }
}

/// Build a token stream that evaluates to a native scalar bound at runtime
/// for a single Clip `min`/`max` input.
///
/// Static bounds are inlined as literals. Runtime bounds come from one of
/// the node's inputs, which is either a native scalar (use directly) or a
/// scalar tensor (extract via `into_scalar().elem()`). The scalar is then
/// coerced via an `as` cast to the native type selected by `bound_cast`,
/// which mirrors the element type of the tensor being clipped. The caller
/// computes `bound_cast` once from the input tensor's dtype so the min
/// and max bounds agree, and so the cast is always wide enough to
/// represent every value in the input's dtype (u64 inputs in particular
/// need `u64`, since `i64` would wrap at i64::MAX).
fn clip_bound_expr(
    bound: &Option<onnx_ir::node::clip::ClipInput>,
    inputs: &[Argument],
    scope: &mut ScopeAtPosition<'_>,
    bound_cast: ClipBoundCast,
) -> Option<TokenStream> {
    match bound {
        None => None,
        Some(onnx_ir::node::clip::ClipInput::Static(v)) => {
            let v = *v;
            Some(quote! { #v })
        }
        Some(onnx_ir::node::clip::ClipInput::Runtime(r)) => {
            let arg = &inputs[r.input_index];
            let cast_ty = bound_cast.tokens();
            match &arg.ty {
                ArgType::ScalarNative(_) => {
                    let ident = arg_to_ident(arg);
                    Some(quote! { (#ident as #cast_ty) })
                }
                ArgType::ScalarTensor(dtype) => {
                    let tensor = scope.arg(arg);
                    let native = on_device_to_native(quote! { #tensor }, dtype);
                    Some(quote! { (#native as #cast_ty) })
                }
                other => panic!(
                    "Clip min/max must be a scalar (ScalarNative or ScalarTensor), got {other:?}"
                ),
            }
        }
    }
}

impl NodeCodegen for onnx_ir::clip::ClipNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // The input dtype determines whether runtime bounds should be
        // carried as i64, u64, or f64. Picking the widest type that can
        // hold every value in the input's dtype avoids silently wrapping
        // large bounds through a narrower intermediate.
        let input_dtype = self.inputs.first().unwrap().ty.elem_type();
        let bound_cast = ClipBoundCast::from_dtype(input_dtype);

        // Extract bound expressions first so the `match (min_expr, max_expr)`
        // below is a straight token-stream assembly rather than a nested
        // branch on the enum structure.
        let min_expr = clip_bound_expr(&self.config.min, &self.inputs, scope, bound_cast);
        let max_expr = clip_bound_expr(&self.config.max, &self.inputs, scope, bound_cast);
        let input = scope.arg(self.inputs.first().unwrap());

        match (min_expr, max_expr) {
            (Some(min), Some(max)) => quote! {
                let #output = {
                    let __clip_min = #min;
                    let __clip_max = #max;
                    #input.clamp(__clip_min, __clip_max)
                };
            },
            (Some(min), None) => quote! {
                let #output = {
                    let __clip_min = #min;
                    #input.clamp_min(__clip_min)
                };
            },
            (None, Some(max)) => quote! {
                let #output = {
                    let __clip_max = #max;
                    #input.clamp_max(__clip_max)
                };
            },
            (None, None) => panic!("Clip node must have at least one min or max value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::clip::{ClipConfig, ClipNode, ClipNodeBuilder};
    use onnx_ir::node::clip::ClipInput;

    fn create_clip_node(name: &str, min: Option<f64>, max: Option<f64>) -> ClipNode {
        let config = ClipConfig {
            min: min.map(ClipInput::Static),
            max: max.map(ClipInput::Static),
        };

        ClipNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_clip_both_bounds() {
        let node = create_clip_node("clip1", Some(-1.0), Some(1.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = -1f64;
                let __clip_max = 1f64;
                input.clamp(__clip_min, __clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_min_only() {
        let node = create_clip_node("clip1", Some(0.0), None);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = 0f64;
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_max_only() {
        let node = create_clip_node("clip1", None, Some(10.0));
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = {
                let __clip_max = 10f64;
                input.clamp_max(__clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_min_scalar_tensor() {
        // ONNX gives us Clip(x, min_tensor) where min_tensor arrives here
        // as `ArgType::ScalarTensor`, which onnx-ir models as a rank-1
        // tensor with shape [1] (see ArgType::rank() for ScalarTensor).
        // Burn's clamp_min takes a native scalar, so we extract it via
        // .into_scalar().elem::<T>() and cast to f64.
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar_tensor("min_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, min_val: Tensor<B, 1>) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<f32>() as f64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_both_scalar_tensors() {
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "max_val".to_string(),
                2,
            ))),
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar_tensor("min_val", DType::F32)
            .input_scalar_tensor("max_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            min_val: Tensor<B, 1>,
            max_val: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<f32>() as f64);
                let __clip_max = (max_val.into_scalar().elem::<f32>() as f64);
                input.clamp(__clip_min, __clip_max)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_int_scalar_tensor() {
        // Clipping an Int tensor with a runtime int scalar bound must keep
        // the bound as i64, not narrow it through f64. For typical int
        // magnitudes above 2^53, an `as f64` cast would silently round and
        // produce wrong clip results. This test pins the i64 lowering.
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::I64)
            .input_scalar_tensor("min_val", DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2, Int>,
            min_val: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<i64>() as i64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_uint_scalar_tensor() {
        // Clipping a U64 tensor with a runtime u64 scalar bound must keep
        // the bound as u64. Routing a u64 bound through i64 would wrap any
        // value above i64::MAX into a large negative number and silently
        // produce the wrong clip result. This test pins the u64 lowering.
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::U64)
            .input_scalar_tensor("min_val", DType::U64)
            .output_tensor("output", 2, DType::U64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2, Int>,
            min_val: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let __clip_min = (min_val.into_scalar().elem::<u64>() as u64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }

    #[test]
    fn test_clip_runtime_min_scalar_native() {
        let config = ClipConfig {
            min: Some(ClipInput::Runtime(onnx_ir::ir::RuntimeInputRef::new(
                "min_val".to_string(),
                1,
            ))),
            max: None,
        };
        let node = ClipNodeBuilder::new("clip1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar("min_val", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, min_val: f32) -> Tensor<B, 2> {
            let output = {
                let __clip_min = (min_val as f64);
                input.clamp_min(__clip_min)
            };
            output
        }
        ");
    }
}
