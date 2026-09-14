use super::prelude::*;
use onnx_ir::ir::ArgType;
use proc_macro2::Literal;

impl NodeCodegen for onnx_ir::node::range::RangeNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut super::super::scope::ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Each parameter is used more than once below, so a static literal or a
        // scalar tensor readback is bound to a local first. Native scalars are
        // plain `Copy` idents and are used directly. Returns the binding
        // statement (possibly empty) and the expression to use.
        let range_param_tokens = |config: &onnx_ir::node::range::RangeInput,
                                  inputs: &[Argument],
                                  scope: &mut super::super::scope::ScopeAtPosition<'_>,
                                  local: &str|
         -> (TokenStream, TokenStream) {
            let local = Ident::new(local, Span::call_site());
            match config {
                onnx_ir::node::range::RangeInput::Static(value) => {
                    let literal = Literal::i64_suffixed(*value);
                    (quote! { let #local = #literal; }, quote! { #local })
                }
                onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                    let arg = &inputs[runtime_ref.input_index];
                    match &arg.ty {
                        ArgType::ScalarNative(_) => {
                            let name = arg_to_ident(arg);
                            (quote! {}, quote! { #name })
                        }
                        ArgType::ScalarTensor(dtype) => {
                            let tensor = scope.arg(arg);
                            let native = on_device_to_native(quote! { #tensor }, dtype);
                            (quote! { let #local = #native; }, quote! { #local })
                        }
                        _ => panic!("Range parameter must be a scalar"),
                    }
                }
            }
        };

        let output_dtype = self.outputs.first().unwrap().ty.elem_type().to_tokens();

        // Use formula: output[i] = start + i * delta, for i in 0..n
        // where n = max(ceil((limit - start) / delta), 0)
        // This correctly handles both positive and negative delta.
        use onnx_ir::node::range::RangeInput;
        match (&self.config.start, &self.config.limit, &self.config.delta) {
            (RangeInput::Static(s), RangeInput::Static(l), RangeInput::Static(d)) => {
                // All static: precompute n at codegen time
                let n = ((*l - *s) as f64 / *d as f64).ceil().max(0.0) as i64;
                let n_lit = Literal::i64_suffixed(n);
                let d_lit = Literal::i64_suffixed(*d);
                let s_lit = Literal::i64_suffixed(*s);
                quote! {
                    let #output = Tensor::arange(0..#n_lit, &self.device)
                        .cast(#output_dtype)
                        .mul_scalar(#d_lit)
                        .add_scalar(#s_lit);
                }
            }
            _ => {
                // At least one runtime value: compute n at runtime
                let (bind_start, start) =
                    range_param_tokens(&self.config.start, &self.inputs, scope, "start");
                let (bind_limit, limit) =
                    range_param_tokens(&self.config.limit, &self.inputs, scope, "limit");
                let (bind_delta, delta) =
                    range_param_tokens(&self.config.delta, &self.inputs, scope, "delta");
                quote! {
                    let #output = {
                        #bind_start
                        #bind_limit
                        #bind_delta
                        assert!(#delta != 0);
                        let n = ((#limit - #start) as f64 / #delta as f64)
                            .ceil().max(0.0) as i64;
                        Tensor::arange(0..n, &self.device)
                            .cast(#output_dtype)
                            .mul_scalar(#delta)
                            .add_scalar(#start)
                    };
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
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::node::range::{RangeConfig, RangeInput, RangeNodeBuilder};

    #[test]
    fn test_range_static() {
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Static(10),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..5i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(2i64)
                .add_scalar(0i64);
            output
        }
        ");
    }

    #[test]
    fn test_range_negative_delta() {
        let config = RangeConfig::new(
            RangeInput::Static(10),
            RangeInput::Static(0),
            RangeInput::Static(-2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..5i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(-2i64)
                .add_scalar(10i64);
            output
        }
        ");
    }

    #[test]
    fn test_range_runtime() {
        let config = RangeConfig::new(
            RangeInput::Runtime(RuntimeInputRef {
                name: "start".to_string(),
                input_index: 0,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 1,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "delta".to_string(),
                input_index: 2,
            }),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar("start", DType::I64)
            .input_scalar("limit", DType::I64)
            .input_scalar("delta", DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, start: i64, limit: i64, delta: i64) -> Tensor<1, Int> {
            let output = {
                assert!(delta != 0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_mixed_static_and_tensor() {
        // A static literal and a scalar tensor readback are each bound once;
        // the native scalar is used directly.
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Runtime(RuntimeInputRef {
                name: "limit".to_string(),
                input_index: 0,
            }),
            RangeInput::Runtime(RuntimeInputRef {
                name: "delta".to_string(),
                input_index: 1,
            }),
        );
        let node = RangeNodeBuilder::new("range1")
            .input_scalar_tensor("limit", DType::I64)
            .input_scalar("delta", DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, limit: Tensor<1, Int>, delta: i64) -> Tensor<1, Int> {
            let output = {
                let start = 0i64;
                let limit = (limit).into_scalar::<i64>();
                assert!(delta != 0);
                let n = ((limit - start) as f64 / delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..n, &self.device)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(delta)
                    .add_scalar(start)
            };
            output
        }
        ");
    }

    #[test]
    fn test_range_empty() {
        // start >= limit with positive delta produces empty range
        let config = RangeConfig::new(
            RangeInput::Static(10),
            RangeInput::Static(0),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1, Int> {
            let output = Tensor::arange(0..0i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(2i64)
                .add_scalar(10i64);
            output
        }
        ");
    }
}
