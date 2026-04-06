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

        // Generate values for start, limit, and delta based on Static or Runtime
        let range_param_tokens = |config: &onnx_ir::node::range::RangeInput,
                                  inputs: &[Argument],
                                  scope: &mut super::super::scope::ScopeAtPosition<'_>|
         -> TokenStream {
            match config {
                onnx_ir::node::range::RangeInput::Static(value) => {
                    let literal = Literal::i64_suffixed(*value);
                    quote! { #literal }
                }
                onnx_ir::node::range::RangeInput::Runtime(runtime_ref) => {
                    let arg = &inputs[runtime_ref.input_index];
                    match &arg.ty {
                        ArgType::ScalarNative(_) => {
                            let name = arg_to_ident(arg);
                            quote! { #name }
                        }
                        ArgType::ScalarTensor(dtype) => {
                            let tensor = scope.arg(arg);
                            on_device_to_native(quote! { #tensor }, dtype)
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
                let start = range_param_tokens(&self.config.start, &self.inputs, scope);
                let limit = range_param_tokens(&self.config.limit, &self.inputs, scope);
                let delta = range_param_tokens(&self.config.delta, &self.inputs, scope);
                quote! {
                    let #output = {
                        let __start = #start;
                        let __limit = #limit;
                        let __delta = #delta;
                        assert!(__delta != 0);
                        let __n = ((__limit - __start) as f64 / __delta as f64)
                            .ceil().max(0.0) as i64;
                        Tensor::arange(0..__n, &self.device)
                            .cast(#output_dtype)
                            .mul_scalar(__delta)
                            .add_scalar(__start)
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
        pub fn forward(&self) -> Tensor<B, 1, Int> {
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
        pub fn forward(&self) -> Tensor<B, 1, Int> {
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
        pub fn forward(&self, start: i64, limit: i64, delta: i64) -> Tensor<B, 1, Int> {
            let output = {
                let __start = start;
                let __limit = limit;
                let __delta = delta;
                assert!(__delta != 0);
                let __n = ((__limit - __start) as f64 / __delta as f64).ceil().max(0.0) as i64;
                Tensor::arange(0..__n, &self.device)
                    .cast(burn::tensor::DType::I64)
                    .mul_scalar(__delta)
                    .add_scalar(__start)
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
        pub fn forward(&self) -> Tensor<B, 1, Int> {
            let output = Tensor::arange(0..0i64, &self.device)
                .cast(burn::tensor::DType::I64)
                .mul_scalar(2i64)
                .add_scalar(10i64);
            output
        }
        ");
    }
}
