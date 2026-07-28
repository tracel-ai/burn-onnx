use super::prelude::*;
use onnx_ir::ir::ArgType;
use onnx_ir::node::range::RangeInput;
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
        let output_elem_type = self.outputs.first().unwrap().ty.elem_type();
        let output_dtype = output_elem_type.to_tokens();

        // Use formula: output[i] = start + i * delta, for i in 0..n
        // where n = max(ceil((limit - start) / delta), 0)
        // This correctly handles both positive and negative delta.
        match (&self.config.start, &self.config.limit, &self.config.delta) {
            (RangeInput::Static(s), RangeInput::Static(l), RangeInput::Static(d)) => {
                // All static: precompute n at codegen time
                let n = ((*l - *s) as f64 / *d as f64).ceil().max(0.0) as i64;
                let n_lit = Literal::i64_suffixed(n);
                let d_lit = Literal::i64_suffixed(*d);
                let s_lit = Literal::i64_suffixed(*s);
                let arange = range_arange_tokens(
                    output_elem_type,
                    output_dtype,
                    quote! { #n_lit },
                    quote! { #s_lit },
                    quote! { #d_lit },
                );
                quote! {
                    let #output = #arange;
                }
            }
            _ => {
                // At least one runtime value: compute n at runtime
                let start = range_param_tokens(&self.config.start, &self.inputs, scope);
                let limit = range_param_tokens(&self.config.limit, &self.inputs, scope);
                let delta = range_param_tokens(&self.config.delta, &self.inputs, scope);
                let n = range_len_tokens(output_elem_type);
                let arange = range_arange_tokens(
                    output_elem_type,
                    output_dtype,
                    quote! { __n },
                    quote! { __start },
                    quote! { __delta },
                );
                quote! {
                    let #output = {
                        let __start = #start;
                        let __limit = #limit;
                        let __delta = #delta;
                        assert!(__delta != 0);
                        let __n = #n;
                        #arange
                    };
                }
            }
        }
    }
}

fn range_param_tokens(
    config: &RangeInput,
    inputs: &[Argument],
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    match config {
        RangeInput::Static(value) => {
            let literal = Literal::i64_suffixed(*value);
            quote! { #literal }
        }
        RangeInput::Runtime(runtime_ref) => {
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
}

fn range_len_tokens(output_elem_type: DType) -> TokenStream {
    if output_elem_type.is_float() {
        quote! {
            ((__limit as f64 - __start as f64) / __delta as f64).ceil().max(0.0) as i64
        }
    } else {
        quote! {
            ((__limit - __start) as f64 / __delta as f64).ceil().max(0.0) as i64
        }
    }
}

fn range_arange_tokens(
    output_elem_type: DType,
    output_dtype: TokenStream,
    n: TokenStream,
    start: TokenStream,
    delta: TokenStream,
) -> TokenStream {
    if output_elem_type.is_float() {
        quote! {
            Tensor::<1, Int>::arange(0..#n, &self.device)
                .float()
                .cast(#output_dtype)
                .mul_scalar(#delta as f64)
                .add_scalar(#start as f64)
        }
    } else {
        quote! {
            Tensor::arange(0..#n, &self.device)
                .cast(#output_dtype)
                .mul_scalar(#delta)
                .add_scalar(#start)
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
    fn test_range_static_float_output() {
        let config = RangeConfig::new(
            RangeInput::Static(0),
            RangeInput::Static(10),
            RangeInput::Static(2),
        );
        let node = RangeNodeBuilder::new("range1")
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1> {
            let output = Tensor::<1, Int>::arange(0..5i64, &self.device)
                .float()
                .cast(burn::tensor::DType::F32)
                .mul_scalar(2i64 as f64)
                .add_scalar(0i64 as f64);
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
    fn test_range_runtime_float_output() {
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
            .input_scalar("limit", DType::F32)
            .input_scalar("delta", DType::I64)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, start: i64, limit: f32, delta: i64) -> Tensor<1> {
            let output = {
                let __start = start;
                let __limit = limit;
                let __delta = delta;
                assert!(__delta != 0);
                let __n = ((__limit as f64 - __start as f64) / __delta as f64).ceil().max(0.0)
                    as i64;
                Tensor::<1, Int>::arange(0..__n, &self.device)
                    .float()
                    .cast(burn::tensor::DType::F32)
                    .mul_scalar(__delta as f64)
                    .add_scalar(__start as f64)
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
