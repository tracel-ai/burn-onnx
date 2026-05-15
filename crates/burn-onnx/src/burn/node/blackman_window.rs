use super::prelude::*;
use onnx_ir::node::blackman_window::WindowSize;

impl NodeCodegen for onnx_ir::node::blackman_window::BlackmanWindowNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, _scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let periodic = self.config.periodic;
        let output_dtype = self.config.output_dtype.to_tokens();

        let size_expr = match &self.config.size {
            WindowSize::Static(size) => {
                let size = *size;
                quote! { #size }
            }
            WindowSize::Runtime(runtime_ref) => {
                let arg = &self.inputs[runtime_ref.input_index];
                let name = arg_to_ident(arg);
                quote! { {
                    let __size = #name;
                    usize::try_from(__size).unwrap_or_else(|_| panic!(
                        "BlackmanWindow: size must be non-negative and fit in usize, got {}",
                        __size
                    ))
                } }
            }
        };

        quote! {
            let #output = blackman_window(#size_expr, #periodic, &self.device)
                .cast(#output_dtype);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::tensor::signal::blackman_window");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::blackman_window::{
        BlackmanWindowConfig, BlackmanWindowNodeBuilder, WindowSize,
    };

    #[test]
    fn test_blackman_window_static() {
        let config = BlackmanWindowConfig {
            periodic: true,
            output_dtype: DType::F32,
            size: WindowSize::Static(10),
        };
        let node = BlackmanWindowNodeBuilder::new("blackman1")
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1> {
            let output = blackman_window(10usize, true, &self.device)
                .cast(burn::tensor::DType::F32);
            output
        }
        ");
    }

    #[test]
    fn test_blackman_window_symmetric() {
        let config = BlackmanWindowConfig {
            periodic: false,
            output_dtype: DType::F64,
            size: WindowSize::Static(8),
        };
        let node = BlackmanWindowNodeBuilder::new("blackman1")
            .output_tensor("output", 1, DType::F64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self) -> Tensor<1> {
            let output = blackman_window(8usize, false, &self.device)
                .cast(burn::tensor::DType::F64);
            output
        }
        ");
    }

    #[test]
    fn test_blackman_window_runtime() {
        use onnx_ir::ir::RuntimeInputRef;
        let config = BlackmanWindowConfig {
            periodic: true,
            output_dtype: DType::F32,
            size: WindowSize::Runtime(RuntimeInputRef {
                name: "size".to_string(),
                input_index: 0,
            }),
        };
        let node = BlackmanWindowNodeBuilder::new("blackman1")
            .input_scalar("size", DType::I64)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, size: i64) -> Tensor<1> {
            let output = blackman_window(
                    {
                        let __size = size;
                        usize::try_from(__size)
                            .unwrap_or_else(|_| {
                                panic!(
                                    "BlackmanWindow: size must be non-negative and fit in usize, got {}",
                                    __size
                                )
                            })
                    },
                    true,
                    &self.device,
                )
                .cast(burn::tensor::DType::F32);
            output
        }
        "#);
    }
}
