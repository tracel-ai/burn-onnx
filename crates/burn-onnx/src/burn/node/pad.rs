use super::prelude::*;
use onnx_ir::ir::ArgType;
use std::str::FromStr;

impl NodeCodegen for onnx_ir::pad::PadNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        // Extract static pads from the enum wrapper
        let pads_vec = match &self.config.pads {
            onnx_ir::pad::PadInput::Static(pads) => pads,
            onnx_ir::pad::PadInput::Runtime(_) => {
                panic!("Runtime pads are not supported in burn-onnx")
            }
        };
        let pads: Vec<TokenStream> = pads_vec
            .iter()
            .map(|(before, after)| quote! { (#before, #after) })
            .collect();

        // Generate PadMode based on the mode in config (using fully qualified path)
        let pad_mode = match &self.config.mode {
            onnx_ir::pad::PadMode::Constant => {
                let constant_value = match &self.config.constant_value {
                    onnx_ir::pad::ConstantValueInput::Static(value) => {
                        let literal = TokenStream::from_str(&format!("{}_f32", value)).unwrap();
                        quote! { #literal }
                    }
                    onnx_ir::pad::ConstantValueInput::Runtime(runtime_ref) => {
                        let arg = &self.inputs[runtime_ref.input_index];
                        let value = scope.arg(arg);
                        match &arg.ty {
                            ArgType::Tensor(t) if t.rank == 0 => {
                                quote! { #value.into_scalar() }
                            }
                            ArgType::Tensor(t) => {
                                panic!(
                                    "Pad: constant_value must be a scalar tensor (rank 0), got rank {}",
                                    t.rank
                                )
                            }
                            ArgType::ScalarNative(_) => {
                                quote! { #value }
                            }
                            ArgType::ScalarTensor(dtype) => {
                                on_device_to_native(quote! { #value }, dtype)
                            }
                            ArgType::Shape(_) => {
                                panic!("Pad: constant_value cannot be a shape")
                            }
                        }
                    }
                };
                quote! { burn::tensor::ops::PadMode::Constant(#constant_value) }
            }
            onnx_ir::pad::PadMode::Reflect => {
                quote! { burn::tensor::ops::PadMode::Reflect }
            }
            onnx_ir::pad::PadMode::Edge => {
                quote! { burn::tensor::ops::PadMode::Edge }
            }
        };

        quote! {
            let #output = #input.pad([#(#pads),*], #pad_mode);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::pad::{ConstantValueInput, PadConfig, PadInput, PadMode, PadNode, PadNodeBuilder};

    fn create_pad_node(
        name: &str,
        pads: Vec<(usize, usize)>,
        constant_value: f32,
        mode: PadMode,
    ) -> PadNode {
        let config = PadConfig {
            pads: PadInput::Static(pads),
            constant_value: ConstantValueInput::Static(constant_value),
            mode,
        };

        PadNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_pad_constant_simple() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(0_f32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_asymmetric() {
        let node = create_pad_node("pad1", vec![(0, 1), (2, 0)], 5.5, PadMode::Constant);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input
                .pad(
                    [(0usize, 1usize), (2usize, 0usize)],
                    burn::tensor::ops::PadMode::Constant(5.5_f32),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_reflect() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Reflect);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input
                .pad([(1usize, 1usize), (1usize, 1usize)], burn::tensor::ops::PadMode::Reflect);
            output
        }
        ");
    }

    #[test]
    fn test_pad_edge() {
        let node = create_pad_node("pad1", vec![(1, 1), (1, 1)], 0.0, PadMode::Edge);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = input
                .pad([(1usize, 1usize), (1usize, 1usize)], burn::tensor::ops::PadMode::Edge);
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_runtime_value() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 1), (1, 1)]),
            constant_value: ConstantValueInput::Runtime(RuntimeInputRef {
                name: "constant_value".to_string(),
                input_index: 1,
            }),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("constant_value", 0, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            constant_value: Tensor<B, 0>,
        ) -> Tensor<B, 2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(constant_value.into_scalar()),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_constant_runtime_scalar() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 1), (1, 1)]),
            constant_value: ConstantValueInput::Runtime(RuntimeInputRef {
                name: "constant_value".to_string(),
                input_index: 1,
            }),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 2, DType::F32)
            .input_scalar("constant_value", DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, constant_value: f32) -> Tensor<B, 2> {
            let output = input
                .pad(
                    [(1usize, 1usize), (1usize, 1usize)],
                    burn::tensor::ops::PadMode::Constant(constant_value),
                );
            output
        }
        ");
    }

    #[test]
    fn test_pad_4d_all_dimensions() {
        let config = PadConfig {
            pads: PadInput::Static(vec![(1, 2), (0, 0), (3, 4), (5, 6)]),
            constant_value: ConstantValueInput::Static(0.0),
            mode: PadMode::Constant,
        };
        let node = PadNodeBuilder::new("pad1")
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input
                .pad(
                    [(1usize, 2usize), (0usize, 0usize), (3usize, 4usize), (5usize, 6usize)],
                    burn::tensor::ops::PadMode::Constant(0_f32),
                );
            output
        }
        ");
    }
}
