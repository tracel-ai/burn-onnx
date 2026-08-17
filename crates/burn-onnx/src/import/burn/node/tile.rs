use super::prelude::*;

impl NodeCodegen for onnx_ir::tile::TileNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        match &self.config.repeats {
            onnx_ir::tile::TileInput::Static(repeats_vec) => {
                let repeats = repeats_vec.iter().map(|r| r.to_tokens());
                quote! {
                    let #output = #input.repeat(&[#(#repeats),*]);
                }
            }
            onnx_ir::tile::TileInput::Runtime(runtime_ref) => {
                // Runtime repeats: the value is known only at forward()
                // time. Shape inputs are native `[i64; N]`; tensor
                // inputs need a to_data() round-trip. Convert to
                // Vec<usize> and call .repeat() with a slice.
                let repeats_arg = &self.inputs[runtime_ref.input_index];
                let repeats_expr = scope.arg(repeats_arg);
                let to_vec_usize = match &repeats_arg.ty {
                    ArgType::Shape(_) => quote! {
                        let __repeats: alloc::vec::Vec<usize> =
                            #repeats_expr.iter().map(|&v| v as usize).collect();
                    },
                    _ => quote! {
                        let __repeats: alloc::vec::Vec<usize> =
                            #repeats_expr
                                .to_data()
                                .convert::<i64>()
                                .into_vec::<i64>()
                                .unwrap()
                                .into_iter()
                                .map(|v| v as usize)
                                .collect();
                    },
                };
                quote! {
                    let #output = {
                        #to_vec_usize
                        #input.repeat(&__repeats)
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
    use onnx_ir::tile::{TileConfig, TileInput, TileNode, TileNodeBuilder};

    fn create_tile_node(name: &str, repeats: Vec<usize>) -> TileNode {
        let config = TileConfig {
            repeats: TileInput::Static(repeats),
        };

        TileNodeBuilder::new(name)
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_tile_simple() {
        let node = create_tile_node("tile1", vec![2, 3]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.repeat(&[2, 3]);
            output
        }
        ");
    }

    #[test]
    fn test_tile_single_repeat() {
        let node = create_tile_node("tile1", vec![1, 2, 3]);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = input.repeat(&[1, 2, 3]);
            output
        }
        ");
    }

    #[test]
    fn test_tile_runtime_tensor() {
        let config = TileConfig {
            repeats: TileInput::Runtime(RuntimeInputRef::new("repeats".to_string(), 1)),
        };
        let node = TileNodeBuilder::new("tile_rt")
            .input_tensor("input", 2, DType::F32)
            .input_tensor("repeats", 1, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, repeats: Tensor<1, Int>) -> Tensor<2> {
            let output = {
                let __repeats: alloc::vec::Vec<usize> = repeats
                    .to_data()
                    .convert::<i64>()
                    .into_vec::<i64>()
                    .unwrap()
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();
                input.repeat(&__repeats)
            };
            output
        }
        ");
    }

    #[test]
    fn test_tile_runtime_shape() {
        let config = TileConfig {
            repeats: TileInput::Runtime(RuntimeInputRef::new("repeats".to_string(), 1)),
        };
        let node = TileNodeBuilder::new("tile_rt_shape")
            .input_tensor("input", 2, DType::F32)
            .input_shape("repeats", 2)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<2>, repeats: [i64; 2]) -> Tensor<2> {
            let output = {
                let __repeats: alloc::vec::Vec<usize> = repeats
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                input.repeat(&__repeats)
            };
            output
        }
        ");
    }
}
