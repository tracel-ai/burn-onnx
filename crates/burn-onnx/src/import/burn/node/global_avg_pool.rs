use super::prelude::*;

impl NodeCodegen for onnx_ir::node::global_avg_pool::GlobalAveragePoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Ranks 3 and 4 use burn's adaptive pool modules; every other rank is handled
        // in `forward` without a field.
        let name = Ident::new(&self.name, Span::call_site());

        let (field_type, init_tokens) = match self.inputs.first().unwrap().ty.rank() {
            3 => (
                quote! { AdaptiveAvgPool1d },
                quote! {
                    let #name = AdaptiveAvgPool1dConfig::new(1)
                        .init();
                },
            ),
            4 => (
                quote! { AdaptiveAvgPool2d },
                quote! {
                    let #name = AdaptiveAvgPool2dConfig::new([1, 1])
                        .init();
                },
            ),
            _ => return None,
        };

        Some(Field::new(self.name.clone(), field_type, init_tokens))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input_arg = self.inputs.first().unwrap();
        let input = scope.arg(input_arg);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let rank = input_arg.ty.rank();

        match rank {
            3 | 4 => {
                let field = Ident::new(&self.name, Span::call_site());
                quote! {
                    let #output = self.#field.forward(#input);
                }
            }
            // burn has no N-d adaptive pool. `mean_dim` keeps each reduced axis as
            // size 1, which gives the [N, C, 1, 1, ...] output the spec requires.
            rank if rank >= 5 => {
                let dims = (2..rank).collect::<Vec<usize>>().to_tokens();
                quote! {
                    let #output = #input.mean_dims(&#dims);
                }
            }
            // onnx-ir rejects rank <= 2, so this is reachable only from a hand-built
            // node. A named `compile_error!` keeps the failure inside the generated
            // crate instead of crashing model generation.
            rank => {
                let msg = format!(
                    "GlobalAveragePool node '{}': requires rank >= 3, got rank {rank}",
                    self.name
                );
                quote! { let #output = { compile_error!(#msg); unreachable!() }; }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match self.inputs.first().unwrap().ty.rank() {
            3 => {
                imports.register("burn::nn::pool::AdaptiveAvgPool1d");
                imports.register("burn::nn::pool::AdaptiveAvgPool1dConfig");
            }
            4 => {
                imports.register("burn::nn::pool::AdaptiveAvgPool2d");
                imports.register("burn::nn::pool::AdaptiveAvgPool2dConfig");
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_avg_pool::{GlobalAveragePoolNode, GlobalAveragePoolNodeBuilder};

    fn create_global_avg_pool_node_3d(name: &str) -> GlobalAveragePoolNode {
        GlobalAveragePoolNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build()
    }

    fn create_global_avg_pool_node_4d(name: &str) -> GlobalAveragePoolNode {
        GlobalAveragePoolNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build()
    }

    #[test]
    fn test_global_avg_pool_forward_3d() {
        let node = create_global_avg_pool_node_3d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_global_avg_pool_forward_4d() {
        let node = create_global_avg_pool_node_4d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_global_avg_pool_forward_with_clone_3d() {
        let node = create_global_avg_pool_node_3d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_global_avg_pool_forward_with_clone_4d() {
        let node = create_global_avg_pool_node_4d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_global_avg_pool_forward_5d() {
        let node = GlobalAveragePoolNodeBuilder::new("pool1")
            .input_tensor("input", 5, DType::F32)
            .output_tensor("output", 5, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<5>) -> Tensor<5> {
            let output = input.mean_dims(&[2, 3, 4]);
            output
        }
        ");
    }

    #[test]
    fn test_global_avg_pool_forward_rank_2_compile_error() {
        let node = GlobalAveragePoolNodeBuilder::new("pool1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 2, DType::F32)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            let output = {
                compile_error!("GlobalAveragePool node 'pool1': requires rank >= 3, got rank 2");
                unreachable!()
            };
            output
        }
        "#);
    }
}
