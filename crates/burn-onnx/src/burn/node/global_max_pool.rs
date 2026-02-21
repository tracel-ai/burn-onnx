use super::prelude::*;

impl NodeCodegen for onnx_ir::node::global_max_pool::GlobalMaxPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());

        let input_arg = self.inputs.first().unwrap();
        let rank = match &input_arg.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for GlobalMaxPool"),
        };

        // GlobalMaxPool reduces over all spatial dimensions, keeping batch and channel dims
        // 3D: [B, C, H] -> reduce dims 1, 2 -> [B, C, 1]
        // 4D: [B, C, H, W] -> reduce dims 2, 3 -> [B, C, 1, 1]
        match rank {
            3 => {
                quote! {
                    let #output = #input.max_dim(2usize);
                }
            }
            4 => {
                quote! {
                    let #output = #input.max_dim(3usize).max_dim(2usize);
                }
            }
            dim => panic!("Unsupported input dim ({dim}) for GlobalMaxPoolNode"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_max_pool::{GlobalMaxPoolNode, GlobalMaxPoolNodeBuilder};

    fn create_global_max_pool_node_3d(name: &str) -> GlobalMaxPoolNode {
        GlobalMaxPoolNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build()
    }

    fn create_global_max_pool_node_4d(name: &str) -> GlobalMaxPoolNode {
        GlobalMaxPoolNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build()
    }

    #[test]
    fn test_global_max_pool_forward_3d() {
        let node = create_global_max_pool_node_3d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = input.max_dim(2usize);
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_4d() {
        let node = create_global_max_pool_node_4d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input.max_dim(3usize).max_dim(2usize);
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_with_clone_3d() {
        let node = create_global_max_pool_node_3d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = input.clone().max_dim(2usize);
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_with_clone_4d() {
        let node = create_global_max_pool_node_4d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = input.clone().max_dim(3usize).max_dim(2usize);
            output
        }
        ");
    }
}
