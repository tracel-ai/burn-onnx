use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::node::batch_norm::{BatchNormConfig, BatchNormalizationNode};

impl NodeCodegen for BatchNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        match &self.config {
            BatchNormConfig::Static(config) => {
                let name = Ident::new(&self.name, Span::call_site());
                let scale_shape = self.inputs[1]
                    .ty
                    .static_shape_known()
                    .expect("BatchNorm: scale tensor shape must be known at codegen time");
                let num_features = scale_shape[0].to_tokens();
                let epsilon = config.epsilon;
                let momentum = config.momentum;

                Some(Field::new(
                    self.name.clone(),
                    quote! {
                        BatchNorm<B>
                    },
                    quote! {
                        let #name = BatchNormConfig::new(#num_features)
                            .with_epsilon(#epsilon)
                            .with_momentum(#momentum)
                            .init(device);
                    },
                ))
            }
            BatchNormConfig::Runtime(_) => None,
        }
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());

        match &self.config {
            BatchNormConfig::Static(_) => {
                let input = scope.arg(self.inputs.first().unwrap());
                let field = Ident::new(&self.name, Span::call_site());

                quote! {
                    let #output = self.#field.forward(#input);
                }
            }
            BatchNormConfig::Runtime(config) => {
                let input = scope.arg(&self.inputs[0]);
                let scale = scope.arg(&self.inputs[1]);
                let bias = scope.arg(&self.inputs[2]);
                let mean = scope.arg(&self.inputs[3]);
                let var = scope.arg(&self.inputs[4]);
                let epsilon = config.epsilon;

                // Determine the input rank from the type info
                let rank = match &self.inputs[0].ty {
                    ArgType::Tensor(t) => t.rank,
                    _ => panic!("BatchNorm input must be a tensor"),
                };

                // Build the reshape dimensions: [1, C, 1, 1, ...] for broadcasting
                // scale/bias/mean/var are 1D tensors of shape [C], need to broadcast
                // to match input shape [N, C, D1, D2, ...]
                let unsqueeze_dims: Vec<isize> = {
                    // For rank-D input, unsqueeze dim 0 and dims 2..rank-1
                    let mut dims = vec![0isize]; // prepend batch dim
                    for i in 2..rank {
                        dims.push(i as isize);
                    }
                    dims
                };

                quote! {
                    let #output = {
                        let scale = #scale.unsqueeze_dims(&[#(#unsqueeze_dims),*]);
                        let bias = #bias.unsqueeze_dims(&[#(#unsqueeze_dims),*]);
                        let mean = #mean.unsqueeze_dims(&[#(#unsqueeze_dims),*]);
                        let var = #var.unsqueeze_dims(&[#(#unsqueeze_dims),*]);
                        (#input - mean) / (var + #epsilon).sqrt() * scale + bias
                    };
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match &self.config {
            BatchNormConfig::Static(_) => {
                imports.register("burn::nn::BatchNorm");
                imports.register("burn::nn::BatchNormConfig");
            }
            BatchNormConfig::Runtime(_) => {
                // No module imports needed for inline math
            }
        }
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        match &self.config {
            BatchNormConfig::Static(_) => {
                use crate::burn::node_traits::create_lazy_snapshot;
                let mut snapshots = vec![];

                if let Some(gamma_input) = self.inputs.get(1) {
                    let gamma_path = format!("{}.gamma", field_name);
                    if let Some(snapshot) =
                        create_lazy_snapshot(gamma_input, &gamma_path, "BatchNorm")
                    {
                        snapshots.push(snapshot);
                    }
                }

                if let Some(beta_input) = self.inputs.get(2) {
                    let beta_path = format!("{}.beta", field_name);
                    if let Some(snapshot) =
                        create_lazy_snapshot(beta_input, &beta_path, "BatchNorm")
                    {
                        snapshots.push(snapshot);
                    }
                }

                if let Some(running_mean_input) = self.inputs.get(3) {
                    let running_mean_path = format!("{}.running_mean", field_name);
                    if let Some(snapshot) =
                        create_lazy_snapshot(running_mean_input, &running_mean_path, "BatchNorm")
                    {
                        snapshots.push(snapshot);
                    }
                }

                if let Some(running_var_input) = self.inputs.get(4) {
                    let running_var_path = format!("{}.running_var", field_name);
                    if let Some(snapshot) =
                        create_lazy_snapshot(running_var_input, &running_var_path, "BatchNorm")
                    {
                        snapshots.push(snapshot);
                    }
                }

                snapshots
            }
            BatchNormConfig::Runtime(_) => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::batch_norm::{
        BatchNormConfig, BatchNormRuntimeConfig, BatchNormStaticConfig, BatchNormalizationNode,
        BatchNormalizationNodeBuilder,
    };

    fn create_batch_norm_node(name: &str) -> BatchNormalizationNode {
        let config = BatchNormConfig::Static(BatchNormStaticConfig::new(1e-5, 0.9));

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("scale", vec![64], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .input_static_tensor_shape("mean", vec![64], DType::F32)
            .input_static_tensor_shape("var", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_runtime_batch_norm_node(name: &str, input_rank: usize) -> BatchNormalizationNode {
        let config = BatchNormConfig::Runtime(BatchNormRuntimeConfig::new(1e-5, 0.9));

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", input_rank, DType::F32)
            .input_tensor("scale", 1, DType::F32)
            .input_tensor("bias", 1, DType::F32)
            .input_tensor("mean", 1, DType::F32)
            .input_tensor("var", 1, DType::F32)
            .output_tensor("output", input_rank, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_batch_norm_forward() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_forward_with_clone() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank3() {
        let node = create_runtime_batch_norm_node("batch_norm1", 3);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
            mean: Tensor<B, 1>,
            var: Tensor<B, 1>,
        ) -> Tensor<B, 3> {
            let output = {
                let scale = scale.unsqueeze_dims(&[0isize, 2isize]);
                let bias = bias.unsqueeze_dims(&[0isize, 2isize]);
                let mean = mean.unsqueeze_dims(&[0isize, 2isize]);
                let var = var.unsqueeze_dims(&[0isize, 2isize]);
                (input - mean) / (var + 0.00001f64).sqrt() * scale + bias
            };
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank4() {
        let node = create_runtime_batch_norm_node("batch_norm1", 4);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
            mean: Tensor<B, 1>,
            var: Tensor<B, 1>,
        ) -> Tensor<B, 4> {
            let output = {
                let scale = scale.unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let bias = bias.unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let mean = mean.unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let var = var.unsqueeze_dims(&[0isize, 2isize, 3isize]);
                (input - mean) / (var + 0.00001f64).sqrt() * scale + bias
            };
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_rank5() {
        let node = create_runtime_batch_norm_node("batch_norm1", 5);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 5>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
            mean: Tensor<B, 1>,
            var: Tensor<B, 1>,
        ) -> Tensor<B, 5> {
            let output = {
                let scale = scale.unsqueeze_dims(&[0isize, 2isize, 3isize, 4isize]);
                let bias = bias.unsqueeze_dims(&[0isize, 2isize, 3isize, 4isize]);
                let mean = mean.unsqueeze_dims(&[0isize, 2isize, 3isize, 4isize]);
                let var = var.unsqueeze_dims(&[0isize, 2isize, 3isize, 4isize]);
                (input - mean) / (var + 0.00001f64).sqrt() * scale + bias
            };
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_runtime_forward_with_clone() {
        let node = create_runtime_batch_norm_node("batch_norm1", 4);
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
            mean: Tensor<B, 1>,
            var: Tensor<B, 1>,
        ) -> Tensor<B, 4> {
            let output = {
                let scale = scale.clone().unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let bias = bias.clone().unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let mean = mean.clone().unsqueeze_dims(&[0isize, 2isize, 3isize]);
                let var = var.clone().unsqueeze_dims(&[0isize, 2isize, 3isize]);
                (input.clone() - mean) / (var + 0.00001f64).sqrt() * scale + bias
            };
            output
        }
        ");
    }
}
