use super::broadcast_helpers::channel_broadcast_shape;
use super::prelude::*;
use burn_store::TensorSnapshot;
use onnx_ir::node::group_norm::GroupNormalizationNode;

/// True when both scale (input[1]) and bias (input[2]) were lifted to static
/// initializers. When either is dynamic, the Burn `GroupNorm` module field
/// can't be populated from the burnpack, so we inline the formula instead.
fn weights_are_static(node: &GroupNormalizationNode) -> bool {
    node.inputs.get(1).is_some_and(|s| s.is_static())
        && node.inputs.get(2).is_some_and(|b| b.is_static())
}

impl NodeCodegen for GroupNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !weights_are_static(self) {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let num_groups = self.config.num_groups.to_tokens();
        let scale_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("GroupNorm: scale tensor shape must be known at codegen time");
        let num_features = scale_shape[0].to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! { GroupNorm<B> },
            quote! {
                let #name = GroupNormConfig::new(#num_groups, #num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        use crate::burn::node_traits::create_lazy_snapshot;

        let mut snapshots = vec![];

        if let Some(gamma_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(snapshot) = create_lazy_snapshot(gamma_input, &gamma_path, "GroupNorm") {
                snapshots.push(snapshot);
            }
        }

        if let Some(beta_input) = self.inputs.get(2) {
            let beta_path = format!("{}.beta", field_name);
            if let Some(snapshot) = create_lazy_snapshot(beta_input, &beta_path, "GroupNorm") {
                snapshots.push(snapshot);
            }
        }

        snapshots
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let x_arg = self.inputs.first().unwrap();
        let output = arg_to_ident(self.outputs.first().unwrap());
        let input = scope.arg(x_arg);

        if weights_are_static(self) {
            let field = Ident::new(&self.name, Span::call_site());
            return if self.config.full_precision {
                quote! {
                    let #output = {
                        let dtype = #input.dtype();
                        self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                    };
                }
            } else {
                quote! {
                    let #output = self.#field.forward(#input);
                }
            };
        }

        // Reshape to [N, num_groups, hidden] before reducing so a single
        // `sum_dim(2)` averages across the per-group spatial+channel slice
        // (mirrors Burn's internal `group_norm`). The channel dim for the
        // affine reshape is read from `__dims` at runtime so we don't need
        // scale's static shape to be known at codegen time.
        let scale = scope.arg(&self.inputs[1]);
        let bias = scope.arg(&self.inputs[2]);
        let rank = x_arg.ty.rank();
        let epsilon = self.config.epsilon;
        let num_groups = self.config.num_groups.to_tokens();
        let affine_shape = channel_broadcast_shape(rank, quote! { __dims[1] });

        // Body uses `__x`, `__scale`, `__bias` so the wrapper can decide whether
        // to bind them directly or via an F32 cast for the full_precision case.
        let body = quote! {
            let __dims = __x.dims();
            let __batch = __dims[0];
            let __num_groups: usize = #num_groups;
            let __spatial: usize = __dims[2..].iter().product();
            let __hidden: usize = __dims[1] * __spatial / __num_groups;
            let __hidden_f = __hidden as f64;
            let __x3 = __x.reshape([__batch, __num_groups, __hidden]);
            let __mean = __x3.clone().sum_dim(2).div_scalar(__hidden_f);
            let __centered = __x3.sub(__mean);
            let __var = __centered.clone().square().sum_dim(2).div_scalar(__hidden_f);
            let __normalized = __centered.div(__var.add_scalar(#epsilon).sqrt());
            __normalized
                .reshape(__dims)
                .mul(__scale.reshape(#affine_shape))
                .add(__bias.reshape(#affine_shape))
        };

        if self.config.full_precision {
            // Cast scale and bias to F32 alongside x so the affine multiply
            // doesn't dtype-mismatch when the runtime inputs are bf16/f16.
            quote! {
                let #output = {
                    let __orig_dtype = #input.dtype();
                    let __x = #input.cast(burn::tensor::DType::F32);
                    let __scale = #scale.cast(burn::tensor::DType::F32);
                    let __bias = #bias.cast(burn::tensor::DType::F32);
                    let __result = { #body };
                    __result.cast(__orig_dtype)
                };
            }
        } else {
            quote! {
                let #output = {
                    let __x = #input;
                    let __scale = #scale;
                    let __bias = #bias;
                    #body
                };
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if weights_are_static(self) {
            imports.register("burn::nn::GroupNorm");
            imports.register("burn::nn::GroupNormConfig");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::group_norm::{
        GroupNormConfig, GroupNormalizationNode, GroupNormalizationNodeBuilder,
    };

    fn create_static_group_norm_node(name: &str) -> GroupNormalizationNode {
        let config = GroupNormConfig::new(8, 1e-5, true);
        GroupNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_static_tensor_shape("scale", vec![64], DType::F32)
            .input_static_tensor_shape("bias", vec![64], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    fn create_dynamic_group_norm_node(name: &str, full_precision: bool) -> GroupNormalizationNode {
        let config = GroupNormConfig::new(2, 1e-5, full_precision);
        GroupNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .input_tensor_shape("scale", vec![4], DType::F32)
            .input_tensor_shape("bias", vec![4], DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_group_norm_forward_static() {
        let node = create_static_group_norm_node("group_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let dtype = input.dtype();
                self.group_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_group_norm_forward_static_with_clone() {
        let node = create_static_group_norm_node("group_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let dtype = input.clone().dtype();
                self.group_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_group_norm_forward_dynamic() {
        let node = create_dynamic_group_norm_node("group_norm1", false);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
        ) -> Tensor<B, 4> {
            let output = {
                let __x = input;
                let __scale = scale;
                let __bias = bias;
                let __dims = __x.dims();
                let __batch = __dims[0];
                let __num_groups: usize = 2;
                let __spatial: usize = __dims[2..].iter().product();
                let __hidden: usize = __dims[1] * __spatial / __num_groups;
                let __hidden_f = __hidden as f64;
                let __x3 = __x.reshape([__batch, __num_groups, __hidden]);
                let __mean = __x3.clone().sum_dim(2).div_scalar(__hidden_f);
                let __centered = __x3.sub(__mean);
                let __var = __centered.clone().square().sum_dim(2).div_scalar(__hidden_f);
                let __normalized = __centered.div(__var.add_scalar(0.00001f64).sqrt());
                __normalized
                    .reshape(__dims)
                    .mul(__scale.reshape([1usize, __dims[1], 1usize, 1usize]))
                    .add(__bias.reshape([1usize, __dims[1], 1usize, 1usize]))
            };
            output
        }
        ");
    }

    #[test]
    fn test_group_norm_forward_dynamic_full_precision() {
        let node = create_dynamic_group_norm_node("group_norm1", true);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 4>,
            scale: Tensor<B, 1>,
            bias: Tensor<B, 1>,
        ) -> Tensor<B, 4> {
            let output = {
                let __orig_dtype = input.dtype();
                let __x = input.cast(burn::tensor::DType::F32);
                let __scale = scale.cast(burn::tensor::DType::F32);
                let __bias = bias.cast(burn::tensor::DType::F32);
                let __result = {
                    let __dims = __x.dims();
                    let __batch = __dims[0];
                    let __num_groups: usize = 2;
                    let __spatial: usize = __dims[2..].iter().product();
                    let __hidden: usize = __dims[1] * __spatial / __num_groups;
                    let __hidden_f = __hidden as f64;
                    let __x3 = __x.reshape([__batch, __num_groups, __hidden]);
                    let __mean = __x3.clone().sum_dim(2).div_scalar(__hidden_f);
                    let __centered = __x3.sub(__mean);
                    let __var = __centered.clone().square().sum_dim(2).div_scalar(__hidden_f);
                    let __normalized = __centered.div(__var.add_scalar(0.00001f64).sqrt());
                    __normalized
                        .reshape(__dims)
                        .mul(__scale.reshape([1usize, __dims[1], 1usize, 1usize]))
                        .add(__bias.reshape([1usize, __dims[1], 1usize, 1usize]))
                };
                __result.cast(__orig_dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_group_norm_field_dynamic_emits_no_field() {
        let node = create_dynamic_group_norm_node("group_norm1", false);
        let code = codegen_field_init(&node);
        assert_eq!(code, "");
    }
}
