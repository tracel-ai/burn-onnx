use burn_pack::Tensor as PackTensor;

use super::prelude::*;

impl NodeCodegen for onnx_ir::node::layer_norm::LayerNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if !self.inputs[1].is_static() {
            return None;
        }
        let name = Ident::new(&self.name, Span::call_site());
        let scale_shape = self.inputs[1]
            .ty
            .static_shape_known()
            .expect("LayerNorm: scale tensor shape must be known at codegen time");
        let num_features = scale_shape[0].to_tokens();
        let epsilon = self.config.epsilon;
        let has_bias = self.inputs.get(2).is_some_and(|bias| !bias.is_optional());

        Some(Field::new(
            self.name.clone(),
            quote! {
                LayerNorm
            },
            quote! {
                let #name = LayerNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .with_bias(#has_bias)
                    .init(device);
            },
        ))
    }

    fn collect_tensors(&self, field_name: &str) -> Vec<PackTensor> {
        if !self.inputs[1].is_static() {
            return vec![];
        }
        use crate::burn::node_traits::create_deferred_tensor;

        let mut tensors = vec![];

        // Gamma (scale) tensor at input index 1
        if let Some(gamma_input) = self.inputs.get(1) {
            let gamma_path = format!("{}.gamma", field_name);
            if let Some(tensor) = create_deferred_tensor(gamma_input, &gamma_path) {
                tensors.push(tensor);
            }
        }

        // Beta (bias) tensor at input index 2 - only if present
        if self.inputs.len() > 2
            && let Some(beta_input) = self.inputs.get(2)
        {
            let beta_path = format!("{}.beta", field_name);
            if let Some(tensor) = create_deferred_tensor(beta_input, &beta_path) {
                tensors.push(tensor);
            }
        }
        // When bias is absent, Burn's LayerNorm is configured with .with_bias(false),
        // so no beta parameter exists and no tensor is needed

        tensors
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        if !self.inputs[1].is_static() {
            return forward_functional(self, scope);
        }

        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if self.config.full_precision {
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
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        if !self.inputs[1].is_static() {
            return;
        }
        imports.register("burn::nn::LayerNorm");
        imports.register("burn::nn::LayerNormConfig");
    }
}

/// LayerNorm through burn's functional `layer_norm`, for a scale that is a graph value,
/// spans several axes, or comes with the Mean/InvStdDev outputs.
///
/// `layer_norm` normalizes the last axis, so X is flattened to `[outer, inner]` at
/// `axis` and the scale and bias to `[inner]`.
fn forward_functional(
    node: &onnx_ir::node::layer_norm::LayerNormalizationNode,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let config = &node.config;
    let input_arg = &node.inputs[0];
    let rank = input_arg.ty.rank();
    let axis = config.axis.rem_euclid(rank as i64) as usize;
    let epsilon = config.epsilon;

    let input = scope.arg(input_arg);
    let scale = scope.arg(&node.inputs[1]);
    // Scale and bias broadcast against the normalized axes, so lower ranks gain
    // leading axes before they are expanded and flattened.
    let normalized_rank = rank - axis;
    // `expand` cannot infer a rank, so the unsqueeze names it.
    let lift = |arg: &Argument, value: TokenStream| {
        if arg.ty.rank() < normalized_rank {
            let r = normalized_rank.to_tokens();
            quote! { #value.unsqueeze::<#r>() }
        } else {
            value
        }
    };
    let scale = lift(&node.inputs[1], scale);
    let bias = node
        .inputs
        .get(2)
        .filter(|arg| !arg.is_optional())
        .map(|arg| {
            let value = scope.arg(arg);
            lift(arg, value)
        });

    // stash_type=1 computes in float32 and casts Y back.
    let (dtype_binding, to_f32, to_input_dtype) = if config.full_precision {
        (
            quote! { let dtype = #input.dtype(); },
            quote! { .cast(burn::tensor::DType::F32) },
            quote! { .cast(dtype) },
        )
    } else {
        (quote! {}, quote! {}, quote! {})
    };
    let normalized_dims = (axis..rank).map(|i| quote! { dims[#i] });
    let normalized = quote! { [#(#normalized_dims),*] };
    let bias = match bias {
        Some(bias) => quote! { Some(#bias.expand(#normalized).reshape([inner])#to_f32) },
        None => quote! { None },
    };

    let y = arg_to_ident(&node.outputs[0]);
    let stat = |index: usize| {
        node.outputs
            .get(index)
            .filter(|arg| !arg.is_optional())
            .map(arg_to_ident)
    };
    let (mean, inv_std) = (stat(1), stat(2));
    let needs_stats = mean.is_some() || inv_std.is_some();

    let flat = if needs_stats {
        quote! { flat.clone() }
    } else {
        quote! { flat }
    };
    let compute_y = quote! {
        burn::tensor::module::layer_norm(
            #flat,
            #scale.expand(#normalized).reshape([inner])#to_f32,
            #bias,
            #epsilon,
        )
        .reshape(dims)#to_input_dtype
    };

    let prelude = quote! {
        #dtype_binding
        let dims = #input.dims();
        let outer: usize = dims[..#axis].iter().product();
        let inner: usize = dims[#axis..].iter().product();
        let flat = #input.reshape([outer, inner])#to_f32;
    };

    if !needs_stats {
        return quote! {
            let #y = {
                #prelude
                #compute_y
            };
        };
    }

    // Mean and InvStdDev keep X's rank with the normalized axes reduced to 1.
    let mut names = vec![quote! { #y }];
    let mut values = vec![quote! { y }];
    if let Some(mean) = &mean {
        names.push(quote! { #mean });
        values.push(quote! { mean.reshape(stat_dims) });
    }
    if let Some(inv_std) = &inv_std {
        names.push(quote! { #inv_std });
        values.push(quote! { inv_std.reshape(stat_dims) });
    }

    let inv_std_binding = inv_std.is_some().then(|| {
        quote! {
            let inv_std = (flat - mean.clone())
                .square()
                .mean_dim(1)
                .add_scalar(#epsilon)
                .sqrt()
                .recip();
        }
    });

    quote! {
        let (#(#names),*) = {
            #prelude
            let y = #compute_y;
            let mut stat_dims = dims;
            for dim in stat_dims.iter_mut().skip(#axis) {
                *dim = 1;
            }
            let mean = flat.clone().mean_dim(1);
            #inv_std_binding
            (#(#values),*)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::layer_norm::{
        LayerNormConfig, LayerNormalizationNode, LayerNormalizationNodeBuilder,
    };

    fn create_layer_norm_node(name: &str) -> LayerNormalizationNode {
        let config = LayerNormConfig::new(1e-5, true);

        LayerNormalizationNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .input_static_tensor_shape("scale", vec![512], DType::F32)
            .input_static_tensor_shape("bias", vec![512], DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_layer_norm_forward() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let dtype = input.dtype();
                self.layer_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_layer_norm_forward_with_clone() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = {
                let dtype = input.clone().dtype();
                self.layer_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
            output
        }
        ");
    }

    fn create_functional_node(
        axis: i64,
        scale_rank: usize,
        outputs: usize,
    ) -> LayerNormalizationNode {
        let mut builder = LayerNormalizationNodeBuilder::new("layer_norm1")
            .input_tensor("input", 3, DType::F32)
            .input_tensor("scale", scale_rank, DType::F32)
            .input_tensor("bias", scale_rank, DType::F32)
            .output_tensor("output", 3, DType::F32);
        if outputs > 1 {
            builder = builder.output_tensor("mean", 3, DType::F32).output_tensor(
                "inv_std_dev",
                3,
                DType::F32,
            );
        }
        builder
            .config(LayerNormConfig::new(1e-5, true).with_axis(axis))
            .build()
    }

    #[test]
    fn test_layer_norm_runtime_scale() {
        let code = codegen_forward_default(&create_functional_node(-1, 1, 1));
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, scale: Tensor<1>, bias: Tensor<1>) -> Tensor<3> {
            let output = {
                let dtype = input.dtype();
                let dims = input.dims();
                let outer: usize = dims[..2usize].iter().product();
                let inner: usize = dims[2usize..].iter().product();
                let flat = input.reshape([outer, inner]).cast(burn::tensor::DType::F32);
                burn::tensor::module::layer_norm(
                        flat,
                        scale
                            .expand([dims[2usize]])
                            .reshape([inner])
                            .cast(burn::tensor::DType::F32),
                        Some(
                            bias
                                .expand([dims[2usize]])
                                .reshape([inner])
                                .cast(burn::tensor::DType::F32),
                        ),
                        0.00001f64,
                    )
                    .reshape(dims)
                    .cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_layer_norm_multi_axis_with_statistics() {
        let code = codegen_forward_default(&create_functional_node(1, 2, 3));
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            scale: Tensor<2>,
            bias: Tensor<2>,
        ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
            let (output, mean, inv_std_dev) = {
                let dtype = input.dtype();
                let dims = input.dims();
                let outer: usize = dims[..1usize].iter().product();
                let inner: usize = dims[1usize..].iter().product();
                let flat = input.reshape([outer, inner]).cast(burn::tensor::DType::F32);
                let y = burn::tensor::module::layer_norm(
                        flat.clone(),
                        scale
                            .expand([dims[1usize], dims[2usize]])
                            .reshape([inner])
                            .cast(burn::tensor::DType::F32),
                        Some(
                            bias
                                .expand([dims[1usize], dims[2usize]])
                                .reshape([inner])
                                .cast(burn::tensor::DType::F32),
                        ),
                        0.00001f64,
                    )
                    .reshape(dims)
                    .cast(dtype);
                let mut stat_dims = dims;
                for dim in stat_dims.iter_mut().skip(1usize) {
                    *dim = 1;
                }
                let mean = flat.clone().mean_dim(1);
                let inv_std = (flat - mean.clone())
                    .square()
                    .mean_dim(1)
                    .add_scalar(0.00001f64)
                    .sqrt()
                    .recip();
                (y, mean.reshape(stat_dims), inv_std.reshape(stat_dims))
            };
            (output, mean, inv_std_dev)
        }
        ");
    }

    #[test]
    fn test_layer_norm_inv_std_dev_only() {
        let mut node = create_functional_node(-1, 1, 3);
        node.outputs[1].name = String::new();
        node.outputs[1].value_source = onnx_ir::ir::ValueSource::Optional;
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            scale: Tensor<1>,
            bias: Tensor<1>,
        ) -> (Tensor<3>, Tensor<3>) {
            let (output, inv_std_dev) = {
                let dtype = input.dtype();
                let dims = input.dims();
                let outer: usize = dims[..2usize].iter().product();
                let inner: usize = dims[2usize..].iter().product();
                let flat = input.reshape([outer, inner]).cast(burn::tensor::DType::F32);
                let y = burn::tensor::module::layer_norm(
                        flat.clone(),
                        scale
                            .expand([dims[2usize]])
                            .reshape([inner])
                            .cast(burn::tensor::DType::F32),
                        Some(
                            bias
                                .expand([dims[2usize]])
                                .reshape([inner])
                                .cast(burn::tensor::DType::F32),
                        ),
                        0.00001f64,
                    )
                    .reshape(dims)
                    .cast(dtype);
                let mut stat_dims = dims;
                for dim in stat_dims.iter_mut().skip(2usize) {
                    *dim = 1;
                }
                let mean = flat.clone().mean_dim(1);
                let inv_std = (flat - mean.clone())
                    .square()
                    .mean_dim(1)
                    .add_scalar(0.00001f64)
                    .sqrt()
                    .recip();
                (y, inv_std.reshape(stat_dims))
            };
            (output, inv_std_dev)
        }
        ");
    }

    #[test]
    fn test_layer_norm_functional_stash_type_zero_without_bias() {
        let node = LayerNormalizationNodeBuilder::new("layer_norm1")
            .input_tensor("input", 3, DType::F16)
            .input_tensor("scale", 1, DType::F16)
            .output_tensor("output", 3, DType::F16)
            .config(LayerNormConfig::new(1e-5, false))
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>, scale: Tensor<1>) -> Tensor<3> {
            let output = {
                let dims = input.dims();
                let outer: usize = dims[..2usize].iter().product();
                let inner: usize = dims[2usize..].iter().product();
                let flat = input.reshape([outer, inner]);
                burn::tensor::module::layer_norm(
                        flat,
                        scale.expand([dims[2usize]]).reshape([inner]),
                        None,
                        0.00001f64,
                    )
                    .reshape(dims)
            };
            output
        }
        ");
    }
}
