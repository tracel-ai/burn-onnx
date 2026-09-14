use super::prelude::*;
use crate::burn::TensorKind;

impl NodeCodegen for onnx_ir::gathernd::GatherNDNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let data_arg = self.inputs.first().unwrap();
        let indices_arg = &self.inputs[1];
        let output_arg = self.outputs.first().unwrap();
        let output = arg_to_ident(output_arg);

        let data = scope.arg(data_arg);
        let indices = scope.arg(indices_arg);

        let (data_tensor, indices_tensor) = match (&data_arg.ty, &indices_arg.ty) {
            (ArgType::Tensor(d), ArgType::Tensor(i)) => (d, i),
            _ => {
                let msg = format!(
                    "GatherND node '{}': data and indices inputs must be tensors",
                    self.name
                );
                return quote! { let #output = { compile_error!(#msg); unreachable!() }; };
            }
        };
        let data_kind = TensorKind::from(data_tensor.dtype);
        let indices_rank = indices_tensor.rank;

        let batch_dims = self.config.batch_dims;
        let indices_rank_lit = indices_rank.to_tokens();
        let batch_dims_lit = batch_dims.to_tokens();

        let normalize =
            indexing_helpers::negative_index_normalize(&data, &indices, indices_rank, batch_dims);

        // For batch_dims > 0, prepend per-batch coords to the K-axis so the
        // native (batch-less) gather_nd matches ONNX's batched semantics.
        let augment_indices = if batch_dims == 0 {
            quote! {
                let aug_indices = indices_norm;
            }
        } else {
            quote! {
                let aug_indices = {
                    let mut target_shape = idx_dims;
                    target_shape[#indices_rank_lit - 1] = 1;
                    let mut components:
                        alloc::vec::Vec<Tensor<#indices_rank_lit, Int>> =
                        alloc::vec::Vec::with_capacity(#batch_dims_lit + 1);
                    for b in 0..#batch_dims_lit {
                        let dim_size = data_dims[b];
                        let arange = Tensor::<1, Int>::arange(
                            0i64..dim_size as i64,
                            (&self.device, burn::tensor::DType::I64),
                        );
                        let mut init_shape = [1usize; #indices_rank_lit];
                        init_shape[b] = dim_size;
                        let part: Tensor<#indices_rank_lit, Int> =
                            arange
                                .reshape(init_shape)
                                .expand(target_shape);
                        components.push(part);
                    }
                    components.push(indices_norm);
                    Tensor::cat(components, #indices_rank_lit - 1)
                };
            }
        };

        // Native gather_nd panics for bool data; round-trip through i64.
        let is_bool = matches!(data_kind, TensorKind::Bool);
        let gather_call = |indices_var: TokenStream| -> TokenStream {
            if is_bool {
                quote! {
                    #data.int()
                        .cast(burn::tensor::DType::I64)
                        .gather_nd(#indices_var)
                        .bool()
                }
            } else {
                quote! { #data.gather_nd(#indices_var) }
            }
        };

        if output_arg.ty.is_scalar() {
            // Native gather_nd with output rank 0 would produce a rank-0 tensor,
            // which burn does not represent. Reshape indices to add a leading 1
            // so the result is a rank-1 size-1 tensor.
            let inner_rank_lit = (indices_rank + 1).to_tokens();

            let scalar_tail = match &output_arg.ty {
                ArgType::ScalarNative(d) => on_device_to_native(quote! { result }, d),
                ArgType::ScalarTensor(_) => quote! { result },
                _ => unreachable!("is_scalar guard"),
            };

            let gather = gather_call(quote! { aug_indices });

            quote! {
                let #output = {
                    #normalize
                    let mut aug_shape = [1usize; #inner_rank_lit];
                    aug_shape[#inner_rank_lit - 1] = k;
                    let aug_indices: Tensor<#inner_rank_lit, Int> =
                        indices_norm.reshape(aug_shape);
                    let result = #gather;
                    #scalar_tail
                };
            }
        } else {
            let gather = gather_call(quote! { aug_indices });
            quote! {
                let #output = {
                    #normalize
                    #augment_indices
                    #gather
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gathernd::{GatherNDConfig, GatherNDNodeBuilder};

    #[test]
    fn test_gathernd_batch0_2d_data_2d_indices() {
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd1")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<2>, indices: Tensor<2, Int>) -> Tensor<1> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = indices_norm;
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_batch0_partial_index() {
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd2")
            .input_tensor("data", 2, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<2>, indices: Tensor<2, Int>) -> Tensor<2> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = indices_norm;
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_batch1() {
        let config = GatherNDConfig::new(1);
        let node = GatherNDNodeBuilder::new("gathernd3")
            .input_tensor("data", 3, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>, indices: Tensor<2, Int>) -> Tensor<2> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[1 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = {
                    let mut target_shape = idx_dims;
                    target_shape[2 - 1] = 1;
                    let mut components: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::with_capacity(
                        1 + 1,
                    );
                    for b in 0..1 {
                        let dim_size = data_dims[b];
                        let arange = Tensor::<
                            1,
                            Int,
                        >::arange(
                            0i64..dim_size as i64,
                            (&self.device, burn::tensor::DType::I64),
                        );
                        let mut init_shape = [1usize; 2];
                        init_shape[b] = dim_size;
                        let part: Tensor<2, Int> = arange
                            .reshape(init_shape)
                            .expand(target_shape);
                        components.push(part);
                    }
                    components.push(indices_norm);
                    Tensor::cat(components, 2 - 1)
                };
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_batch2() {
        // batch_dims=2 exercises the augment loop more than once: one arange/expand
        // per leading data dim is concatenated onto the K-axis.
        let config = GatherNDConfig::new(2);
        let node = GatherNDNodeBuilder::new("gathernd_b2")
            .input_tensor("data", 4, DType::F32)
            .input_tensor("indices", 3, DType::I64)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<4>, indices: Tensor<3, Int>) -> Tensor<3> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[3 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[2 + i] as i64);
                }
                let mut bcast_shape = [1usize; 3];
                bcast_shape[3 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = {
                    let mut target_shape = idx_dims;
                    target_shape[3 - 1] = 1;
                    let mut components: alloc::vec::Vec<Tensor<3, Int>> = alloc::vec::Vec::with_capacity(
                        2 + 1,
                    );
                    for b in 0..2 {
                        let dim_size = data_dims[b];
                        let arange = Tensor::<
                            1,
                            Int,
                        >::arange(
                            0i64..dim_size as i64,
                            (&self.device, burn::tensor::DType::I64),
                        );
                        let mut init_shape = [1usize; 3];
                        init_shape[b] = dim_size;
                        let part: Tensor<3, Int> = arange
                            .reshape(init_shape)
                            .expand(target_shape);
                        components.push(part);
                    }
                    components.push(indices_norm);
                    Tensor::cat(components, 3 - 1)
                };
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_int_data() {
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd_int")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<2, Int>, indices: Tensor<2, Int>) -> Tensor<1, Int> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = indices_norm;
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_scalar_output() {
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd_scalar")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 1, DType::I64)
            .output_scalar("output", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<1>, indices: Tensor<1, Int>) -> f32 {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[1 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 1];
                bcast_shape[1 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let mut aug_shape = [1usize; 2];
                aug_shape[2 - 1] = k;
                let aug_indices: Tensor<2, Int> = indices_norm.reshape(aug_shape);
                let result = data.gather_nd(aug_indices);
                (result).into_scalar::<f32>()
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_3d_data() {
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd_3d")
            .input_tensor("data", 3, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .output_tensor("output", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>, indices: Tensor<2, Int>) -> Tensor<2> {
            let output = {
                let data_dims = data.dims();
                let indices_i64 = indices.cast(burn::tensor::DType::I64);
                let idx_dims = indices_i64.dims();
                let k = idx_dims[2 - 1];
                let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
                for i in 0..k {
                    dim_sizes.push(data_dims[0 + i] as i64);
                }
                let mut bcast_shape = [1usize; 2];
                bcast_shape[2 - 1] = k;
                let dims_tensor = Tensor::<
                    1,
                    Int,
                >::from_data(
                        burn::tensor::TensorData::from(dim_sizes.as_slice()),
                        (&self.device, burn::tensor::DType::I64),
                    )
                    .reshape(bcast_shape);
                let negative = indices_i64.clone().lower_elem(0i64);
                let corrected = indices_i64.clone() + dims_tensor;
                let indices_norm = indices_i64.mask_where(negative, corrected);
                let aug_indices = indices_norm;
                data.gather_nd(aug_indices)
            };
            output
        }
        ");
    }
}
