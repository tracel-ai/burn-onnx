use super::prelude::*;

impl NodeCodegen for onnx_ir::gathernd::GatherNDNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let batch_dims_lit = proc_macro2::Literal::usize_unsuffixed(self.config.batch_dims);

        let is_scalar = matches!(&self.outputs[0].ty, ArgType::Scalar(_));

        if is_scalar {
            // Scalar output: k == r, each index tuple fully specifies a single element
            quote! {
                let #output = {
                    let data_dims = #data.dims();
                    let indices_data = #indices.to_data().convert::<i64>();
                    let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();

                    let r = data_dims.len();
                    let b = #batch_dims_lit;

                    let mut data_strides = alloc::vec![1usize; r];
                    for i in (0..r.saturating_sub(1)).rev() {
                        data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                    }

                    let mut offset = 0usize;
                    for j in b..r {
                        let mut idx = indices_values[j - b];
                        if idx < 0 { idx += data_dims[j] as i64; }
                        offset += idx as usize * data_strides[j];
                    }

                    let data_flat = #data.reshape([data_dims.iter().product::<usize>()]);
                    data_flat.select(0, Tensor::<B, 1, Int>::from_data(
                        burn::tensor::TensorData::from([offset as i32].as_slice()),
                        &*self.device,
                    )).into_scalar()
                };
            }
        } else {
            let output_rank = match &self.outputs[0].ty {
                ArgType::Tensor(t) => t.rank,
                _ => unreachable!(),
            };
            let output_rank_lit = proc_macro2::Literal::usize_unsuffixed(output_rank);

            quote! {
                let #output = {
                    let data_dims = #data.dims();
                    let indices_dims = #indices.dims();
                    let indices_data = #indices.to_data().convert::<i64>();
                    let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();

                    let r = data_dims.len();
                    let q = indices_dims.len();
                    let b = #batch_dims_lit;
                    let k = indices_dims[q - 1];

                    // Compute data strides
                    let mut data_strides = alloc::vec![1usize; r];
                    for i in (0..r.saturating_sub(1)).rev() {
                        data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                    }

                    let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                    let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                    let slice_size: usize = if b + k < r { data_dims[b + k..].iter().product() } else { 1 };
                    let total_data_size: usize = data_dims.iter().product();
                    let batch_data_stride: usize = if b > 0 {
                        data_dims[b..].iter().product()
                    } else {
                        total_data_size
                    };
                    let total_slices = batch_count * lookups_per_batch;
                    let output_size = total_slices * slice_size;

                    // Compute flat indices for all output elements on CPU,
                    // then do a single select() on the GPU
                    let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(output_size);
                    for bi in 0..batch_count {
                        for li in 0..lookups_per_batch {
                            let lookup_idx = bi * lookups_per_batch + li;
                            let mut offset = bi * batch_data_stride;
                            for j in 0..k {
                                let mut idx = indices_values[lookup_idx * k + j];
                                if idx < 0 { idx += data_dims[b + j] as i64; }
                                offset += idx as usize * data_strides[b + j];
                            }
                            for s in 0..slice_size {
                                flat_indices.push((offset + s) as i32);
                            }
                        }
                    }

                    let data_flat = #data.reshape([total_data_size]);
                    let indices_tensor = Tensor::<B, 1, Int>::from_data(
                        burn::tensor::TensorData::from(flat_indices.as_slice()),
                        &*self.device,
                    );
                    let output_flat = data_flat.select(0, indices_tensor);

                    // Compute output shape: data_dims[:b] + indices_dims[b:q-1] + data_dims[b+k:]
                    let mut output_shape = [0usize; #output_rank_lit];
                    let mut si = 0;
                    for i in 0..b {
                        output_shape[si] = data_dims[i];
                        si += 1;
                    }
                    for i in b..q - 1 {
                        output_shape[si] = indices_dims[i];
                        si += 1;
                    }
                    for i in b + k..r {
                        output_shape[si] = data_dims[i];
                        si += 1;
                    }

                    output_flat.reshape(output_shape)
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
        pub fn forward(&self, data: Tensor<B, 2>, indices: Tensor<B, 2, Int>) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let b = 0;
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                let slice_size: usize = if b + k < r {
                    data_dims[b + k..].iter().product()
                } else {
                    1
                };
                let total_data_size: usize = data_dims.iter().product();
                let batch_data_stride: usize = if b > 0 {
                    data_dims[b..].iter().product()
                } else {
                    total_data_size
                };
                let total_slices = batch_count * lookups_per_batch;
                let output_size = total_slices * slice_size;
                let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(
                    output_size,
                );
                for bi in 0..batch_count {
                    for li in 0..lookups_per_batch {
                        let lookup_idx = bi * lookups_per_batch + li;
                        let mut offset = bi * batch_data_stride;
                        for j in 0..k {
                            let mut idx = indices_values[lookup_idx * k + j];
                            if idx < 0 {
                                idx += data_dims[b + j] as i64;
                            }
                            offset += idx as usize * data_strides[b + j];
                        }
                        for s in 0..slice_size {
                            flat_indices.push((offset + s) as i32);
                        }
                    }
                }
                let data_flat = data.reshape([total_data_size]);
                let indices_tensor = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(flat_indices.as_slice()),
                    &*self.device,
                );
                let output_flat = data_flat.select(0, indices_tensor);
                let mut output_shape = [0usize; 1];
                let mut si = 0;
                for i in 0..b {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                for i in b..q - 1 {
                    output_shape[si] = indices_dims[i];
                    si += 1;
                }
                for i in b + k..r {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                output_flat.reshape(output_shape)
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
        pub fn forward(&self, data: Tensor<B, 2>, indices: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let b = 0;
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                let slice_size: usize = if b + k < r {
                    data_dims[b + k..].iter().product()
                } else {
                    1
                };
                let total_data_size: usize = data_dims.iter().product();
                let batch_data_stride: usize = if b > 0 {
                    data_dims[b..].iter().product()
                } else {
                    total_data_size
                };
                let total_slices = batch_count * lookups_per_batch;
                let output_size = total_slices * slice_size;
                let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(
                    output_size,
                );
                for bi in 0..batch_count {
                    for li in 0..lookups_per_batch {
                        let lookup_idx = bi * lookups_per_batch + li;
                        let mut offset = bi * batch_data_stride;
                        for j in 0..k {
                            let mut idx = indices_values[lookup_idx * k + j];
                            if idx < 0 {
                                idx += data_dims[b + j] as i64;
                            }
                            offset += idx as usize * data_strides[b + j];
                        }
                        for s in 0..slice_size {
                            flat_indices.push((offset + s) as i32);
                        }
                    }
                }
                let data_flat = data.reshape([total_data_size]);
                let indices_tensor = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(flat_indices.as_slice()),
                    &*self.device,
                );
                let output_flat = data_flat.select(0, indices_tensor);
                let mut output_shape = [0usize; 2];
                let mut si = 0;
                for i in 0..b {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                for i in b..q - 1 {
                    output_shape[si] = indices_dims[i];
                    si += 1;
                }
                for i in b + k..r {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                output_flat.reshape(output_shape)
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
        pub fn forward(&self, data: Tensor<B, 3>, indices: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let b = 1;
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                let slice_size: usize = if b + k < r {
                    data_dims[b + k..].iter().product()
                } else {
                    1
                };
                let total_data_size: usize = data_dims.iter().product();
                let batch_data_stride: usize = if b > 0 {
                    data_dims[b..].iter().product()
                } else {
                    total_data_size
                };
                let total_slices = batch_count * lookups_per_batch;
                let output_size = total_slices * slice_size;
                let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(
                    output_size,
                );
                for bi in 0..batch_count {
                    for li in 0..lookups_per_batch {
                        let lookup_idx = bi * lookups_per_batch + li;
                        let mut offset = bi * batch_data_stride;
                        for j in 0..k {
                            let mut idx = indices_values[lookup_idx * k + j];
                            if idx < 0 {
                                idx += data_dims[b + j] as i64;
                            }
                            offset += idx as usize * data_strides[b + j];
                        }
                        for s in 0..slice_size {
                            flat_indices.push((offset + s) as i32);
                        }
                    }
                }
                let data_flat = data.reshape([total_data_size]);
                let indices_tensor = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(flat_indices.as_slice()),
                    &*self.device,
                );
                let output_flat = data_flat.select(0, indices_tensor);
                let mut output_shape = [0usize; 2];
                let mut si = 0;
                for i in 0..b {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                for i in b..q - 1 {
                    output_shape[si] = indices_dims[i];
                    si += 1;
                }
                for i in b + k..r {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                output_flat.reshape(output_shape)
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
        pub fn forward(
            &self,
            data: Tensor<B, 2, Int>,
            indices: Tensor<B, 2, Int>,
        ) -> Tensor<B, 1, Int> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let b = 0;
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                let slice_size: usize = if b + k < r {
                    data_dims[b + k..].iter().product()
                } else {
                    1
                };
                let total_data_size: usize = data_dims.iter().product();
                let batch_data_stride: usize = if b > 0 {
                    data_dims[b..].iter().product()
                } else {
                    total_data_size
                };
                let total_slices = batch_count * lookups_per_batch;
                let output_size = total_slices * slice_size;
                let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(
                    output_size,
                );
                for bi in 0..batch_count {
                    for li in 0..lookups_per_batch {
                        let lookup_idx = bi * lookups_per_batch + li;
                        let mut offset = bi * batch_data_stride;
                        for j in 0..k {
                            let mut idx = indices_values[lookup_idx * k + j];
                            if idx < 0 {
                                idx += data_dims[b + j] as i64;
                            }
                            offset += idx as usize * data_strides[b + j];
                        }
                        for s in 0..slice_size {
                            flat_indices.push((offset + s) as i32);
                        }
                    }
                }
                let data_flat = data.reshape([total_data_size]);
                let indices_tensor = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(flat_indices.as_slice()),
                    &*self.device,
                );
                let output_flat = data_flat.select(0, indices_tensor);
                let mut output_shape = [0usize; 1];
                let mut si = 0;
                for i in 0..b {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                for i in b..q - 1 {
                    output_shape[si] = indices_dims[i];
                    si += 1;
                }
                for i in b + k..r {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                output_flat.reshape(output_shape)
            };
            output
        }
        ");
    }

    #[test]
    fn test_gathernd_scalar_output() {
        // data rank=1, indices rank=1, k=1: output_rank = 1+1-1-1-0 = 0 -> scalar
        let config = GatherNDConfig::new(0);
        let node = GatherNDNodeBuilder::new("gathernd_scalar")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 1, DType::I64)
            .output_scalar("output", DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<B, 1>, indices: Tensor<B, 1, Int>) -> f32 {
            let output = {
                let data_dims = data.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let b = 0;
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let mut offset = 0usize;
                for j in b..r {
                    let mut idx = indices_values[j - b];
                    if idx < 0 {
                        idx += data_dims[j] as i64;
                    }
                    offset += idx as usize * data_strides[j];
                }
                let data_flat = data.reshape([data_dims.iter().product::<usize>()]);
                data_flat
                    .select(
                        0,
                        Tensor::<
                            B,
                            1,
                            Int,
                        >::from_data(
                            burn::tensor::TensorData::from([offset as i32].as_slice()),
                            &*self.device,
                        ),
                    )
                    .into_scalar()
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
        pub fn forward(&self, data: Tensor<B, 3>, indices: Tensor<B, 2, Int>) -> Tensor<B, 2> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let b = 0;
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let batch_count: usize = if b > 0 { data_dims[..b].iter().product() } else { 1 };
                let lookups_per_batch: usize = indices_dims[b..q - 1].iter().product();
                let slice_size: usize = if b + k < r {
                    data_dims[b + k..].iter().product()
                } else {
                    1
                };
                let total_data_size: usize = data_dims.iter().product();
                let batch_data_stride: usize = if b > 0 {
                    data_dims[b..].iter().product()
                } else {
                    total_data_size
                };
                let total_slices = batch_count * lookups_per_batch;
                let output_size = total_slices * slice_size;
                let mut flat_indices: alloc::vec::Vec<i32> = alloc::vec::Vec::with_capacity(
                    output_size,
                );
                for bi in 0..batch_count {
                    for li in 0..lookups_per_batch {
                        let lookup_idx = bi * lookups_per_batch + li;
                        let mut offset = bi * batch_data_stride;
                        for j in 0..k {
                            let mut idx = indices_values[lookup_idx * k + j];
                            if idx < 0 {
                                idx += data_dims[b + j] as i64;
                            }
                            offset += idx as usize * data_strides[b + j];
                        }
                        for s in 0..slice_size {
                            flat_indices.push((offset + s) as i32);
                        }
                    }
                }
                let data_flat = data.reshape([total_data_size]);
                let indices_tensor = Tensor::<
                    B,
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::from(flat_indices.as_slice()),
                    &*self.device,
                );
                let output_flat = data_flat.select(0, indices_tensor);
                let mut output_shape = [0usize; 2];
                let mut si = 0;
                for i in 0..b {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                for i in b..q - 1 {
                    output_shape[si] = indices_dims[i];
                    si += 1;
                }
                for i in b + k..r {
                    output_shape[si] = data_dims[i];
                    si += 1;
                }
                output_flat.reshape(output_shape)
            };
            output
        }
        ");
    }
}
