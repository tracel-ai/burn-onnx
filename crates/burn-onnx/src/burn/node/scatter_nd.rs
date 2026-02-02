use super::prelude::*;
use crate::burn::TensorKind;
use onnx_ir::scatter_nd::ScatterNDReduction;

impl NodeCodegen for onnx_ir::scatter_nd::ScatterNDNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let data = scope.arg(self.inputs.first().unwrap());
        let indices = scope.arg(&self.inputs[1]);
        let updates = scope.arg(&self.inputs[2]);
        let output = arg_to_ident(self.outputs.first().unwrap());

        let data_arg = self.inputs.first().unwrap();
        let data_kind = match &data_arg.ty {
            ArgType::Tensor(t) => TensorKind::from(t.dtype),
            _ => panic!("Expected tensor input for data"),
        };

        if matches!(data_kind, TensorKind::Bool)
            && !matches!(self.config.reduction, ScatterNDReduction::None)
        {
            panic!(
                "ScatterND with {:?} reduction not supported for bool tensors",
                self.config.reduction
            );
        }

        let reduction_body = match self.config.reduction {
            ScatterNDReduction::None => quote! {
                output_flat = output_flat.slice_assign(
                    [target_offset..target_offset + slice_size],
                    update_slice,
                );
            },
            ScatterNDReduction::Add => quote! {
                let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                output_flat = output_flat.slice_assign(
                    [target_offset..target_offset + slice_size],
                    existing.add(update_slice),
                );
            },
            ScatterNDReduction::Mul => quote! {
                let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                output_flat = output_flat.slice_assign(
                    [target_offset..target_offset + slice_size],
                    existing.mul(update_slice),
                );
            },
            ScatterNDReduction::Max => quote! {
                let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                let mask = update_slice.clone().greater_equal(existing.clone());
                let result = existing.mask_where(mask, update_slice);
                output_flat = output_flat.slice_assign(
                    [target_offset..target_offset + slice_size],
                    result,
                );
            },
            ScatterNDReduction::Min => quote! {
                let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                let mask = update_slice.clone().lower_equal(existing.clone());
                let result = existing.mask_where(mask, update_slice);
                output_flat = output_flat.slice_assign(
                    [target_offset..target_offset + slice_size],
                    result,
                );
            },
        };

        quote! {
            let #output = {
                let data_dims = #data.dims();
                let indices_dims = #indices.dims();
                let indices_data = #indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = #data.reshape([total_size]);
                let updates_flat = #updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 { idx += data_dims[j] as i64; }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat.clone().narrow(0, i * slice_size, slice_size);
                    #reduction_body
                }
                output_flat.reshape(data_dims)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::scatter_nd::{ScatterNDConfig, ScatterNDNodeBuilder, ScatterNDReduction};

    #[test]
    fn test_scatter_nd_none() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1>,
        ) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + slice_size], update_slice);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_add() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Add);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1>,
        ) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                    output_flat = output_flat
                        .slice_assign(
                            [target_offset..target_offset + slice_size],
                            existing.add(update_slice),
                        );
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_mul() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Mul);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1>,
        ) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                    output_flat = output_flat
                        .slice_assign(
                            [target_offset..target_offset + slice_size],
                            existing.mul(update_slice),
                        );
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_max() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Max);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1>,
        ) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                    let mask = update_slice.clone().greater_equal(existing.clone());
                    let result = existing.mask_where(mask, update_slice);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + slice_size], result);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_min() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Min);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::F32)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1>,
        ) -> Tensor<B, 1> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    let existing = output_flat.clone().narrow(0, target_offset, slice_size);
                    let mask = update_slice.clone().lower_equal(existing.clone());
                    let result = existing.mask_where(mask, update_slice);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + slice_size], result);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_int() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 2, DType::I64)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::I64)
            .output_tensor("output", 2, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 2, Int>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1, Int>,
        ) -> Tensor<B, 2, Int> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + slice_size], update_slice);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    fn test_scatter_nd_bool_none() {
        let config = ScatterNDConfig::new(ScatterNDReduction::None);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::Bool)
            .output_tensor("output", 1, DType::Bool)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<B, 1, Bool>,
            indices: Tensor<B, 2, Int>,
            updates: Tensor<B, 1, Bool>,
        ) -> Tensor<B, 1, Bool> {
            let output = {
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let indices_data = indices.to_data().convert::<i64>();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let r = data_dims.len();
                let q = indices_dims.len();
                let k = indices_dims[q - 1];
                let mut data_strides = alloc::vec![1usize; r];
                for i in (0..r.saturating_sub(1)).rev() {
                    data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                }
                let num_updates: usize = indices_dims[..q - 1].iter().product();
                let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                let total_size: usize = data_dims.iter().product();
                let mut output_flat = data.reshape([total_size]);
                let updates_flat = updates.reshape([num_updates * slice_size]);
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let update_slice = updates_flat
                        .clone()
                        .narrow(0, i * slice_size, slice_size);
                    output_flat = output_flat
                        .slice_assign([target_offset..target_offset + slice_size], update_slice);
                }
                output_flat.reshape(data_dims)
            };
            output
        }
        ");
    }

    #[test]
    #[should_panic(expected = "reduction not supported for bool tensors")]
    fn test_scatter_nd_bool_add_panics() {
        let config = ScatterNDConfig::new(ScatterNDReduction::Add);
        let node = ScatterNDNodeBuilder::new("scatter1")
            .input_tensor("data", 1, DType::Bool)
            .input_tensor("indices", 2, DType::I64)
            .input_tensor("updates", 1, DType::Bool)
            .output_tensor("output", 1, DType::Bool)
            .config(config)
            .build();
        codegen_forward_default(&node);
    }
}
