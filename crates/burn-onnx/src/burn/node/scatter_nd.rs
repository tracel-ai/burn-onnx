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

        // Choose the element type for to_data/from_data based on tensor kind
        let reduction_body = match self.config.reduction {
            ScatterNDReduction::None => quote! {
                output_values[dst_idx] = update_values[src_idx];
            },
            ScatterNDReduction::Add => quote! {
                output_values[dst_idx] = output_values[dst_idx] + update_values[src_idx];
            },
            ScatterNDReduction::Mul => quote! {
                output_values[dst_idx] = output_values[dst_idx] * update_values[src_idx];
            },
            ScatterNDReduction::Max => match data_kind {
                TensorKind::Float => quote! {
                    output_values[dst_idx] = f32::max(output_values[dst_idx], update_values[src_idx]);
                },
                TensorKind::Int => quote! {
                    output_values[dst_idx] = core::cmp::max(output_values[dst_idx], update_values[src_idx]);
                },
                TensorKind::Bool => {
                    panic!("ScatterND max reduction not supported for bool tensors")
                }
            },
            ScatterNDReduction::Min => match data_kind {
                TensorKind::Float => quote! {
                    output_values[dst_idx] = f32::min(output_values[dst_idx], update_values[src_idx]);
                },
                TensorKind::Int => quote! {
                    output_values[dst_idx] = core::cmp::min(output_values[dst_idx], update_values[src_idx]);
                },
                TensorKind::Bool => {
                    panic!("ScatterND min reduction not supported for bool tensors")
                }
            },
        };

        match data_kind {
            TensorKind::Float => quote! {
                let #output = {
                    let device = #data.device();
                    let data_dims = #data.dims();
                    let indices_dims = #indices.dims();
                    let data_data = #data.to_data().convert::<f32>();
                    let indices_data = #indices.to_data().convert::<i64>();
                    let updates_data = #updates.to_data().convert::<f32>();
                    let mut output_values: alloc::vec::Vec<f32> = data_data.into_vec::<f32>().unwrap();
                    let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();
                    let update_values: alloc::vec::Vec<f32> = updates_data.into_vec::<f32>().unwrap();
                    let r = data_dims.len();
                    let q = indices_dims.len();
                    let k = indices_dims[q - 1];
                    let mut data_strides = alloc::vec![1usize; r];
                    for i in (0..r.saturating_sub(1)).rev() {
                        data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                    }
                    let num_updates: usize = indices_dims[..q - 1].iter().product();
                    let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                    for i in 0..num_updates {
                        let mut target_offset = 0usize;
                        for j in 0..k {
                            let mut idx = indices_values[i * k + j];
                            if idx < 0 { idx += data_dims[j] as i64; }
                            target_offset += idx as usize * data_strides[j];
                        }
                        let src_offset = i * slice_size;
                        for s in 0..slice_size {
                            let dst_idx = target_offset + s;
                            let src_idx = src_offset + s;
                            #reduction_body
                        }
                    }
                    burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(output_values, data_dims),
                        &device,
                    )
                };
            },
            TensorKind::Int => quote! {
                let #output = {
                    let device = #data.device();
                    let data_dims = #data.dims();
                    let indices_dims = #indices.dims();
                    let data_data = #data.to_data().convert::<i64>();
                    let indices_data = #indices.to_data().convert::<i64>();
                    let updates_data = #updates.to_data().convert::<i64>();
                    let mut output_values: alloc::vec::Vec<i64> = data_data.into_vec::<i64>().unwrap();
                    let indices_values: alloc::vec::Vec<i64> = indices_data.into_vec::<i64>().unwrap();
                    let update_values: alloc::vec::Vec<i64> = updates_data.into_vec::<i64>().unwrap();
                    let r = data_dims.len();
                    let q = indices_dims.len();
                    let k = indices_dims[q - 1];
                    let mut data_strides = alloc::vec![1usize; r];
                    for i in (0..r.saturating_sub(1)).rev() {
                        data_strides[i] = data_strides[i + 1] * data_dims[i + 1];
                    }
                    let num_updates: usize = indices_dims[..q - 1].iter().product();
                    let slice_size: usize = if k < r { data_dims[k..].iter().product() } else { 1 };
                    for i in 0..num_updates {
                        let mut target_offset = 0usize;
                        for j in 0..k {
                            let mut idx = indices_values[i * k + j];
                            if idx < 0 { idx += data_dims[j] as i64; }
                            target_offset += idx as usize * data_strides[j];
                        }
                        let src_offset = i * slice_size;
                        for s in 0..slice_size {
                            let dst_idx = target_offset + s;
                            let src_idx = src_offset + s;
                            #reduction_body
                        }
                    }
                    burn::tensor::Tensor::from_data(
                        burn::tensor::TensorData::new(output_values, data_dims),
                        &device,
                    )
                };
            },
            TensorKind::Bool => panic!("ScatterND not supported for bool tensors"),
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<f32>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<f32>();
                let mut output_values: alloc::vec::Vec<f32> = data_data
                    .into_vec::<f32>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<f32> = updates_data
                    .into_vec::<f32>()
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = update_values[src_idx];
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<f32>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<f32>();
                let mut output_values: alloc::vec::Vec<f32> = data_data
                    .into_vec::<f32>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<f32> = updates_data
                    .into_vec::<f32>()
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = output_values[dst_idx] + update_values[src_idx];
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<f32>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<f32>();
                let mut output_values: alloc::vec::Vec<f32> = data_data
                    .into_vec::<f32>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<f32> = updates_data
                    .into_vec::<f32>()
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = output_values[dst_idx] * update_values[src_idx];
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<f32>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<f32>();
                let mut output_values: alloc::vec::Vec<f32> = data_data
                    .into_vec::<f32>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<f32> = updates_data
                    .into_vec::<f32>()
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = f32::max(
                            output_values[dst_idx],
                            update_values[src_idx],
                        );
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<f32>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<f32>();
                let mut output_values: alloc::vec::Vec<f32> = data_data
                    .into_vec::<f32>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<f32> = updates_data
                    .into_vec::<f32>()
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = f32::min(
                            output_values[dst_idx],
                            update_values[src_idx],
                        );
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
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
                let device = data.device();
                let data_dims = data.dims();
                let indices_dims = indices.dims();
                let data_data = data.to_data().convert::<i64>();
                let indices_data = indices.to_data().convert::<i64>();
                let updates_data = updates.to_data().convert::<i64>();
                let mut output_values: alloc::vec::Vec<i64> = data_data
                    .into_vec::<i64>()
                    .unwrap();
                let indices_values: alloc::vec::Vec<i64> = indices_data
                    .into_vec::<i64>()
                    .unwrap();
                let update_values: alloc::vec::Vec<i64> = updates_data
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
                for i in 0..num_updates {
                    let mut target_offset = 0usize;
                    for j in 0..k {
                        let mut idx = indices_values[i * k + j];
                        if idx < 0 {
                            idx += data_dims[j] as i64;
                        }
                        target_offset += idx as usize * data_strides[j];
                    }
                    let src_offset = i * slice_size;
                    for s in 0..slice_size {
                        let dst_idx = target_offset + s;
                        let src_idx = src_offset + s;
                        output_values[dst_idx] = update_values[src_idx];
                    }
                }
                burn::tensor::Tensor::from_data(
                    burn::tensor::TensorData::new(output_values, data_dims),
                    &device,
                )
            };
            output
        }
        ");
    }
}
