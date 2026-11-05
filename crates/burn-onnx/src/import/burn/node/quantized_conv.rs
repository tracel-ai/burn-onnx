use super::prelude::*;
use onnx_ir::node::padding::AutoPad;
use onnx_ir::quantized_conv::QuantizedConvConfig;

fn parameter(arg: &Argument, value: TokenStream, float: bool) -> TokenStream {
    let dtype = if float {
        quote! { burn::tensor::DType::F32 }
    } else {
        quote! { burn::tensor::DType::I32 }
    };
    let kind = if float {
        quote! {}
    } else {
        quote! { , burn::tensor::Int }
    };
    if matches!(arg.ty, ArgType::ScalarNative(_)) {
        if float {
            quote! { burn::tensor::Tensor::<1>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<f32>(#value)]), (self.device(), #dtype)) }
        } else {
            quote! { burn::tensor::Tensor::<1 #kind>::from_data(burn::tensor::TensorData::from([burn::tensor::ElementConversion::elem::<i32>(#value)]), (self.device(), #dtype)) }
        }
    } else {
        quote! { #value.cast(#dtype) }
    }
}
fn forward(
    inputs: &[Argument],
    output: &Argument,
    c: &QuantizedConvConfig,
    quantized: bool,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let rank = inputs[0].ty.rank();
    let n = rank - 2;
    let wi = if quantized { 3 } else { 1 };
    let x = scope.arg(&inputs[0]);
    let w = scope.arg(&inputs[wi]);
    let out = arg_to_ident(output);
    let groups = c.group as usize;
    let stride = c
        .strides
        .clone()
        .unwrap_or(vec![1; n])
        .iter()
        .map(|v| *v as usize)
        .collect::<Vec<_>>()
        .to_tokens();
    let dilation = c
        .dilations
        .clone()
        .unwrap_or(vec![1; n])
        .iter()
        .map(|v| *v as usize)
        .collect::<Vec<_>>()
        .to_tokens();
    let kernel = if let Some(k) = &c.kernel_shape {
        k.iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>()
            .to_tokens()
    } else {
        let dims = (2..rank).map(|i| quote! {weight.dims()[#i]});
        quote! {[#(#dims),*]}
    };
    let xzi = 2;
    let wzi = if quantized { 5 } else { 3 };
    let zero = |idx: usize, scope: &mut ScopeAtPosition<'_>| {
        inputs.get(idx).filter(|a| !a.is_optional()).map(|a| parameter(a,scope.arg(a),false)).unwrap_or_else(|| quote!{burn::tensor::Tensor::<1,burn::tensor::Int>::from_data([0i32], (self.device(),burn::tensor::DType::I32))})
    };
    let xz = zero(xzi, scope);
    let wz = zero(wzi, scope);
    let mut weight_shape = vec![1i64; rank];
    weight_shape[0] = -1;
    let weight_shape = weight_shape.to_tokens();
    let mut channel_shape = vec![1i64; rank];
    channel_shape[1] = -1;
    let channel_shape = channel_shape.to_tokens();
    let pads = c.pads.clone().unwrap_or(vec![0; n * 2]);
    let padding = match c.auto_pad {
        AutoPad::NotSet => {
            let pairs = (0..n).map(|i| {
                let a = pads[i] as usize;
                let b = pads[i + n] as usize;
                quote! {(#a,#b)}
            });
            quote! {[#(#pairs),*]}
        }
        AutoPad::Valid => quote! {[(0usize,0usize); #n]},
        AutoPad::SameUpper | AutoPad::SameLower => {
            let lower = matches!(c.auto_pad, AutoPad::SameLower);
            quote! {core::array::from_fn::<_, #n, _>(|axis| {
                let size=input.dims()[axis+2];
                let total=((size.div_ceil(strides[axis])-1)*strides[axis]+(kernel[axis]-1)*dilations[axis]+1).saturating_sub(size);
                let begin=if #lower {total.div_ceil(2)} else {total/2};
                (begin,total-begin)
            })}
        }
    };
    let unfold=(0..n).map(|i| {let dim=i+2;let d=rank+i+1;let last=d-1;quote!{
        let patches=patches.unfold::<#d,_>(#dim,(kernel[#i]-1)*dilations[#i]+1,strides[#i]);
        let patches=patches.slice_dim(#last,burn::tensor::Slice::new(0,None,dilations[#i] as isize));
    }});
    // [N,C,spatial...,kernel...] -> [N,spatial...,C,kernel...]. Then split C into groups.
    let permutation = std::iter::once(0)
        .chain(2..rank)
        .chain(std::iter::once(1))
        .chain(rank..rank + n)
        .collect::<Vec<_>>()
        .to_tokens();
    let spatial_dims = (2..rank)
        .map(|i| quote! {patches.dims()[#i]})
        .collect::<Vec<_>>();
    let out_dims = (0..n).map(|i| quote! {spatial[#i]});
    let requant = if quantized {
        let xs = parameter(&inputs[1], scope.arg(&inputs[1]), true);
        let ws = parameter(&inputs[4], scope.arg(&inputs[4]), true);
        let ys = parameter(&inputs[6], scope.arg(&inputs[6]), true);
        let yz = parameter(&inputs[7], scope.arg(&inputs[7]), false);
        let bias = inputs
            .get(8)
            .filter(|a| !a.is_optional())
            .map(|a| {
                let b = scope.arg(a);
                quote! {Some(#b)}
            })
            .unwrap_or_else(|| quote! {None::<burn::tensor::Tensor<1,burn::tensor::Int>>});
        let dtype = output.ty.elem_type().to_tokens();
        let (min, max) = if output.ty.elem_type() == DType::U8 {
            (0f32, 255f32)
        } else {
            (-128f32, 127f32)
        };
        let min = min.to_tokens();
        let max = max.to_tokens();
        (
            quote! {(input_scale,weight_scale,output_scale,output_zero,bias)},
            quote! {(#xs,#ws,#ys,#yz,#bias)},
            quote! {
                let accumulated=if let Some(bias)=bias {accumulated+bias.reshape(#channel_shape)} else {accumulated};
                (accumulated.float().cast(burn::tensor::DType::F32)
                    * (input_scale*weight_scale/output_scale).reshape(#channel_shape))
                    .round().add(output_zero.float().cast(burn::tensor::DType::F32).reshape([1;#rank]))
                    .clamp(#min,#max).int().cast(#dtype)
            },
        )
    } else {
        (quote! {()}, quote! {()}, quote! {accumulated})
    };
    let (bindings, params, result) = requant;
    // All graph inputs are evaluated before introducing temporaries that could shadow their names.
    quote! {
        let #out={
            let (input,weight,input_zero,weight_zero,#bindings)=(#x.cast(burn::tensor::DType::I32),#w.cast(burn::tensor::DType::I32),#xz,#wz,#params);
            let input=input-input_zero.reshape([1;#rank]);
            let weight=weight-weight_zero.reshape(#weight_shape);
            let kernel=#kernel;
            assert_eq!(kernel.as_slice(), &weight.dims()[2..], "kernel_shape must match the weight dimensions");
            let strides=#stride;
            let dilations=#dilation;
            let padding=#padding;
            let batch=input.dims()[0];
            let channels=input.dims()[1];
            let outputs=weight.dims()[0];
            assert_eq!(channels % #groups,0,"Input channels must divide into groups");
            assert_eq!(outputs % #groups,0,"Output channels must divide into groups");
            assert_eq!(weight.dims()[1],channels/#groups,"Weight channels must match input groups");
            let patches=input.pad(padding,burn::tensor::ops::PadMode::Constant(0.0));
            #(#unfold)*
            let spatial=[#(#spatial_dims),*];
            let positions=spatial.iter().product::<usize>();
            let features=channels/#groups*kernel.iter().product::<usize>();
            let patches=patches.permute(#permutation).reshape([batch,positions,#groups,features]).swap_dims(1,2);
            let weight=weight.reshape([1,#groups,outputs/#groups,features]).swap_dims(2,3);
            let accumulated=patches.matmul(weight).swap_dims(2,3).reshape([batch,outputs,#(#out_dims),*]);
            #result
        };
    }
}
macro_rules! codegen {
    ($node:ident,$quantized:literal) => {
        impl NodeCodegen for onnx_ir::quantized_conv::$node {
            fn inputs(&self) -> &[Argument] {
                &self.inputs
            }
            fn outputs(&self) -> &[Argument] {
                &self.outputs
            }
            fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
                forward(
                    &self.inputs,
                    &self.outputs[0],
                    &self.config,
                    $quantized,
                    scope,
                )
            }
        }
    };
}
codegen!(ConvIntegerNode, false);
codegen!(QLinearConvNode, true);

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use insta::assert_snapshot;
    use onnx_ir::quantized_conv::{ConvIntegerNodeBuilder, QLinearConvNodeBuilder};
    #[test]
    fn integer_default() {
        let node = ConvIntegerNodeBuilder::new("conv")
            .input_tensor("input", 3, DType::U8)
            .input_tensor("weight", 3, DType::I8)
            .output_tensor("output", 3, DType::I32)
            .config(QuantizedConvConfig::default())
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, input: Tensor<3, Int>, weight: Tensor<3, Int>) -> Tensor<3, Int> {
            let output = {
                let (input, weight, input_zero, weight_zero, ()) = (
                    input.cast(burn::tensor::DType::I32),
                    weight.cast(burn::tensor::DType::I32),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data([0i32], (self.device(), burn::tensor::DType::I32)),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data([0i32], (self.device(), burn::tensor::DType::I32)),
                    (),
                );
                let input = input - input_zero.reshape([1; 3usize]);
                let weight = weight - weight_zero.reshape([-1, 1, 1]);
                let kernel = [weight.dims()[2usize]];
                assert_eq!(
                    kernel.as_slice(), & weight.dims() [2..],
                    "kernel_shape must match the weight dimensions"
                );
                let strides = [1];
                let dilations = [1];
                let padding = [(0usize, 0usize)];
                let batch = input.dims()[0];
                let channels = input.dims()[1];
                let outputs = weight.dims()[0];
                assert_eq!(channels % 1usize, 0, "Input channels must divide into groups");
                assert_eq!(outputs % 1usize, 0, "Output channels must divide into groups");
                assert_eq!(
                    weight.dims() [1], channels / 1usize,
                    "Weight channels must match input groups"
                );
                let patches = input.pad(padding, burn::tensor::ops::PadMode::Constant(0.0));
                let patches = patches
                    .unfold::<
                        4usize,
                        _,
                    >(2usize, (kernel[0usize] - 1) * dilations[0usize] + 1, strides[0usize]);
                let patches = patches
                    .slice_dim(
                        3usize,
                        burn::tensor::Slice::new(0, None, dilations[0usize] as isize),
                    );
                let spatial = [patches.dims()[2usize]];
                let positions = spatial.iter().product::<usize>();
                let features = channels / 1usize * kernel.iter().product::<usize>();
                let patches = patches
                    .permute([0, 2, 1, 3])
                    .reshape([batch, positions, 1usize, features])
                    .swap_dims(1, 2);
                let weight = weight
                    .reshape([1, 1usize, outputs / 1usize, features])
                    .swap_dims(2, 3);
                let accumulated = patches
                    .matmul(weight)
                    .swap_dims(2, 3)
                    .reshape([batch, outputs, spatial[0usize]]);
                accumulated
            };
            output
        }
        "#);
    }
    #[test]
    fn integer_same_lower() {
        let node = ConvIntegerNodeBuilder::new("conv")
            .input_tensor("input", 5, DType::I8)
            .input_tensor("weight", 5, DType::U8)
            .input_scalar("xz", DType::I8)
            .input_tensor("wz", 1, DType::U8)
            .output_tensor("output", 5, DType::I32)
            .config(QuantizedConvConfig {
                auto_pad: AutoPad::SameLower,
                strides: Some(vec![2, 1, 1]),
                ..Default::default()
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(
            &self,
            input: Tensor<5, Int>,
            weight: Tensor<5, Int>,
            xz: i8,
            wz: Tensor<1, Int>,
        ) -> Tensor<5, Int> {
            let output = {
                let (input, weight, input_zero, weight_zero, ()) = (
                    input.cast(burn::tensor::DType::I32),
                    weight.cast(burn::tensor::DType::I32),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<i32>(xz),
                        ]),
                        (self.device(), burn::tensor::DType::I32),
                    ),
                    wz.cast(burn::tensor::DType::I32),
                    (),
                );
                let input = input - input_zero.reshape([1; 5usize]);
                let weight = weight - weight_zero.reshape([-1, 1, 1, 1, 1]);
                let kernel = [
                    weight.dims()[2usize],
                    weight.dims()[3usize],
                    weight.dims()[4usize],
                ];
                assert_eq!(
                    kernel.as_slice(), & weight.dims() [2..],
                    "kernel_shape must match the weight dimensions"
                );
                let strides = [2, 1, 1];
                let dilations = [1, 1, 1];
                let padding = core::array::from_fn::<
                    _,
                    3usize,
                    _,
                >(|axis| {
                    let size = input.dims()[axis + 2];
                    let total = ((size.div_ceil(strides[axis]) - 1) * strides[axis]
                        + (kernel[axis] - 1) * dilations[axis] + 1)
                        .saturating_sub(size);
                    let begin = if true { total.div_ceil(2) } else { total / 2 };
                    (begin, total - begin)
                });
                let batch = input.dims()[0];
                let channels = input.dims()[1];
                let outputs = weight.dims()[0];
                assert_eq!(channels % 1usize, 0, "Input channels must divide into groups");
                assert_eq!(outputs % 1usize, 0, "Output channels must divide into groups");
                assert_eq!(
                    weight.dims() [1], channels / 1usize,
                    "Weight channels must match input groups"
                );
                let patches = input.pad(padding, burn::tensor::ops::PadMode::Constant(0.0));
                let patches = patches
                    .unfold::<
                        6usize,
                        _,
                    >(2usize, (kernel[0usize] - 1) * dilations[0usize] + 1, strides[0usize]);
                let patches = patches
                    .slice_dim(
                        5usize,
                        burn::tensor::Slice::new(0, None, dilations[0usize] as isize),
                    );
                let patches = patches
                    .unfold::<
                        7usize,
                        _,
                    >(3usize, (kernel[1usize] - 1) * dilations[1usize] + 1, strides[1usize]);
                let patches = patches
                    .slice_dim(
                        6usize,
                        burn::tensor::Slice::new(0, None, dilations[1usize] as isize),
                    );
                let patches = patches
                    .unfold::<
                        8usize,
                        _,
                    >(4usize, (kernel[2usize] - 1) * dilations[2usize] + 1, strides[2usize]);
                let patches = patches
                    .slice_dim(
                        7usize,
                        burn::tensor::Slice::new(0, None, dilations[2usize] as isize),
                    );
                let spatial = [
                    patches.dims()[2usize],
                    patches.dims()[3usize],
                    patches.dims()[4usize],
                ];
                let positions = spatial.iter().product::<usize>();
                let features = channels / 1usize * kernel.iter().product::<usize>();
                let patches = patches
                    .permute([0, 2, 3, 4, 1, 5, 6, 7])
                    .reshape([batch, positions, 1usize, features])
                    .swap_dims(1, 2);
                let weight = weight
                    .reshape([1, 1usize, outputs / 1usize, features])
                    .swap_dims(2, 3);
                let accumulated = patches
                    .matmul(weight)
                    .swap_dims(2, 3)
                    .reshape([
                        batch,
                        outputs,
                        spatial[0usize],
                        spatial[1usize],
                        spatial[2usize],
                    ]);
                accumulated
            };
            output
        }
        "#);
    }
    #[test]
    fn quantized_channels_bias() {
        let node = QLinearConvNodeBuilder::new("conv")
            .input_tensor("input", 4, DType::U8)
            .input_scalar("xs", DType::F32)
            .input_scalar("xz", DType::U8)
            .input_tensor("weight", 4, DType::I8)
            .input_tensor("ws", 1, DType::F32)
            .input_tensor("wz", 1, DType::I8)
            .input_scalar("ys", DType::F32)
            .input_scalar("yz", DType::I8)
            .input_tensor("bias", 1, DType::I32)
            .output_tensor("output", 4, DType::I8)
            .config(QuantizedConvConfig {
                auto_pad: AutoPad::SameUpper,
                ..Default::default()
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(
            &self,
            input: Tensor<4, Int>,
            xs: f32,
            xz: u8,
            weight: Tensor<4, Int>,
            ws: Tensor<1>,
            wz: Tensor<1, Int>,
            ys: f32,
            yz: i8,
            bias: Tensor<1, Int>,
        ) -> Tensor<4, Int> {
            let output = {
                let (
                    input,
                    weight,
                    input_zero,
                    weight_zero,
                    (input_scale, weight_scale, output_scale, output_zero, bias),
                ) = (
                    input.cast(burn::tensor::DType::I32),
                    weight.cast(burn::tensor::DType::I32),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<i32>(xz),
                        ]),
                        (self.device(), burn::tensor::DType::I32),
                    ),
                    wz.cast(burn::tensor::DType::I32),
                    (
                        burn::tensor::Tensor::<
                            1,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<f32>(xs),
                            ]),
                            (self.device(), burn::tensor::DType::F32),
                        ),
                        ws.cast(burn::tensor::DType::F32),
                        burn::tensor::Tensor::<
                            1,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<f32>(ys),
                            ]),
                            (self.device(), burn::tensor::DType::F32),
                        ),
                        burn::tensor::Tensor::<
                            1,
                            burn::tensor::Int,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<i32>(yz),
                            ]),
                            (self.device(), burn::tensor::DType::I32),
                        ),
                        Some(bias),
                    ),
                );
                let input = input - input_zero.reshape([1; 4usize]);
                let weight = weight - weight_zero.reshape([-1, 1, 1, 1]);
                let kernel = [weight.dims()[2usize], weight.dims()[3usize]];
                assert_eq!(
                    kernel.as_slice(), & weight.dims() [2..],
                    "kernel_shape must match the weight dimensions"
                );
                let strides = [1, 1];
                let dilations = [1, 1];
                let padding = core::array::from_fn::<
                    _,
                    2usize,
                    _,
                >(|axis| {
                    let size = input.dims()[axis + 2];
                    let total = ((size.div_ceil(strides[axis]) - 1) * strides[axis]
                        + (kernel[axis] - 1) * dilations[axis] + 1)
                        .saturating_sub(size);
                    let begin = if false { total.div_ceil(2) } else { total / 2 };
                    (begin, total - begin)
                });
                let batch = input.dims()[0];
                let channels = input.dims()[1];
                let outputs = weight.dims()[0];
                assert_eq!(channels % 1usize, 0, "Input channels must divide into groups");
                assert_eq!(outputs % 1usize, 0, "Output channels must divide into groups");
                assert_eq!(
                    weight.dims() [1], channels / 1usize,
                    "Weight channels must match input groups"
                );
                let patches = input.pad(padding, burn::tensor::ops::PadMode::Constant(0.0));
                let patches = patches
                    .unfold::<
                        5usize,
                        _,
                    >(2usize, (kernel[0usize] - 1) * dilations[0usize] + 1, strides[0usize]);
                let patches = patches
                    .slice_dim(
                        4usize,
                        burn::tensor::Slice::new(0, None, dilations[0usize] as isize),
                    );
                let patches = patches
                    .unfold::<
                        6usize,
                        _,
                    >(3usize, (kernel[1usize] - 1) * dilations[1usize] + 1, strides[1usize]);
                let patches = patches
                    .slice_dim(
                        5usize,
                        burn::tensor::Slice::new(0, None, dilations[1usize] as isize),
                    );
                let spatial = [patches.dims()[2usize], patches.dims()[3usize]];
                let positions = spatial.iter().product::<usize>();
                let features = channels / 1usize * kernel.iter().product::<usize>();
                let patches = patches
                    .permute([0, 2, 3, 1, 4, 5])
                    .reshape([batch, positions, 1usize, features])
                    .swap_dims(1, 2);
                let weight = weight
                    .reshape([1, 1usize, outputs / 1usize, features])
                    .swap_dims(2, 3);
                let accumulated = patches
                    .matmul(weight)
                    .swap_dims(2, 3)
                    .reshape([batch, outputs, spatial[0usize], spatial[1usize]]);
                let accumulated = if let Some(bias) = bias {
                    accumulated + bias.reshape([1, -1, 1, 1])
                } else {
                    accumulated
                };
                (accumulated.float().cast(burn::tensor::DType::F32)
                    * (input_scale * weight_scale / output_scale).reshape([1, -1, 1, 1]))
                    .round()
                    .add(output_zero.float().cast(burn::tensor::DType::F32).reshape([1; 4usize]))
                    .clamp(-128.0, 127.0)
                    .int()
                    .cast(burn::tensor::DType::I8)
            };
            output
        }
        "#);
    }
    #[test]
    fn quantized_valid_without_bias() {
        let node = QLinearConvNodeBuilder::new("conv")
            .input_tensor("input", 3, DType::U8)
            .input_scalar("xs", DType::F32)
            .input_scalar("xz", DType::U8)
            .input_tensor("weight", 3, DType::I8)
            .input_scalar("ws", DType::F32)
            .input_scalar("wz", DType::I8)
            .input_scalar("ys", DType::F32)
            .input_scalar("yz", DType::U8)
            .output_tensor("output", 3, DType::U8)
            .config(QuantizedConvConfig {
                auto_pad: AutoPad::Valid,
                kernel_shape: Some(vec![3]),
                ..Default::default()
            })
            .build();
        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(
            &self,
            input: Tensor<3, Int>,
            xs: f32,
            xz: u8,
            weight: Tensor<3, Int>,
            ws: f32,
            wz: i8,
            ys: f32,
            yz: u8,
        ) -> Tensor<3, Int> {
            let output = {
                let (
                    input,
                    weight,
                    input_zero,
                    weight_zero,
                    (input_scale, weight_scale, output_scale, output_zero, bias),
                ) = (
                    input.cast(burn::tensor::DType::I32),
                    weight.cast(burn::tensor::DType::I32),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<i32>(xz),
                        ]),
                        (self.device(), burn::tensor::DType::I32),
                    ),
                    burn::tensor::Tensor::<
                        1,
                        burn::tensor::Int,
                    >::from_data(
                        burn::tensor::TensorData::from([
                            burn::tensor::ElementConversion::elem::<i32>(wz),
                        ]),
                        (self.device(), burn::tensor::DType::I32),
                    ),
                    (
                        burn::tensor::Tensor::<
                            1,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<f32>(xs),
                            ]),
                            (self.device(), burn::tensor::DType::F32),
                        ),
                        burn::tensor::Tensor::<
                            1,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<f32>(ws),
                            ]),
                            (self.device(), burn::tensor::DType::F32),
                        ),
                        burn::tensor::Tensor::<
                            1,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<f32>(ys),
                            ]),
                            (self.device(), burn::tensor::DType::F32),
                        ),
                        burn::tensor::Tensor::<
                            1,
                            burn::tensor::Int,
                        >::from_data(
                            burn::tensor::TensorData::from([
                                burn::tensor::ElementConversion::elem::<i32>(yz),
                            ]),
                            (self.device(), burn::tensor::DType::I32),
                        ),
                        None::<burn::tensor::Tensor<1, burn::tensor::Int>>,
                    ),
                );
                let input = input - input_zero.reshape([1; 3usize]);
                let weight = weight - weight_zero.reshape([-1, 1, 1]);
                let kernel = [3];
                assert_eq!(
                    kernel.as_slice(), & weight.dims() [2..],
                    "kernel_shape must match the weight dimensions"
                );
                let strides = [1];
                let dilations = [1];
                let padding = [(0usize, 0usize); 1usize];
                let batch = input.dims()[0];
                let channels = input.dims()[1];
                let outputs = weight.dims()[0];
                assert_eq!(channels % 1usize, 0, "Input channels must divide into groups");
                assert_eq!(outputs % 1usize, 0, "Output channels must divide into groups");
                assert_eq!(
                    weight.dims() [1], channels / 1usize,
                    "Weight channels must match input groups"
                );
                let patches = input.pad(padding, burn::tensor::ops::PadMode::Constant(0.0));
                let patches = patches
                    .unfold::<
                        4usize,
                        _,
                    >(2usize, (kernel[0usize] - 1) * dilations[0usize] + 1, strides[0usize]);
                let patches = patches
                    .slice_dim(
                        3usize,
                        burn::tensor::Slice::new(0, None, dilations[0usize] as isize),
                    );
                let spatial = [patches.dims()[2usize]];
                let positions = spatial.iter().product::<usize>();
                let features = channels / 1usize * kernel.iter().product::<usize>();
                let patches = patches
                    .permute([0, 2, 1, 3])
                    .reshape([batch, positions, 1usize, features])
                    .swap_dims(1, 2);
                let weight = weight
                    .reshape([1, 1usize, outputs / 1usize, features])
                    .swap_dims(2, 3);
                let accumulated = patches
                    .matmul(weight)
                    .swap_dims(2, 3)
                    .reshape([batch, outputs, spatial[0usize]]);
                let accumulated = if let Some(bias) = bias {
                    accumulated + bias.reshape([1, -1, 1])
                } else {
                    accumulated
                };
                (accumulated.float().cast(burn::tensor::DType::F32)
                    * (input_scale * weight_scale / output_scale).reshape([1, -1, 1]))
                    .round()
                    .add(output_zero.float().cast(burn::tensor::DType::F32).reshape([1; 3usize]))
                    .clamp(0.0, 255.0)
                    .int()
                    .cast(burn::tensor::DType::U8)
            };
            output
        }
        "#);
    }
}
