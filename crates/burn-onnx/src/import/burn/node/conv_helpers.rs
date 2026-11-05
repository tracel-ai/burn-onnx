//! Codegen shared by the conv and conv-transpose nodes: functional ops for weights that
//! arrive at run time, which have no module to live in, and conv-transpose padding for both
//! the module and the functional path.

use super::prelude::*;
use onnx_ir::node::padding::AutoPad;

/// `Some(input)` for a present optional input, `None` when it is absent or omitted.
pub(crate) fn optional_input(
    scope: &mut ScopeAtPosition<'_>,
    arg: Option<&Argument>,
) -> TokenStream {
    match arg {
        Some(arg) if !arg.is_optional() => {
            let value = scope.arg(arg);
            quote! { Some(#value) }
        }
        _ => quote! { None },
    }
}

/// A conv's geometry, one entry per spatial axis. `explicit` holds the `(begin, end)`
/// pads used when `auto_pad` is not set.
pub(crate) struct ConvGeometry<'a> {
    pub auto_pad: &'a AutoPad,
    pub explicit: &'a [(usize, usize)],
    pub kernel: &'a [usize],
    pub stride: &'a [usize],
    pub dilation: &'a [usize],
    pub groups: usize,
}

/// `burn::tensor::module::<op>` (`conv1d`, `conv2d` or `conv3d`) over inputs
/// `[x, weight, bias?]`. SAME padding on an input sized only at run time is computed
/// from it before the input moves into the call.
pub(crate) fn functional_conv(
    scope: &mut ScopeAtPosition<'_>,
    inputs: &[Argument],
    output: &Argument,
    op: &str,
    geometry: ConvGeometry<'_>,
) -> TokenStream {
    let input = scope.arg(&inputs[0]);
    let weight = scope.arg(&inputs[1]);
    let bias = optional_input(scope, inputs.get(2));
    let operands = ConvOperands {
        input,
        weight,
        bias,
    };
    functional_conv_call(&inputs[0], operands, output, op, geometry)
}

/// The operand expressions of a functional conv call.
pub(crate) struct ConvOperands {
    pub input: TokenStream,
    pub weight: TokenStream,
    /// `Option<Tensor<1>>` expression.
    pub bias: TokenStream,
}

/// `burn::tensor::module::<op>` over operands already turned into expressions, so the
/// weight and bias can also come from a module field. `x` is the graph input behind
/// `operands.input`, read for its static spatial size.
pub(crate) fn functional_conv_call(
    x: &Argument,
    operands: ConvOperands,
    output: &Argument,
    op: &str,
    geometry: ConvGeometry<'_>,
) -> TokenStream {
    let ConvOperands {
        input,
        weight,
        bias,
    } = operands;
    let output = arg_to_ident(output);
    let op = Ident::new(op, Span::call_site());

    let ConvGeometry {
        auto_pad,
        explicit,
        kernel,
        stride,
        dilation,
        groups,
    } = geometry;
    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&x.ty);
    let static_padding = crate::burn::codegen::conv_padding_pairs(
        auto_pad,
        explicit,
        input_spatial.as_deref(),
        kernel,
        stride,
        dilation,
    );
    let runtime_padding = static_padding.is_none().then(|| {
        crate::burn::codegen::runtime_same_padding(auto_pad, &input, kernel, stride, dilation)
    });
    let padding = static_padding.unwrap_or_else(|| quote! { padding });
    let stride = stride.to_vec().to_tokens();
    let dilation = dilation.to_vec().to_tokens();
    let groups = groups.to_tokens();
    let call = quote! {
        burn::tensor::module::#op(
            #input,
            #weight,
            #bias,
            burn::tensor::ops::ConvOptions::new_with_padding(#stride, #padding, #dilation, #groups),
        )
    };
    match runtime_padding {
        None => quote! { let #output = #call; },
        Some(runtime_padding) => quote! {
            let #output = {
                let padding = #runtime_padding;
                #call
            };
        },
    }
}

/// A conv-transpose's geometry, one entry per spatial axis, as ONNX gives it.
pub(crate) struct ConvTransposeGeometry<'a> {
    pub auto_pad: &'a AutoPad,
    pub output_shape: Option<&'a [usize]>,
    /// Symmetric explicit pads, used when neither `auto_pad` nor `output_shape` is set.
    pub padding: &'a [usize],
    /// ONNX `output_padding`.
    pub padding_out: &'a [usize],
    pub kernel: &'a [usize],
    pub stride: &'a [usize],
    pub dilation: &'a [usize],
}

/// burn's `padding` and `padding_out` for a conv-transpose, per spatial axis.
pub(crate) struct TransposePadding {
    pub padding: Vec<usize>,
    pub padding_out: Vec<usize>,
    /// Output length of each axis where ONNX trims more from the end than from the start.
    /// burn trims `padding` from both ends and can only grow the end, so the excess is sliced
    /// off after.
    pub crop: Vec<Option<usize>>,
}

impl TransposePadding {
    /// `.slice(..)` cropping the conv output, or nothing when no axis needs it.
    pub(crate) fn crop_tokens(&self) -> TokenStream {
        if self.crop.iter().all(Option::is_none) {
            return quote! {};
        }
        let axes = self.crop.iter().map(|len| match len {
            Some(len) => {
                let len = len.to_tokens();
                quote! { 0..#len }
            }
            None => quote! { .. },
        });
        quote! { .slice(s![.., .., #(#axes),*]) }
    }
}

/// Resolve `auto_pad` and `output_shape` into burn's conv-transpose padding.
///
/// Per the ONNX spec, the total padding of an axis is `full + output_padding - output_size`,
/// where `full` is the length of the transposed convolution before any padding and
/// `output_size` is `output_shape` or, for SAME, `input_size * stride`. SAME_UPPER puts the
/// odd unit at the end, every other mode at the start. A negative total is clamped to zero and
/// the output grows at the end instead, as ONNX Runtime does.
pub(crate) fn transpose_padding(
    input: &Argument,
    geometry: ConvTransposeGeometry<'_>,
) -> TransposePadding {
    let ConvTransposeGeometry {
        auto_pad,
        output_shape,
        padding,
        padding_out,
        kernel,
        stride,
        dilation,
    } = geometry;
    let rank = padding.len();
    let same = matches!(auto_pad, AutoPad::SameUpper | AutoPad::SameLower);
    if !same && output_shape.is_none() {
        let padding = match auto_pad {
            AutoPad::Valid => vec![0; rank],
            _ => padding.to_vec(),
        };
        return TransposePadding {
            padding,
            padding_out: padding_out.to_vec(),
            crop: vec![None; rank],
        };
    }

    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&input.ty)
        .expect("ConvTranspose: onnx-ir rejects derived pads on a dynamic input");
    let mut resolved = TransposePadding {
        padding: Vec::with_capacity(rank),
        padding_out: Vec::with_capacity(rank),
        crop: Vec::with_capacity(rank),
    };
    for axis in 0..rank {
        let full =
            stride[axis] * (input_spatial[axis] - 1) + (kernel[axis] - 1) * dilation[axis] + 1;
        let size = output_shape.map_or(input_spatial[axis] * stride[axis], |shape| shape[axis]);
        let total = (full + padding_out[axis]).saturating_sub(size);
        let begin = match auto_pad {
            AutoPad::SameUpper => total / 2,
            _ => total - total / 2,
        };
        // With burn's `padding_out` set to `grow`, burn keeps positions `begin..full - begin +
        // grow` and ONNX wants `begin..begin + size`. A negative `grow` becomes a crop.
        let grow = (2 * begin + size) as i64 - full as i64;
        resolved.padding.push(begin);
        resolved.padding_out.push(grow.max(0) as usize);
        resolved.crop.push((grow < 0).then_some(size));
    }
    resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::DType;
    use onnx_ir::ir::{ArgType, TensorType};

    /// `(padding, padding_out, crop)` for one axis of size `input` with a 3-wide kernel.
    fn resolve(
        auto_pad: AutoPad,
        output_shape: Option<usize>,
        input: usize,
        stride: usize,
        output_padding: usize,
    ) -> (usize, usize, Option<usize>) {
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new_known(DType::F32, vec![1, 1, input])),
        );
        let resolved = transpose_padding(
            &input,
            ConvTransposeGeometry {
                auto_pad: &auto_pad,
                output_shape: output_shape.as_ref().map(std::slice::from_ref),
                padding: &[2],
                padding_out: &[output_padding],
                kernel: &[3],
                stride: &[stride],
                dilation: &[1],
            },
        );
        (
            resolved.padding[0],
            resolved.padding_out[0],
            resolved.crop[0],
        )
    }

    #[test]
    fn transpose_padding_rules() {
        // Explicit pads pass through, VALID zeroes them.
        assert_eq!(resolve(AutoPad::NotSet, None, 3, 2, 1), (2, 1, None));
        assert_eq!(resolve(AutoPad::Valid, None, 3, 2, 1), (0, 1, None));
        // Full 7, SAME wants 6: odd unit at the end is cropped, at the start it is trimmed.
        assert_eq!(resolve(AutoPad::SameUpper, None, 3, 2, 0), (0, 0, Some(6)));
        assert_eq!(resolve(AutoPad::SameLower, None, 3, 2, 0), (1, 1, None));
        // output_padding feeds the SAME total: full 7 + 1, SAME wants 6, even split.
        assert_eq!(resolve(AutoPad::SameUpper, None, 3, 2, 1), (1, 1, None));
        // output_shape without auto_pad splits like SAME_LOWER.
        assert_eq!(resolve(AutoPad::NotSet, Some(6), 3, 2, 0), (1, 1, None));
        // output_shape past the full result grows the end with no pads.
        assert_eq!(resolve(AutoPad::NotSet, Some(8), 3, 2, 0), (0, 1, None));
        // Stride larger than the kernel extent: negative SAME total, grown at the end.
        assert_eq!(resolve(AutoPad::SameUpper, None, 3, 4, 0), (0, 1, None));
    }
}
