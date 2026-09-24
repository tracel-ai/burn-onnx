//! Codegen shared by the conv and conv-transpose nodes for weights that arrive at run
//! time, which have no module to live in and go through burn's functional ops.

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
    let input_spatial = onnx_ir::node::padding::static_spatial_dims(&inputs[0].ty);
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
