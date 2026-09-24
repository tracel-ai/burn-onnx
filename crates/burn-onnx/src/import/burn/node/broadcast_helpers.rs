use onnx_ir::ir::ArgType;
use proc_macro2::{Literal, TokenStream};
use quote::quote;

use crate::burn::ToTokens;

/// Build the shape literal `[1, channels, 1, ..., 1]` of length `rank` used to
/// broadcast a per-channel `[C]` tensor (gamma/beta/scale/bias) against an
/// activation tensor whose channel axis is at index 1.
pub(crate) fn channel_broadcast_shape(rank: usize, channels: TokenStream) -> TokenStream {
    let dims: Vec<TokenStream> = (0..rank)
        .map(|i| {
            if i == 1 {
                channels.clone()
            } else {
                quote! { 1usize }
            }
        })
        .collect();
    quote! { [#(#dims),*] }
}

/// Prepend leading unsqueeze dimensions to `expr` so its rank matches `target_rank`.
/// Returns `expr` unchanged when `expr_rank >= target_rank`.
pub(crate) fn leading_broadcast(
    expr: TokenStream,
    expr_rank: usize,
    target_rank: usize,
) -> TokenStream {
    if expr_rank >= target_rank {
        return expr;
    }
    let num_dims = target_rank - expr_rank;
    let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
    quote! { (#expr).unsqueeze_dims(&[#(#dims),*]) }
}

/// Materializes a `Shape` operand (`[i64; N]`) as an Int tensor of shape `[N]` with the dtype of
/// the on-device Int operand `other_ty`, then prepends unit dims so its rank matches
/// `other_ty.rank()`. `N` stays on the last axis, per ONNX trailing-axis broadcasting.
pub(crate) fn shape_operand_tensor(shape: TokenStream, other_ty: &ArgType) -> TokenStream {
    let dtype = other_ty.elem_type().to_tokens();
    let tensor = quote! {
        Tensor::<1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::from(&#shape as &[i64]),
            (&self.device, #dtype)
        )
    };
    leading_broadcast(tensor, 1, other_ty.rank())
}

/// Performs an element wise binary `op` over two `Shape` operands with numpy-style broadcasting of
/// a length 1 operand.
///
/// Output length is `max(lhs_len, rhs_len)` matching what `broadcast_output_type` infers.
///
/// An operand that is neither length 1 nor the output length cannot broadcast, we index it at 0
/// nonetheless so that the generated code still compiles, c.f. `broadcast_output_type` incompatible
/// handling.
pub(crate) fn shape_binary_elementwise(
    lhs: TokenStream,
    lhs_len: usize,
    rhs: TokenStream,
    rhs_len: usize,
    op: impl Fn(TokenStream, TokenStream) -> TokenStream,
) -> TokenStream {
    let out_len = lhs_len.max(rhs_len);
    let len_lit = Literal::usize_suffixed(out_len);

    let lhs_idx = if lhs_len == out_len {
        quote! { __i }
    } else {
        quote! { 0 }
    };
    let rhs_idx = if rhs_len == out_len {
        quote! { __i }
    } else {
        quote! { 0 }
    };
    let elem = op(quote! { __lhs[#lhs_idx] }, quote! { __rhs[#rhs_idx] });

    quote! {
        {
            let __lhs = #lhs;
            let __rhs = #rhs;
            core::array::from_fn::<i64, #len_lit, _>(|__i| #elem)
        }
    }
}

pub(crate) fn align_rhs_for_lhs_rank(
    rhs_expr: TokenStream,
    lhs_rank: usize,
    rhs_rank: usize,
    axis: Option<i64>,
) -> TokenStream {
    if lhs_rank <= rhs_rank {
        return rhs_expr;
    }

    if rhs_rank == 1 && lhs_rank > 1 {
        let axis = axis.unwrap_or(1);
        let axis_norm = if axis < 0 {
            (lhs_rank as i64 + axis) as usize
        } else {
            axis as usize
        };

        let dims: Vec<isize> = (0..lhs_rank)
            .filter(|&i| i != axis_norm)
            .map(|i| i as isize)
            .collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    } else {
        let num_dims = lhs_rank - rhs_rank;
        let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
        quote! { (#rhs_expr).unsqueeze_dims(&[#(#dims),*]) }
    }
}
