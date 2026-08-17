use proc_macro2::TokenStream;
use quote::quote;

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

/// Generates numpy-style broadcasting for a binary operation.
///
/// Expands both operands to a common shape (per-dimension max) before applying `op`.
/// Both operands must already have the same rank (apply `leading_broadcast` first).
///
/// Use this for operations where Burn does not guarantee internal broadcasting
/// (e.g. `remainder`, see <https://github.com/tracel-ai/burn/issues/4712>).
/// For `add`/`mul`/`sub`/`div`, Burn handles broadcasting natively, so this
/// is not needed.
pub(crate) fn broadcast_binary_op(
    lhs: TokenStream,
    rhs: TokenStream,
    output_rank: usize,
    op: TokenStream,
) -> TokenStream {
    let rank_lit = proc_macro2::Literal::usize_suffixed(output_rank);
    quote! {
        {
            let __lhs = #lhs;
            let __rhs = #rhs;
            let __lhs_dims: [usize; #rank_lit] = __lhs.dims();
            let __rhs_dims: [usize; #rank_lit] = __rhs.dims();
            let mut __shape = [0i64; #rank_lit];
            #[allow(clippy::needless_range_loop)]
            for __i in 0..#rank_lit {
                __shape[__i] = core::cmp::max(__lhs_dims[__i] as i64, __rhs_dims[__i] as i64);
            }
            __lhs.expand(__shape).#op(__rhs.expand(__shape))
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
