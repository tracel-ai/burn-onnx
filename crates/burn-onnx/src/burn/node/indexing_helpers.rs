use proc_macro2::TokenStream;
use quote::quote;

use crate::burn::ToTokens;

/// Emit on-device negative-index normalization for ONNX `GatherND` / `ScatterND`.
///
/// ONNX allows indices to be negative (Python-style: `-1` is the last element),
/// but burn's native `gather_nd`/`scatter_nd` panic on negatives. This helper
/// emits a block that casts indices to i64, builds a `[1, ..., 1, K]` tensor of
/// the dim sizes being indexed, and replaces every negative entry with
/// `idx + dim_size` via `mask_where`.
///
/// The K dims being indexed are `data_dims[batch_dims..batch_dims + K]`
/// (`batch_dims = 0` for ScatterND).
///
/// # Reserved locals
///
/// The emitted block introduces locals prefixed with `__nd_`. Callers
/// downstream may reference `__nd_data_dims`, `__nd_idx_dims`, `__nd_k`, and
/// `__nd_indices_norm`; the rest (`__nd_indices`, `__nd_dim_sizes`,
/// `__nd_bcast_shape`, `__nd_dims_tensor`, `__nd_mask`, `__nd_corrected`,
/// `__nd_i`) are internal scaffolding. The whole `__nd_*` namespace is
/// reserved by this helper, so callers should not introduce other `__nd_*`
/// bindings in the same block.
///
/// # Required bindings at the call site
///
/// `Tensor`, `B`, `Int`, and `self.device` must resolve in the enclosing
/// scope.
///
/// # Out-of-bounds indices
///
/// Only negative indices are normalized. Indices `>= dim_size` are passed
/// through unchanged; behavior on positive out-of-bounds is backend-defined
/// (some panic, others produce undefined data) per the native
/// `gather_nd`/`scatter_nd` contract.
pub(crate) fn negative_index_normalize(
    data: &TokenStream,
    indices: &TokenStream,
    indices_rank: usize,
    batch_dims: usize,
) -> TokenStream {
    let indices_rank_lit = indices_rank.to_tokens();
    let batch_dims_lit = batch_dims.to_tokens();
    quote! {
        let __nd_data_dims = #data.dims();
        let __nd_indices = #indices.cast(burn::tensor::DType::I64);
        let __nd_idx_dims = __nd_indices.dims();
        let __nd_k = __nd_idx_dims[#indices_rank_lit - 1];
        let mut __nd_dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(__nd_k);
        for __nd_i in 0..__nd_k {
            __nd_dim_sizes.push(__nd_data_dims[#batch_dims_lit + __nd_i] as i64);
        }
        let mut __nd_bcast_shape = [1usize; #indices_rank_lit];
        __nd_bcast_shape[#indices_rank_lit - 1] = __nd_k;
        let __nd_dims_tensor = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::from(__nd_dim_sizes.as_slice()),
            (&self.device, burn::tensor::DType::I64),
        )
        .reshape(__nd_bcast_shape);
        let __nd_mask = __nd_indices.clone().lower_elem(0i64);
        let __nd_corrected = __nd_indices.clone() + __nd_dims_tensor;
        let __nd_indices_norm = __nd_indices.mask_where(__nd_mask, __nd_corrected);
    }
}
