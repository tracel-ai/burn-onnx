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
/// # Emitted locals
///
/// The emitted statements are meant to be spliced into the caller's block
/// expression. Callers downstream may reference `data_dims`, `idx_dims`,
/// `k`, and `indices_norm`; the rest (`indices_i64`, `dim_sizes`,
/// `bcast_shape`, `dims_tensor`, `negative`, `corrected`, `i`) are internal
/// scaffolding that callers should not redeclare in the same block.
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
        let data_dims = #data.dims();
        let indices_i64 = #indices.cast(burn::tensor::DType::I64);
        let idx_dims = indices_i64.dims();
        let k = idx_dims[#indices_rank_lit - 1];
        let mut dim_sizes: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(k);
        for i in 0..k {
            dim_sizes.push(data_dims[#batch_dims_lit + i] as i64);
        }
        let mut bcast_shape = [1usize; #indices_rank_lit];
        bcast_shape[#indices_rank_lit - 1] = k;
        let dims_tensor = Tensor::<1, Int>::from_data(
            burn::tensor::TensorData::from(dim_sizes.as_slice()),
            (&self.device, burn::tensor::DType::I64),
        )
        .reshape(bcast_shape);
        let negative = indices_i64.clone().lower_elem(0i64);
        let corrected = indices_i64.clone() + dims_tensor;
        let indices_norm = indices_i64.mask_where(negative, corrected);
    }
}
