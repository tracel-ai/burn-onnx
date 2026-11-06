use super::prelude::*;
/// Expand a blocked quantization parameter along its quantized axis, trimming the final block.
pub(super) fn expand_blocks(
    value: TokenStream,
    input_dims: &TokenStream,
    rank: usize,
    axis: i64,
    block: i64,
) -> TokenStream {
    let axis = if axis < 0 {
        (axis + rank as i64) as usize
    } else {
        axis as usize
    };
    let expanded_rank = rank + 1;
    let repeated_axis = axis + 1;
    let block = block as usize;
    quote! {{
        let (parameter, axis_size) = (#value, #input_dims[#axis]);
        let mut shape = parameter.dims();
        shape[#axis] *= #block;
        parameter.unsqueeze_dim::<#expanded_rank>(#repeated_axis)
            .repeat_dim(#repeated_axis, #block).reshape::<#rank, _>(shape)
            .slice_dim(#axis, 0..axis_size)
    }}
}
