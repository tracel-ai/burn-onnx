#![allow(clippy::needless_range_loop)]

use super::prelude::*;
use proc_macro2::Literal;

impl NodeCodegen for onnx_ir::slice::SliceNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let output = arg_to_ident(self.outputs.first().unwrap());
        let input_arg = self.inputs.first().unwrap();

        match &input_arg.ty {
            ArgType::Tensor(tensor) => {
                // The runtime-bounds path reads `steps` itself.
                let steps_guard = if has_runtime_tensor_bounds(self) {
                    quote! {}
                } else {
                    runtime_steps_assertion(self, scope)
                };
                let body = generate_tensor_slice(self, input_arg, tensor.rank, scope, &output);
                quote! { #steps_guard #body }
            }
            ArgType::Shape(shape_rank) => {
                let output_arg = self.outputs.first().unwrap();
                match &output_arg.ty {
                    ArgType::Shape(_) => {
                        generate_shape_slice(self, input_arg, *shape_rank, &output, scope)
                    }
                    ArgType::Tensor(t) if t.rank == 1 => {
                        generate_shape_slice_to_tensor(self, input_arg, *shape_rank, &output, scope)
                    }
                    other => panic!(
                        "Slice node {}: unexpected output type for Shape input: {:?}",
                        self.name, other
                    ),
                }
            }
            ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                panic!("Unsupported input type for SliceNode")
            }
        }
    }
}

fn generate_tensor_slice(
    node: &onnx_ir::slice::SliceNode,
    input_arg: &Argument,
    rank: usize,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
    output: &proc_macro2::Ident,
) -> TokenStream {
    let input = scope.arg(input_arg);
    let mut ranges = vec![quote! { .. }; rank];

    // Build slice ranges based on parameter types
    match (&node.config.starts, &node.config.ends) {
        // Both static: simple case
        (onnx_ir::slice::SliceInput::Static(starts), onnx_ir::slice::SliceInput::Static(ends)) => {
            // Get steps (provided by onnx-ir with ONNX spec defaults)
            let steps = match &node.config.steps {
                Some(onnx_ir::slice::SliceInput::Static(s)) => s,
                _ => panic!("Steps must be Static for static slice"),
            };

            let static_shape = input_arg.ty.static_shape();
            let axes = match &node.config.axes {
                Some(onnx_ir::slice::SliceInput::Static(axes)) => Some(axes.as_slice()),
                _ => None,
            };

            // (axis, index into starts/ends/steps), in ONNX order. Axes are
            // already normalized to positive values by onnx-ir. Falling back to
            // the leading dimensions is the ONNX default for an absent `axes`
            // input; a runtime `axes` lands here too and is not yet supported.
            let axis_params = (0..starts.len().min(ends.len()))
                .filter_map(|i| match axes {
                    Some(axes) => axes.get(i).map(|&axis| (axis as usize, i)),
                    None => Some((i, i)),
                })
                .filter(|&(axis, _)| axis < rank);

            let mut needs_runtime_dims = false;
            for (axis, i) in axis_params {
                let step = *steps.get(i).expect("Step value missing for axis");
                let dim = static_shape.and_then(|shape| shape.get(axis).copied().flatten());
                match axis_range(starts[i], ends[i], step, dim) {
                    Some(tokens) => ranges[axis] = tokens,
                    // Reverse slice on an axis whose size is only known at
                    // runtime: read the bounds off the tensor instead.
                    None => {
                        ranges[axis] = runtime_reverse_range(axis, starts[i], ends[i], step);
                        needs_runtime_dims = true;
                    }
                }
            }

            if needs_runtime_dims {
                return generate_runtime_dim_slice(&input, output, &ranges);
            }
        }

        // Both runtime shapes: multi-dimensional slicing
        (
            onnx_ir::slice::SliceInput::Runtime(start_ref),
            onnx_ir::slice::SliceInput::Runtime(end_ref),
        ) => {
            let start_arg = &node.inputs[start_ref.input_index];
            let end_arg = &node.inputs[end_ref.input_index];

            if let (ArgType::Shape(start_rank), ArgType::Shape(end_rank)) =
                (&start_arg.ty, &end_arg.ty)
            {
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Check if axes are provided
                if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                    // Apply slicing to specified axes (already normalized by onnx-ir)
                    let num_dims = axes.len().min(*start_rank).min(*end_rank);
                    for i in 0..num_dims {
                        let axis_idx = axes[i] as usize;
                        if axis_idx < rank {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            ranges[axis_idx] = quote! { #start_name[#idx]..#end_name[#idx] };
                        }
                    }
                } else {
                    // No axes provided - use default behavior
                    let num_dims = start_rank.min(end_rank).min(&rank);
                    for (i, range) in ranges.iter_mut().enumerate().take(*num_dims) {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        *range = quote! { #start_name[#idx]..#end_name[#idx] };
                    }
                }
            } else if matches!(
                (&start_arg.ty, &end_arg.ty),
                (ArgType::Tensor(_), ArgType::Tensor(_))
            ) {
                return generate_runtime_bounds_slice(
                    node, start_arg, end_arg, rank, scope, &input, output,
                );
            } else if matches!(
                (&start_arg.ty, &end_arg.ty),
                (
                    ArgType::ScalarNative(_) | ArgType::ScalarTensor(_),
                    ArgType::ScalarNative(_) | ArgType::ScalarTensor(_)
                )
            ) {
                // Both scalars: use as single axis slice
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Get the axis to slice (provided by onnx-ir with ONNX spec defaults)
                let axis_idx = match &node.config.axes {
                    Some(onnx_ir::slice::SliceInput::Static(axes)) => {
                        *axes.first().expect("Axes array is empty for scalar slice") as usize
                    }
                    _ => panic!("Axes must be Static for scalar slice"),
                };

                if axis_idx < rank {
                    ranges[axis_idx] = quote! { (#start_name as usize)..(#end_name as usize) };
                }
            } else {
                // Mixed types: Shape/Tensor combination
                let start_name = arg_to_ident(start_arg);
                let end_name = arg_to_ident(end_arg);

                // Determine the number of dims we can safely index
                let start_rank = start_arg.ty.rank();
                let end_rank = end_arg.ty.rank();
                let num_slice_dims = start_rank.min(end_rank);

                // Need to extract tensor values at runtime
                let start_is_tensor = start_arg.ty.is_tensor();
                let end_is_tensor = end_arg.ty.is_tensor();

                let start_vec_var = quote! { start_vec };
                let end_vec_var = quote! { end_vec };

                // Build the extraction code
                let mut extraction_code = quote! {};

                if start_is_tensor {
                    extraction_code = quote! {
                        #extraction_code
                        let start_data = #start_name.to_data();
                        let #start_vec_var: alloc::vec::Vec<i64> = start_data.iter::<i64>().collect();
                    };
                }

                if end_is_tensor {
                    extraction_code = quote! {
                        #extraction_code
                        let end_data = #end_name.to_data();
                        let #end_vec_var: alloc::vec::Vec<i64> = end_data.iter::<i64>().collect();
                    };
                }

                // Check if axes are provided
                if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                    // Apply slicing to specified axes only
                    let mut ranges = vec![quote! { .. }; rank];

                    for (i, &axis) in axes.iter().enumerate() {
                        let axis_idx = axis as usize;
                        if axis_idx < rank {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            let start_expr = if start_is_tensor {
                                quote! { #start_vec_var[#idx] as usize }
                            } else {
                                quote! { #start_name[#idx] as usize }
                            };
                            let end_expr = if end_is_tensor {
                                quote! { #end_vec_var[#idx] as usize }
                            } else {
                                quote! { #end_name[#idx] as usize }
                            };
                            ranges[axis_idx] = quote! { #start_expr..#end_expr };
                        }
                    }

                    return quote! {
                        #extraction_code
                        let #output = #input.slice(s![#(#ranges),*]);
                    };
                } else {
                    // No axes provided - apply to first N dimensions
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            if i < num_slice_dims {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                let start_expr = if start_is_tensor {
                                    quote! { #start_vec_var[#idx] as usize }
                                } else {
                                    quote! { #start_name[#idx] as usize }
                                };
                                let end_expr = if end_is_tensor {
                                    quote! { #end_vec_var[#idx] as usize }
                                } else {
                                    quote! { #end_name[#idx] as usize }
                                };
                                quote! { #start_expr..#end_expr }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        #extraction_code
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
            }
        }

        // Static start, runtime end
        (
            onnx_ir::slice::SliceInput::Static(starts),
            onnx_ir::slice::SliceInput::Runtime(end_ref),
        ) => {
            let end_arg = &node.inputs[end_ref.input_index];

            match &end_arg.ty {
                ArgType::Shape(end_rank) => {
                    let end_name = arg_to_ident(end_arg);

                    // Check if axes are provided
                    if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                        // Apply slicing to specified axes (already normalized by onnx-ir)
                        let num_dims = axes.len().min(starts.len()).min(*end_rank);
                        for i in 0..num_dims {
                            let axis_idx = axes[i] as usize;
                            if axis_idx < rank {
                                let start = starts[i].to_tokens();
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                ranges[axis_idx] = quote! { #start..#end_name[#idx] };
                            }
                        }
                    } else {
                        // No axes provided - use default behavior
                        let num_dims = starts.len().min(*end_rank).min(rank);
                        for (i, range) in ranges.iter_mut().enumerate().take(num_dims) {
                            let start = starts[i].to_tokens();
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            *range = quote! { #start..#end_name[#idx] };
                        }
                    }
                }
                ArgType::Tensor(_) => {
                    // Static start, 1D tensor end
                    let end_name = arg_to_ident(end_arg);

                    // Generate code to extract values from end tensor
                    let end_data_var = quote! { end_data };
                    let end_vec_var = quote! { end_vec };

                    // Build ranges for each dimension
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            if i < starts.len() {
                                let start = Literal::i64_suffixed(starts[i]);
                                quote! {
                                    #start as usize..#end_vec_var[#idx] as usize
                                }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        let #end_data_var = #end_name.to_data();
                        let #end_vec_var: alloc::vec::Vec<i64> = #end_data_var.iter::<i64>().collect();
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
                ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                    // Static start, scalar end
                    let end_name = arg_to_ident(end_arg);

                    // Get the axis to slice (provided by onnx-ir with ONNX spec defaults)
                    let axis_idx = match &node.config.axes {
                        Some(onnx_ir::slice::SliceInput::Static(axes)) => {
                            *axes.first().expect("Axes array is empty for scalar slice") as usize
                        }
                        _ => panic!("Axes must be Static for scalar slice"),
                    };

                    if axis_idx < rank {
                        // Use the first start value (starts[0]) for the specified axis
                        let start = starts.first().expect("Starts array is empty").to_tokens();
                        ranges[axis_idx] = quote! { #start..(#end_name as usize) };
                    }
                }
            }
        }

        // Runtime start, static end
        (
            onnx_ir::slice::SliceInput::Runtime(start_ref),
            onnx_ir::slice::SliceInput::Static(ends),
        ) => {
            let start_arg = &node.inputs[start_ref.input_index];

            match &start_arg.ty {
                ArgType::Shape(start_rank) => {
                    let start_name = arg_to_ident(start_arg);

                    // Check if axes are provided
                    if let Some(onnx_ir::slice::SliceInput::Static(ref axes)) = node.config.axes {
                        // Apply slicing to specified axes (already normalized by onnx-ir)
                        let num_dims = axes.len().min(*start_rank).min(ends.len());
                        for i in 0..num_dims {
                            let axis_idx = axes[i] as usize;
                            if axis_idx < rank {
                                let idx = proc_macro2::Literal::usize_unsuffixed(i);
                                let end = ends[i].to_tokens();
                                ranges[axis_idx] = quote! { #start_name[#idx]..#end };
                            }
                        }
                    } else {
                        // No axes provided - use default behavior
                        let ends_len = ends.len();
                        let num_dims = start_rank.min(&ends_len).min(&rank);
                        for (i, range) in ranges.iter_mut().enumerate().take(*num_dims) {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            let end = ends[i].to_tokens();
                            *range = quote! { #start_name[#idx]..#end };
                        }
                    }
                }
                ArgType::Tensor(_) => {
                    // 1D tensor start, static end
                    let start_name = arg_to_ident(start_arg);

                    // Generate code to extract values from start tensor
                    let start_data_var = quote! { start_data };
                    let start_vec_var = quote! { start_vec };

                    // Build ranges for each dimension
                    let range_exprs: Vec<_> = (0..rank)
                        .map(|i| {
                            let idx = proc_macro2::Literal::usize_unsuffixed(i);
                            if i < ends.len() {
                                let end = Literal::i64_suffixed(ends[i]);
                                quote! {
                                    #start_vec_var[#idx] as usize..#end as usize
                                }
                            } else {
                                quote! { .. }
                            }
                        })
                        .collect();

                    return quote! {
                        let #start_data_var = #start_name.to_data();
                        let #start_vec_var: alloc::vec::Vec<i64> = #start_data_var.iter::<i64>().collect();
                        let #output = #input.slice(s![#(#range_exprs),*]);
                    };
                }
                ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                    panic!("Unsupported runtime start type for slice")
                }
            }
        }
    }

    quote! {
        let #output = #input.slice(s![#(#ranges),*]);
    }
}

/// Translate one ONNX `(start, end, step)` triple into a Burn `s!` range.
///
/// Returns `None` for a reverse slice on an axis whose size is unknown at
/// codegen time; those bounds have to be resolved from the tensor's own dims at
/// runtime (see `runtime_reverse_range`).
///
/// Forward steps mostly carry over as written: both ONNX and Burn read
/// `start..end` as a half-open range, resolve negative bounds against the
/// dimension size, and clamp the same way, so the bounds can stay symbolic and
/// work on an axis of unknown size. Only the sentinels need translating.
///
/// Reverse steps do not. ONNX walks backwards from `start` and stops before
/// `end`, whereas Burn always takes a forward `lo..hi` range and lets the sign of
/// the step decide the traversal order, so `s![0..5; -1]` yields `[4, 3, 2, 1, 0]`.
/// Lining them up needs `lo = end + 1` and `hi = start + 1` once both bounds are
/// resolved against the dimension size, which is also what turns the ONNX
/// `i64::MIN` "past the first element" sentinel into `lo = 0`.
fn axis_range(start: i64, end: i64, step: i64, dim: Option<usize>) -> Option<TokenStream> {
    if step < 0 {
        let (lo, hi) = slice_bounds(start, end, step, dim? as i64);
        let lo = Literal::usize_unsuffixed(lo);
        let hi = Literal::usize_unsuffixed(hi);
        let step = step.to_tokens();
        return Some(quote! { #lo..#hi;#step });
    }

    // Neither sentinel can be passed through to Burn, which resolves a negative
    // bound as `size as isize + index` and so overflows on i64::MIN.
    if end == i64::MIN {
        // Clamps to 0, so a forward step can never select anything.
        return Some(quote! { 0..0 });
    }
    let start = if start == i64::MIN {
        // Clamps to 0 whatever the dimension size turns out to be.
        quote! { 0 }
    } else {
        bound_tokens(start)
    };
    // i64::MAX means "to the end".
    let end = if end == i64::MAX {
        quote! {}
    } else {
        bound_tokens(end)
    };
    let step = if step == 1 {
        quote! {}
    } else {
        let step = step.to_tokens();
        quote! { ;#step }
    };
    Some(quote! { #start..#end #step })
}

/// Burn `lo..hi` bounds for one ONNX `(start, end, step)` triple over a
/// dimension of `dim`.
///
/// Clamping absorbs each sentinel in its ONNX-blessed position: `i64::MAX` is
/// the forward `end` and lands on `dim`; `i64::MIN` is the reverse `end` and
/// lands on `-1`, becoming `lo = 0` after the shift. The `.min(hi)` / `.max(lo)`
/// guards collapse the opposite pairings to an empty range.
///
/// Callers with a static `dim` only. `axis_range` routes just its reverse case
/// here on purpose: a forward slice can hand Burn the raw ONNX bounds and let it
/// clamp at runtime, so making it resolve `dim` too would push every forward
/// slice on a dynamic axis into the runtime path for nothing.
///
/// `generate_runtime_dim_slice` emits the reverse half of this arithmetic as a
/// closure for axes sized only at runtime, and `generate_runtime_bounds_slice`
/// emits all of it for runtime bounds; keep the three in sync.
fn slice_bounds(start: i64, end: i64, step: i64, dim: i64) -> (usize, usize) {
    if dim == 0 {
        return (0, 0);
    }
    let resolve = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
    let (lo, hi) = if step < 0 {
        let hi = resolve(start).clamp(-1, dim - 1) + 1;
        ((resolve(end).clamp(-1, dim - 1) + 1).min(hi), hi)
    } else {
        let lo = resolve(start).clamp(0, dim);
        // Rust panics on an inverted range; ONNX just selects nothing.
        (lo, resolve(end).clamp(0, dim).max(lo))
    };
    (lo as usize, hi as usize)
}

/// The `s!` range for a reverse slice whose bounds are only known at runtime.
/// Only valid inside the block `generate_runtime_dim_slice` emits.
fn runtime_reverse_range(axis: usize, start: i64, end: i64, step: i64) -> TokenStream {
    let axis_lit = Literal::usize_unsuffixed(axis);
    let start = i64_tokens(start);
    let end = i64_tokens(end);
    let step = step.to_tokens();
    quote! { reverse_bounds(slice_dims[#axis_lit], #start, #end);#step }
}

/// Wrap the slice in the block that defines `slice_dims` and `reverse_bounds`,
/// the helpers `runtime_reverse_range` emits calls to.
fn generate_runtime_dim_slice(
    input: &TokenStream,
    output: &proc_macro2::Ident,
    ranges: &[TokenStream],
) -> TokenStream {
    quote! {
        let #output = {
            let slice_input = #input;
            let slice_dims = slice_input.dims();
            // See `axis_range`: ONNX stops before `end` walking backwards, Burn
            // takes a forward range and reverses the traversal.
            let reverse_bounds = |dim: usize, start: i64, end: i64| -> core::ops::Range<usize> {
                if dim == 0 {
                    return 0..0;
                }
                let dim = dim as i64;
                let start = if start < 0 { start.saturating_add(dim) } else { start };
                let end = if end < 0 { end.saturating_add(dim) } else { end };
                let hi = start.clamp(-1, dim - 1) + 1;
                let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                (lo as usize)..(hi as usize)
            };
            slice_input.slice(s![#(#ranges),*])
        };
    }
}

/// Whether `starts` and `ends` are both runtime 1-D tensors, the case
/// `generate_runtime_bounds_slice` handles.
fn has_runtime_tensor_bounds(node: &onnx_ir::slice::SliceNode) -> bool {
    match (&node.config.starts, &node.config.ends) {
        (
            onnx_ir::slice::SliceInput::Runtime(start_ref),
            onnx_ir::slice::SliceInput::Runtime(end_ref),
        ) => matches!(
            (
                &node.inputs[start_ref.input_index].ty,
                &node.inputs[end_ref.input_index].ty
            ),
            (ArgType::Tensor(_), ArgType::Tensor(_))
        ),
        _ => false,
    }
}

/// Slice with `starts` and `ends` read from runtime tensors. `axes` and
/// `steps` may be static, runtime or absent, so every bound is resolved
/// against the input's dims at runtime, with the same clamping as
/// `slice_bounds`.
fn generate_runtime_bounds_slice(
    node: &onnx_ir::slice::SliceNode,
    start_arg: &Argument,
    end_arg: &Argument,
    rank: usize,
    scope: &mut ScopeAtPosition<'_>,
    input: &TokenStream,
    output: &proc_macro2::Ident,
) -> TokenStream {
    let starts = arg_to_ident(start_arg);
    let ends = arg_to_ident(end_arg);
    let rank_lit = Literal::usize_unsuffixed(rank);
    let rank_i64 = Literal::i64_suffixed(rank as i64);

    // ONNX defaults: axes = 0..len(starts), steps = 1.
    let axes = match &node.config.axes {
        Some(axes) => runtime_i64_vec(node, axes, scope),
        None => quote! { (0..slice_starts.len() as i64).collect() },
    };
    let steps = match &node.config.steps {
        Some(steps) => runtime_i64_vec(node, steps, scope),
        None => quote! { alloc::vec![1i64; slice_starts.len()] },
    };

    quote! {
        let #output = {
            let slice_input = #input;
            let slice_dims = slice_input.dims();
            let slice_starts: alloc::vec::Vec<i64> = #starts.to_data().iter::<i64>().collect();
            let slice_ends: alloc::vec::Vec<i64> = #ends.to_data().iter::<i64>().collect();
            let slice_axes: alloc::vec::Vec<i64> = #axes;
            let slice_steps: alloc::vec::Vec<i64> = #steps;
            let mut slices = [burn::tensor::Slice::full(); #rank_lit];
            for (i, &axis) in slice_axes.iter().enumerate() {
                let axis = (if axis < 0 { axis + #rank_i64 } else { axis }) as usize;
                let dim = slice_dims[axis] as i64;
                let resolve = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                let step = slice_steps[i];
                // See `axis_range`: ONNX stops before `end` walking backwards, Burn
                // takes a forward range and reverses the traversal.
                let (lo, hi) = if dim == 0 {
                    (0, 0)
                } else if step < 0 {
                    let hi = resolve(slice_starts[i]).clamp(-1, dim - 1) + 1;
                    ((resolve(slice_ends[i]).clamp(-1, dim - 1) + 1).min(hi), hi)
                } else {
                    let lo = resolve(slice_starts[i]).clamp(0, dim);
                    (lo, resolve(slice_ends[i]).clamp(0, dim).max(lo))
                };
                slices[axis] = burn::tensor::Slice::new(lo as isize, Some(hi as isize), step as isize);
            }
            slice_input.slice(slices)
        };
    }
}

/// An `alloc::vec::Vec<i64>` expression for a static or runtime `axes` /
/// `steps` parameter.
fn runtime_i64_vec(
    node: &onnx_ir::slice::SliceNode,
    param: &onnx_ir::slice::SliceInput,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    match param {
        onnx_ir::slice::SliceInput::Static(values) => {
            let values = values.iter().map(|&v| Literal::i64_suffixed(v));
            quote! { alloc::vec![#(#values),*] }
        }
        onnx_ir::slice::SliceInput::Runtime(param_ref) => {
            let arg = &node.inputs[param_ref.input_index];
            match &arg.ty {
                ArgType::Tensor(_) => {
                    let name = arg_to_ident(arg);
                    quote! { #name.to_data().iter::<i64>().collect() }
                }
                ArgType::Shape(_) => {
                    let name = arg_to_ident(arg);
                    quote! { #name.to_vec() }
                }
                ArgType::ScalarNative(_) | ArgType::ScalarTensor(_) => {
                    let value = scalar_as_i64(arg, scope.arg(arg));
                    quote! { alloc::vec![#value] }
                }
            }
        }
    }
}

/// Assert at model-run time that a runtime `steps` really is 1.
///
/// The runtime-bound slice paths other than `generate_runtime_bounds_slice`
/// emit a plain forward range and never read `steps`, so a non-unit value would
/// silently select the wrong elements. The
/// value cannot be checked here, and rejecting the model is not an option: the
/// ONNX backend test suite passes every Slice parameter as a graph input, so a
/// runtime `steps` is normal and virtually always 1. Check it where the value
/// exists instead. onnx-ir rejects a runtime `steps` alongside static bounds,
/// so only the runtime-bound paths reach this.
fn runtime_steps_assertion(
    node: &onnx_ir::slice::SliceNode,
    scope: &mut ScopeAtPosition<'_>,
) -> TokenStream {
    let Some(onnx_ir::slice::SliceInput::Runtime(steps_ref)) = &node.config.steps else {
        return quote! {};
    };
    let steps_arg = &node.inputs[steps_ref.input_index];
    let name = node.name.as_str();
    let all_unit = match &steps_arg.ty {
        ArgType::Shape(_) => {
            let steps = arg_to_ident(steps_arg);
            quote! { #steps.iter().all(|&s| s == 1) }
        }
        ArgType::ScalarNative(_) => {
            let steps = arg_to_ident(steps_arg);
            quote! { #steps == 1 }
        }
        _ => {
            let steps = scope.arg(steps_arg);
            quote! { #steps.to_data().iter::<i64>().all(|s| s == 1) }
        }
    };
    quote! {
        assert!(
            #all_unit,
            "Slice node {}: only step=1 is supported when `steps` is a runtime input",
            #name
        );
    }
}

/// A forward slice bound, as a literal Burn will accept.
///
/// Bare literals in an `s![..]` range infer as `i32`, so a bound outside that
/// range has to carry a suffix or the generated code will not compile. The value
/// itself needs no clamping: ONNX clamps out-of-range bounds and so does Burn.
fn bound_tokens(value: i64) -> TokenStream {
    if (i32::MIN as i64..=i32::MAX as i64).contains(&value) {
        value.to_tokens()
    } else {
        let lit = Literal::i64_suffixed(value);
        quote! { #lit }
    }
}

/// Spell the sentinel as `i64::MIN` rather than the 19-digit literal
/// `to_tokens` would emit, so the generated code names it. (The literal does
/// compile here, because the closure pins the parameter to `i64`; inside an
/// `s![..]` range it would infer `i32` and be rejected.)
fn i64_tokens(value: i64) -> TokenStream {
    if value == i64::MIN {
        quote! { i64::MIN }
    } else {
        let lit = Literal::i64_suffixed(value);
        quote! { #lit }
    }
}

fn generate_shape_slice(
    node: &onnx_ir::slice::SliceNode,
    input_arg: &Argument,
    shape_rank: usize,
    output: &proc_macro2::Ident,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> TokenStream {
    let shape_name = arg_to_ident(input_arg);

    // Get the output rank from the output type
    let output_rank = match &node.outputs.first().unwrap().ty {
        ArgType::Shape(rank) => rank,
        _ => panic!("Expected Shape output type for shape slice operation"),
    };
    let output_rank_lit = Literal::usize_unsuffixed(*output_rank);

    match (&node.config.starts, &node.config.ends) {
        (onnx_ir::slice::SliceInput::Static(starts), onnx_ir::slice::SliceInput::Static(ends))
            if starts.len() == 1 =>
        {
            let start_val = starts[0];
            let end_val = ends[0];

            // Get step value (provided by onnx-ir with ONNX spec defaults)
            let step_val = match &node.config.steps {
                Some(onnx_ir::slice::SliceInput::Static(steps)) => {
                    *steps.first().expect("Steps array is empty")
                }
                _ => panic!("Steps must be Static for shape slice"),
            };

            // A reverse step needs the same bound shift as a tensor slice,
            // otherwise the element count disagrees with the output rank
            // onnx-ir inferred and the generated `try_into` fails at runtime.
            // The rank is always static here, so both signs resolve now.
            let (actual_start, actual_end) =
                slice_bounds(start_val, end_val, step_val, shape_rank as i64);

            let start_lit = Literal::usize_unsuffixed(actual_start);
            let end_lit = Literal::usize_unsuffixed(actual_end);

            if step_val == 1 {
                quote! {
                    let #output: [i64; #output_rank_lit] = #shape_name[#start_lit..#end_lit].try_into().unwrap();
                }
            } else if step_val == -1 {
                // For negative step, we need to reverse the slice
                quote! {
                    let #output: [i64; #output_rank_lit] = {
                        let mut slice = #shape_name[#start_lit..#end_lit].to_vec();
                        slice.reverse();
                        slice.try_into().unwrap()
                    };
                }
            } else {
                // For other step values, we need to collect with step
                let step_abs = step_val.unsigned_abs() as usize;
                if step_val > 0 {
                    quote! {
                        let #output: [i64; #output_rank_lit] = {
                            let mut shape_out = [0i64; #output_rank_lit];
                            for (i, &s) in #shape_name[#start_lit..#end_lit].iter().step_by(#step_abs).enumerate() {
                                shape_out[i] = s;
                            }
                            shape_out
                        };
                    }
                } else {
                    quote! {
                        let #output: [i64; #output_rank_lit] = {
                            let mut shape_out = [0i64; #output_rank_lit];
                            for (i, &s) in #shape_name[#start_lit..#end_lit].iter().rev().step_by(#step_abs).enumerate() {
                                shape_out[i] = s;
                            }
                            shape_out
                        };
                    }
                }
            }
        }
        _ => {
            // Runtime slicing with scalars (unreachable from real flows since
            // the IR routes runtime bounds to a Tensor output, but kept for
            // direct-build callers; clamps match generate_shape_slice_to_tensor).
            let (start_expr, end_expr) = get_slice_range_expressions(node, scope);
            let shape_len_lit = Literal::i64_suffixed(shape_rank as i64);
            let shape_len_usize = Literal::usize_unsuffixed(shape_rank);

            quote! {
                let #output: [i64; #output_rank_lit] = {
                    let start_val = #start_expr as i64;
                    let end_val = #end_expr as i64;
                    let start_idx = if start_val < 0 {
                        (#shape_len_lit + start_val).max(0) as usize
                    } else {
                        (start_val as usize).min(#shape_len_usize)
                    };
                    let end_idx = if end_val == i64::MAX {
                        #shape_len_usize
                    } else if end_val < 0 {
                        (#shape_len_lit + end_val).max(0) as usize
                    } else {
                        (end_val as usize).min(#shape_len_usize)
                    };
                    #shape_name[start_idx..end_idx].try_into().unwrap()
                };
            }
        }
    }
}

fn generate_shape_slice_to_tensor(
    node: &onnx_ir::slice::SliceNode,
    input_arg: &Argument,
    shape_rank: usize,
    output: &proc_macro2::Ident,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> TokenStream {
    let shape_name = arg_to_ident(input_arg);
    let (start_expr, end_expr) = get_slice_range_expressions(node, scope);
    let shape_len_lit = Literal::i64_suffixed(shape_rank as i64);
    let shape_len_usize = Literal::usize_unsuffixed(shape_rank);

    quote! {
        let #output: Tensor<1, Int> = {
            let start_val = #start_expr as i64;
            let end_val = #end_expr as i64;
            let start_idx = if start_val < 0 {
                (#shape_len_lit + start_val).max(0) as usize
            } else {
                (start_val as usize).min(#shape_len_usize)
            };
            let end_idx = if end_val == i64::MAX {
                #shape_len_usize
            } else if end_val < 0 {
                (#shape_len_lit + end_val).max(0) as usize
            } else {
                (end_val as usize).min(#shape_len_usize)
            };
            let end_idx = end_idx.max(start_idx);
            let len = end_idx - start_idx;
            let slice_data: alloc::vec::Vec<i64> = #shape_name[start_idx..end_idx].to_vec();
            Tensor::<1, Int>::from_data(
                burn::tensor::TensorData::new(slice_data, [len]),
                (&self.device, burn::tensor::DType::I64),
            )
        };
    }
}

fn get_slice_range_expressions(
    node: &onnx_ir::slice::SliceNode,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> (TokenStream, TokenStream) {
    let start_expr = match &node.config.starts {
        onnx_ir::slice::SliceInput::Static(starts) => starts[0].to_tokens(),
        onnx_ir::slice::SliceInput::Runtime(start_ref) => {
            let start_arg = &node.inputs[start_ref.input_index];
            get_scalar_expr(start_arg, scope)
        }
    };

    let end_expr = match &node.config.ends {
        onnx_ir::slice::SliceInput::Static(ends) => ends[0].to_tokens(),
        onnx_ir::slice::SliceInput::Runtime(end_ref) => {
            let end_arg = &node.inputs[end_ref.input_index];
            get_scalar_expr(end_arg, scope)
        }
    };

    (start_expr, end_expr)
}

fn get_scalar_expr(
    arg: &Argument,
    scope: &mut super::super::scope::ScopeAtPosition<'_>,
) -> TokenStream {
    match &arg.ty {
        ArgType::ScalarNative(_) => {
            let name = arg_to_ident(arg);
            quote! { #name }
        }
        ArgType::ScalarTensor(dtype) => {
            let tensor = scope.arg(arg);
            on_device_to_native(tensor, dtype)
        }
        ArgType::Shape(_) => {
            let name = arg_to_ident(arg);
            // For single-dimension slicing, use the first element of the shape
            quote! { #name[0] }
        }
        ArgType::Tensor(_) => {
            // ONNX permits int32 or int64 for Slice bound tensors, so cast
            // before reading. The runtime length check guards the case where
            // the IR couldn't statically prove the bound is 1 element.
            let tensor = scope.arg(arg);
            // The bound is bound to a local before the block ends: the iterator
            // borrows `bound_data`, and only edition 2024 drops that temporary
            // before the block's locals.
            quote! {
                {
                    let bound_data = #tensor.clone()
                        .cast(burn::tensor::DType::I64)
                        .to_data();
                    assert_eq!(
                        bound_data.num_elements(), 1,
                        "Slice runtime bound must contain exactly one element, got {}",
                        bound_data.num_elements()
                    );
                    let bound_value = bound_data.iter::<i64>().next()
                        .expect("Slice runtime bound iter empty after num_elements==1 check");
                    bound_value
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::RuntimeInputRef;
    use onnx_ir::slice::{SliceConfig, SliceInput, SliceNodeBuilder};

    // ===== Static Tensor Slicing =====

    #[test]
    fn test_slice_static_simple() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![2]),
            axes: None,
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("data", 3, DType::F32)
            .output_tensor("sliced", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>) -> Tensor<3> {
            let sliced = data.slice(s![0..2, .., ..]);
            sliced
        }
        ");
    }

    #[test]
    fn test_slice_static_with_axes() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![1]),
            ends: SliceInput::Static(vec![3]),
            axes: Some(SliceInput::Static(vec![1])),
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("tensor", 3, DType::F32)
            .output_tensor("result", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<3>) -> Tensor<3> {
            let result = tensor.slice(s![.., 1..3, ..]);
            result
        }
        ");
    }

    #[test]
    fn test_slice_static_multiple_dims() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0, 1, 0]),
            ends: SliceInput::Static(vec![2, 3, 3]),
            axes: None,
            steps: Some(SliceInput::Static(vec![1, 1, 1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
            let output = input.slice(s![0..2, 1..3, 0..3]);
            output
        }
        ");
    }

    #[test]
    fn test_slice_static_with_step() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![10]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![2])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("x", 3, DType::F32)
            .output_tensor("y", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
            let y = x.slice(s![0..10; 2, .., ..]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_static_open_ended() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![5]),
            ends: SliceInput::Static(vec![i64::MAX]),
            axes: Some(SliceInput::Static(vec![2])),
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("tensor", 4, DType::F32)
            .output_tensor("tail", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<4>) -> Tensor<4> {
            let tail = tensor.slice(s![.., .., 5.., ..]);
            tail
        }
        ");
    }

    #[test]
    fn test_slice_static_open_ended_with_step() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![i64::MAX]),
            axes: Some(SliceInput::Static(vec![1])),
            steps: Some(SliceInput::Static(vec![3])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("data", 3, DType::F32)
            .output_tensor("every_third", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>) -> Tensor<3> {
            let every_third = data.slice(s![.., 0..; 3, ..]);
            every_third
        }
        ");
    }

    #[test]
    fn test_slice_static_multiple_axes() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![1, 2]),
            ends: SliceInput::Static(vec![5, 8]),
            axes: Some(SliceInput::Static(vec![0, 2])),
            steps: Some(SliceInput::Static(vec![1, 1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("volume", 4, DType::F32)
            .output_tensor("cropped", 4, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, volume: Tensor<4>) -> Tensor<4> {
            let cropped = volume.slice(s![1..5, .., 2..8, ..]);
            cropped
        }
        ");
    }

    // ===== Reverse (negative step) slicing =====

    #[test]
    fn test_slice_bounds() {
        use super::slice_bounds;

        // Reverse. ONNX x[4:MIN:-1] over dim 5 -> indices 4..0, Burn range 0..5.
        assert_eq!(slice_bounds(4, i64::MIN, -1, 5), (0, 5));
        // x[-1::-1] is the same slice spelled with a negative start.
        assert_eq!(slice_bounds(-1, i64::MIN, -1, 5), (0, 5));
        // x[3:0:-1] -> indices 3, 2, 1.
        assert_eq!(slice_bounds(3, 0, -1, 5), (1, 4));
        // x[-2:-4:-1] -> indices 3, 2.
        assert_eq!(slice_bounds(-2, -4, -1, 5), (2, 4));
        // Out-of-range bounds clamp instead of wrapping.
        assert_eq!(slice_bounds(99, i64::MIN, -1, 5), (0, 5));
        // But a start below -dim begins before the first element, selecting
        // nothing, rather than clamping up to index 0.
        assert_eq!(slice_bounds(-100, i64::MIN, -1, 5), (0, 0));
        assert_eq!(slice_bounds(-6, i64::MIN, -1, 5), (0, 0));
        assert_eq!(slice_bounds(-5, i64::MIN, -1, 5), (0, 1));
        // ONNX stops before `ends`, so this selects nothing.
        assert_eq!(slice_bounds(0, 5, -1, 5), (1, 1));

        // Forward. Both sentinels fall out of the same clamp.
        assert_eq!(slice_bounds(1, 4, 1, 5), (1, 4));
        assert_eq!(slice_bounds(1, i64::MAX, 1, 5), (1, 5));
        assert_eq!(slice_bounds(1, i64::MIN, 1, 5), (1, 1));
        assert_eq!(slice_bounds(-3, -1, 1, 5), (2, 4));
        // An inverted range would panic in Rust; ONNX just selects nothing.
        assert_eq!(slice_bounds(4, 1, 1, 5), (4, 4));

        // A zero-sized dimension has no valid index to clamp against.
        assert_eq!(slice_bounds(0, i64::MIN, -1, 0), (0, 0));
    }

    #[test]
    fn test_slice_reverse_full_axis() {
        // x[::-1] as exporters emit it: start at the last index and run past the
        // front with the i64::MIN sentinel.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![4]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5, 3], DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
            let y = x.slice(s![0..5; - 1, ..]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_reverse_partial_axis() {
        // x[3:0:-1] selects indices 3, 2, 1.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![3]),
            ends: SliceInput::Static(vec![0]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![1..4; - 1]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_reverse_negative_start() {
        // x[-1::-1] is the other common spelling of a full reverse.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![-1]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: Some(SliceInput::Static(vec![1])),
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![2, 6], DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
            let y = x.slice(s![.., 0..6; - 1]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_reverse_step_2() {
        // x[4::-2] selects indices 4, 2, 0.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![4]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![-2])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![0..5; - 2]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_reverse_empty() {
        // For step < 0 ONNX stops before `ends`, so 0..12 selects nothing.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![12]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![12], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![1..1; - 1]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_reverse_dynamic_dim() {
        // Without a static shape the bounds have to be resolved from the
        // tensor's own dims at runtime.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![-1]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("x", 2, DType::F32)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
            let y = {
                let slice_input = x;
                let slice_dims = slice_input.dims();
                let reverse_bounds = |
                    dim: usize,
                    start: i64,
                    end: i64,
                | -> core::ops::Range<usize> {
                    if dim == 0 {
                        return 0..0;
                    }
                    let dim = dim as i64;
                    let start = if start < 0 { start.saturating_add(dim) } else { start };
                    let end = if end < 0 { end.saturating_add(dim) } else { end };
                    let hi = start.clamp(-1, dim - 1) + 1;
                    let lo = (end.clamp(-1, dim - 1) + 1).min(hi);
                    (lo as usize)..(hi as usize)
                };
                slice_input.slice(s![reverse_bounds(slice_dims[0], - 1i64, i64::MIN); - 1, ..])
            };
            y
        }
        ");
    }

    #[test]
    fn test_slice_forward_start_min_sentinel() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![i64::MIN]),
            ends: SliceInput::Static(vec![3]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![0..3]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_forward_bound_beyond_i32() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![3_000_000_000]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![0..3000000000i64]);
            y
        }
        ");
    }

    #[test]
    fn test_slice_forward_step_min_sentinel() {
        // i64::MIN clamps to 0, so a forward step can never select anything.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![1]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor_shape("x", vec![5], DType::F32)
            .output_tensor("y", 1, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<1>) -> Tensor<1> {
            let y = x.slice(s![0..0]);
            y
        }
        ");
    }

    // ===== Runtime Tensor Slicing with Shape arguments =====

    #[test]
    fn test_slice_runtime_shape_with_axes() {
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "start_idx".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end_idx".to_string(),
                input_index: 2,
            }),
            axes: Some(SliceInput::Static(vec![1])),
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("data", 3, DType::F32)
            .input_shape("start_idx", 1)
            .input_shape("end_idx", 1)
            .output_tensor("sliced", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            data: Tensor<3>,
            start_idx: [i64; 1],
            end_idx: [i64; 1],
        ) -> Tensor<3> {
            let sliced = data.slice(s![.., start_idx[0]..end_idx[0], ..]);
            sliced
        }
        ");
    }

    #[test]
    fn test_slice_runtime_steps_asserted() {
        // A runtime `steps` cannot be inspected at codegen time, so the
        // assumption that it is 1 is checked where the value exists.
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "start_idx".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end_idx".to_string(),
                input_index: 2,
            }),
            axes: Some(SliceInput::Static(vec![0])),
            steps: Some(SliceInput::Runtime(RuntimeInputRef {
                name: "step_vals".to_string(),
                input_index: 3,
            })),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("data", 2, DType::F32)
            .input_shape("start_idx", 1)
            .input_shape("end_idx", 1)
            .input_shape("step_vals", 1)
            .output_tensor("sliced", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            data: Tensor<2>,
            start_idx: [i64; 1],
            end_idx: [i64; 1],
            step_vals: [i64; 1],
        ) -> Tensor<2> {
            assert!(
                step_vals.iter().all(| & s | s == 1),
                "Slice node {}: only step=1 is supported when `steps` is a runtime input",
                "slice1"
            );
            let sliced = data.slice(s![start_idx[0]..end_idx[0], ..]);
            sliced
        }
        "#);
    }

    #[test]
    fn test_slice_runtime_shape_no_axes() {
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "starts".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "ends".to_string(),
                input_index: 2,
            }),
            axes: None,
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("tensor", 2, DType::F32)
            .input_shape("starts", 2)
            .input_shape("ends", 2)
            .output_tensor("result", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<2>, starts: [i64; 2], ends: [i64; 2]) -> Tensor<2> {
            let result = tensor.slice(s![starts[0]..ends[0], starts[1]..ends[1]]);
            result
        }
        ");
    }

    // ===== Runtime Tensor Slicing with Scalar arguments =====

    #[test]
    fn test_slice_runtime_scalar() {
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "start".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end".to_string(),
                input_index: 2,
            }),
            axes: Some(SliceInput::Static(vec![0])),
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("x", 2, DType::F32)
            .input_scalar("start", DType::I64)
            .input_scalar("end", DType::I64)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, x: Tensor<2>, start: i64, end: i64) -> Tensor<2> {
            let y = x.slice(s![(start as usize).. (end as usize), ..]);
            y
        }
        ");
    }

    // ===== Mixed Static/Runtime Slicing =====

    #[test]
    fn test_slice_static_start_runtime_end_shape() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end_pos".to_string(),
                input_index: 1,
            }),
            axes: Some(SliceInput::Static(vec![1])),
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("data", 3, DType::F32)
            .input_shape("end_pos", 1)
            .output_tensor("prefix", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, data: Tensor<3>, end_pos: [i64; 1]) -> Tensor<3> {
            let prefix = data.slice(s![.., 0..end_pos[0], ..]);
            prefix
        }
        ");
    }

    #[test]
    fn test_slice_static_start_runtime_end_scalar() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![5]),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "stop".to_string(),
                input_index: 1,
            }),
            axes: Some(SliceInput::Static(vec![0])),
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("array", 2, DType::F32)
            .input_scalar("stop", DType::I64)
            .output_tensor("segment", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, array: Tensor<2>, stop: i64) -> Tensor<2> {
            let segment = array.slice(s![5.. (stop as usize), ..]);
            segment
        }
        ");
    }

    #[test]
    fn test_slice_runtime_start_static_end_shape() {
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "begin".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Static(vec![10]),
            axes: Some(SliceInput::Static(vec![0])),
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("tensor", 2, DType::F32)
            .input_shape("begin", 1)
            .output_tensor("chunk", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, tensor: Tensor<2>, begin: [i64; 1]) -> Tensor<2> {
            let chunk = tensor.slice(s![begin[0]..10, ..]);
            chunk
        }
        ");
    }

    #[test]
    fn test_slice_runtime_tensor_default_axes() {
        // Both starts/ends arrive as runtime 1-D tensors and no axes
        // input is provided, so axes default to 0..len(starts) at runtime
        // (here [0, 1], not the input rank 3).
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "starts".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "ends".to_string(),
                input_index: 2,
            }),
            axes: None,
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("x", 3, DType::F32)
            .input_tensor_shape("starts", vec![2], DType::I64)
            .input_tensor_shape("ends", vec![2], DType::I64)
            .output_tensor("y", 3, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<3>,
            starts: Tensor<1, Int>,
            ends: Tensor<1, Int>,
        ) -> Tensor<3> {
            let y = {
                let slice_input = x;
                let slice_dims = slice_input.dims();
                let slice_starts: alloc::vec::Vec<i64> = starts
                    .to_data()
                    .iter::<i64>()
                    .collect();
                let slice_ends: alloc::vec::Vec<i64> = ends.to_data().iter::<i64>().collect();
                let slice_axes: alloc::vec::Vec<i64> = (0..slice_starts.len() as i64).collect();
                let slice_steps: alloc::vec::Vec<i64> = alloc::vec![1i64; slice_starts.len()];
                let mut slices = [burn::tensor::Slice::full(); 3];
                for (i, &axis) in slice_axes.iter().enumerate() {
                    let axis = (if axis < 0 { axis + 3i64 } else { axis }) as usize;
                    let dim = slice_dims[axis] as i64;
                    let resolve = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                    let step = slice_steps[i];
                    let (lo, hi) = if dim == 0 {
                        (0, 0)
                    } else if step < 0 {
                        let hi = resolve(slice_starts[i]).clamp(-1, dim - 1) + 1;
                        ((resolve(slice_ends[i]).clamp(-1, dim - 1) + 1).min(hi), hi)
                    } else {
                        let lo = resolve(slice_starts[i]).clamp(0, dim);
                        (lo, resolve(slice_ends[i]).clamp(0, dim).max(lo))
                    };
                    slices[axis] = burn::tensor::Slice::new(
                        lo as isize,
                        Some(hi as isize),
                        step as isize,
                    );
                }
                slice_input.slice(slices)
            };
            y
        }
        ");
    }

    #[test]
    fn test_slice_runtime_tensor_axes_and_steps() {
        // Every parameter arrives as a runtime tensor, as in the ONNX backend
        // tests: axes and steps are read at runtime rather than assumed.
        let runtime = |name: &str, input_index| {
            SliceInput::Runtime(RuntimeInputRef {
                name: name.to_string(),
                input_index,
            })
        };
        let config = SliceConfig {
            starts: runtime("starts", 1),
            ends: runtime("ends", 2),
            axes: Some(runtime("axes", 3)),
            steps: Some(runtime("steps", 4)),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_tensor("x", 2, DType::F32)
            .input_tensor_shape("starts", vec![1], DType::I64)
            .input_tensor_shape("ends", vec![1], DType::I64)
            .input_tensor_shape("axes", vec![1], DType::I64)
            .input_tensor_shape("steps", vec![1], DType::I64)
            .output_tensor("y", 2, DType::F32)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            x: Tensor<2>,
            starts: Tensor<1, Int>,
            ends: Tensor<1, Int>,
            axes: Tensor<1, Int>,
            steps: Tensor<1, Int>,
        ) -> Tensor<2> {
            let y = {
                let slice_input = x;
                let slice_dims = slice_input.dims();
                let slice_starts: alloc::vec::Vec<i64> = starts
                    .to_data()
                    .iter::<i64>()
                    .collect();
                let slice_ends: alloc::vec::Vec<i64> = ends.to_data().iter::<i64>().collect();
                let slice_axes: alloc::vec::Vec<i64> = axes.to_data().iter::<i64>().collect();
                let slice_steps: alloc::vec::Vec<i64> = steps.to_data().iter::<i64>().collect();
                let mut slices = [burn::tensor::Slice::full(); 2];
                for (i, &axis) in slice_axes.iter().enumerate() {
                    let axis = (if axis < 0 { axis + 2i64 } else { axis }) as usize;
                    let dim = slice_dims[axis] as i64;
                    let resolve = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                    let step = slice_steps[i];
                    let (lo, hi) = if dim == 0 {
                        (0, 0)
                    } else if step < 0 {
                        let hi = resolve(slice_starts[i]).clamp(-1, dim - 1) + 1;
                        ((resolve(slice_ends[i]).clamp(-1, dim - 1) + 1).min(hi), hi)
                    } else {
                        let lo = resolve(slice_starts[i]).clamp(0, dim);
                        (lo, resolve(slice_ends[i]).clamp(0, dim).max(lo))
                    };
                    slices[axis] = burn::tensor::Slice::new(
                        lo as isize,
                        Some(hi as isize),
                        step as isize,
                    );
                }
                slice_input.slice(slices)
            };
            y
        }
        ");
    }

    // ===== Shape Slicing =====

    #[test]
    fn test_slice_shape_static() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![1]),
            ends: SliceInput::Static(vec![3]),
            axes: None,
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("input_shape", 4)
            .output_shape("output_shape", 4)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input_shape: [i64; 4]) -> [i64; 4] {
            let output_shape: [i64; 4] = input_shape[1..3].try_into().unwrap();
            output_shape
        }
        ");
    }

    #[test]
    fn test_slice_shape_static_negative_indices() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![-2]),
            ends: SliceInput::Static(vec![i64::MAX]),
            axes: None,
            steps: Some(SliceInput::Static(vec![1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("dims", 4)
            .output_shape("last_two", 2)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, dims: [i64; 4]) -> [i64; 2] {
            let last_two: [i64; 2] = dims[2..4].try_into().unwrap();
            last_two
        }
        ");
    }

    #[test]
    fn test_slice_shape_with_step_2() {
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![4]),
            axes: None,
            steps: Some(SliceInput::Static(vec![2])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("shape_in", 3)
            .output_shape("shape_out", 2)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_in: [i64; 3]) -> [i64; 2] {
            let shape_out: [i64; 2] = {
                let mut shape_out = [0i64; 2];
                for (i, &s) in shape_in[0..3].iter().step_by(2usize).enumerate() {
                    shape_out[i] = s;
                }
                shape_out
            };
            shape_out
        }
        ");
    }

    #[test]
    fn test_slice_shape_with_negative_step() {
        // shape[::-1]: start at the last index and run past the front.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![3]),
            ends: SliceInput::Static(vec![i64::MIN]),
            axes: None,
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("original", 4)
            .output_shape("reversed", 4)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, original: [i64; 4]) -> [i64; 4] {
            let reversed: [i64; 4] = {
                let mut slice = original[0..4].to_vec();
                slice.reverse();
                slice.try_into().unwrap()
            };
            reversed
        }
        ");
    }

    #[test]
    fn test_slice_shape_negative_step_empty() {
        // For step < 0 ONNX stops before `ends`, so 0..4 selects nothing.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![0]),
            ends: SliceInput::Static(vec![4]),
            axes: None,
            steps: Some(SliceInput::Static(vec![-1])),
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("original", 4)
            .output_shape("reversed", 0)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, original: [i64; 4]) -> [i64; 0] {
            let reversed: [i64; 0] = {
                let mut slice = original[1..1].to_vec();
                slice.reverse();
                slice.try_into().unwrap()
            };
            reversed
        }
        ");
    }

    #[test]
    fn test_slice_shape_runtime_tensor_bound() {
        // A bound that is a rank-1 tensor is read back on host. The value has
        // to be bound to a local: the iterator borrows the tensor data, which
        // pre-2024 editions drop after, not before, the block's locals.
        let config = SliceConfig {
            starts: SliceInput::Static(vec![1]),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end".to_string(),
                input_index: 1,
            }),
            axes: None,
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("shape_data", 3)
            .input_tensor("end", 1, DType::I64)
            .output_tensor("sliced_shape", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        // Spelled out separately from the snapshot: accepting a regenerated
        // snapshot must not quietly drop the binding and re-break edition 2021.
        assert!(code.contains("let bound_value ="));
        assert_snapshot!(code, @r#"
        pub fn forward(&self, shape_data: [i64; 3], end: Tensor<1, Int>) -> Tensor<1, Int> {
            let sliced_shape: Tensor<1, Int> = {
                let start_val = 1 as i64;
                let end_val = {
                    let bound_data = end.clone().cast(burn::tensor::DType::I64).to_data();
                    assert_eq!(
                        bound_data.num_elements(), 1,
                        "Slice runtime bound must contain exactly one element, got {}",
                        bound_data.num_elements()
                    );
                    let bound_value = bound_data
                        .iter::<i64>()
                        .next()
                        .expect("Slice runtime bound iter empty after num_elements==1 check");
                    bound_value
                } as i64;
                let start_idx = if start_val < 0 {
                    (3i64 + start_val).max(0) as usize
                } else {
                    (start_val as usize).min(3)
                };
                let end_idx = if end_val == i64::MAX {
                    3
                } else if end_val < 0 {
                    (3i64 + end_val).max(0) as usize
                } else {
                    (end_val as usize).min(3)
                };
                let end_idx = end_idx.max(start_idx);
                let len = end_idx - start_idx;
                let slice_data: alloc::vec::Vec<i64> = shape_data[start_idx..end_idx].to_vec();
                Tensor::<
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::new(slice_data, [len]),
                    (&self.device, burn::tensor::DType::I64),
                )
            };
            sliced_shape
        }
        "#);
    }

    #[test]
    fn test_slice_shape_runtime_to_tensor() {
        // When bounds are runtime, the IR cannot derive the output rank, so
        // it produces a rank-1 Int tensor instead of a fixed-size Shape array.
        let config = SliceConfig {
            starts: SliceInput::Runtime(RuntimeInputRef {
                name: "start".to_string(),
                input_index: 1,
            }),
            ends: SliceInput::Runtime(RuntimeInputRef {
                name: "end".to_string(),
                input_index: 2,
            }),
            axes: None,
            steps: None,
        };
        let node = SliceNodeBuilder::new("slice1")
            .input_shape("shape_data", 5)
            .input_scalar("start", DType::I64)
            .input_scalar("end", DType::I64)
            .output_tensor("sliced_shape", 1, DType::I64)
            .config(config)
            .build();
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, shape_data: [i64; 5], start: i64, end: i64) -> Tensor<1, Int> {
            let sliced_shape: Tensor<1, Int> = {
                let start_val = start as i64;
                let end_val = end as i64;
                let start_idx = if start_val < 0 {
                    (5i64 + start_val).max(0) as usize
                } else {
                    (start_val as usize).min(5)
                };
                let end_idx = if end_val == i64::MAX {
                    5
                } else if end_val < 0 {
                    (5i64 + end_val).max(0) as usize
                } else {
                    (end_val as usize).min(5)
                };
                let end_idx = end_idx.max(start_idx);
                let len = end_idx - start_idx;
                let slice_data: alloc::vec::Vec<i64> = shape_data[start_idx..end_idx].to_vec();
                Tensor::<
                    1,
                    Int,
                >::from_data(
                    burn::tensor::TensorData::new(slice_data, [len]),
                    (&self.device, burn::tensor::DType::I64),
                )
            };
            sliced_shape
        }
        ");
    }
}
