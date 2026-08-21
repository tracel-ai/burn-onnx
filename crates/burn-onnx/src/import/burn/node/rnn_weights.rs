//! Shared codegen for RNN-family weights that arrive as runtime graph inputs.
//!
//! GRU, LSTM and RNN all pack their gate weights into three ONNX tensors that Burn
//! spreads across per-gate `Linear` modules:
//!
//! - `W`: `[num_directions, gates * hidden_size, input_size]`
//! - `R`: `[num_directions, gates * hidden_size, hidden_size]`
//! - `B`: `[num_directions, 2 * gates * hidden_size]`, all of `Wb` then all of `Rb`
//!
//! When those arrive as initializers the split runs at build time in each op's
//! `collect_*_snapshots` and the result is written into the `.bpk`. When they arrive
//! as graph inputs there is nothing to snapshot, so the same split is emitted as
//! generated code and applied to a module built inside `forward`.

use super::prelude::*;

/// How ONNX's packed `B` input maps onto Burn's two per-gate `Linear` biases.
#[derive(Clone, Copy)]
pub(crate) enum BiasLayout {
    /// `Wb[g]` on `input_transform`, `Rb[g]` on `hidden_transform`.
    ///
    /// GRU keeps the two apart because `linear_before_reset` applies the reset gate
    /// between them, which makes the recurrent bias observable on its own.
    Split,
    /// `Wb[g] + Rb[g]` on `input_transform`, zeros on `hidden_transform`.
    ///
    /// LSTM and RNN only ever add the two, so folding them is exact.
    Merged,
}

/// Packed-weight layout of one RNN-family operator.
pub(crate) struct GateLayout {
    /// Burn's per-gate field names, in Burn's own declaration order.
    pub gates: &'static [&'static str],
    /// ONNX gate index for each entry of `gates`.
    pub onnx_order: &'static [usize],
    /// How `B` splits across the two `Linear` modules of a gate.
    pub bias: BiasLayout,
}

/// Whether this node's weights arrive at forward time rather than as initializers.
///
/// `lift_constants` lifts `W`, `R` and `B` together or not at all, so one dynamic
/// weight puts all of them on the runtime path, where each is referenced by name.
pub(crate) fn weights_are_runtime(inputs: &[Argument]) -> bool {
    [1usize, 2, 3]
        .iter()
        .filter_map(|&index| inputs.get(index))
        .any(|arg| arg.is_dynamic())
}

/// Emit the statements that load runtime `W`/`R`/`B` into `module`'s gate parameters.
///
/// `module` must already be bound as `mut`. Only the parameters the Burn config
/// actually created are assigned: without a `B` input every `Linear` holds
/// `bias: None`, and the bias arms are skipped.
pub(crate) fn load_runtime_weights(
    module: &TokenStream,
    scope: &mut ScopeAtPosition<'_>,
    inputs: &[Argument],
    layout: &GateLayout,
    hidden_size: usize,
    num_directions: usize,
) -> TokenStream {
    let w = scope.arg(&inputs[1]);
    let r = scope.arg(&inputs[2]);
    let b = inputs
        .get(3)
        .filter(|arg| !arg.is_optional())
        .map(|arg| scope.arg(arg));

    let gate_count = layout.gates.len();
    let bind_bias = b.map(|b| quote! { let __b = #b; });
    let has_bias = bind_bias.is_some();

    let mut directions = quote! {};
    for direction in 0..num_directions {
        // Burn nests the two directions of a bidirectional module under `forward`
        // and `reverse`; a unidirectional module holds its gates directly.
        let gate_owner = if num_directions == 2 {
            let field = Ident::new(
                if direction == 0 { "forward" } else { "reverse" },
                Span::call_site(),
            );
            quote! { #module.#field }
        } else {
            quote! { #module }
        };

        let index = direction.to_tokens();
        let bias_dir = has_bias.then(|| {
            quote! { let __b_dir = __b.clone().select_dim::<1>(0, #index); }
        });

        let mut gates = quote! {};
        for (position, gate) in layout.gates.iter().enumerate() {
            let gate = Ident::new(gate, Span::call_site());
            let onnx_gate = layout.onnx_order[position];

            let start = (onnx_gate * hidden_size).to_tokens();
            let end = ((onnx_gate + 1) * hidden_size).to_tokens();

            // ONNX stores a gate as [hidden_size, in], Burn's row-major `Linear`
            // wants [in, hidden_size].
            gates.extend(quote! {
                #gate_owner.#gate.input_transform.weight = burn::module::Param::from_tensor(
                    __w_dir.clone().slice_dim(0, #start..#end).transpose(),
                );
                #gate_owner.#gate.hidden_transform.weight = burn::module::Param::from_tensor(
                    __r_dir.clone().slice_dim(0, #start..#end).transpose(),
                );
            });

            if !has_bias {
                continue;
            }

            let wb_start = start;
            let wb_end = end;
            let rb_start = ((gate_count + onnx_gate) * hidden_size).to_tokens();
            let rb_end = ((gate_count + onnx_gate + 1) * hidden_size).to_tokens();

            gates.extend(match layout.bias {
                BiasLayout::Split => quote! {
                    #gate_owner.#gate.input_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #wb_start..#wb_end),
                    ));
                    #gate_owner.#gate.hidden_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #rb_start..#rb_end),
                    ));
                },
                BiasLayout::Merged => quote! {
                    #gate_owner.#gate.input_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #wb_start..#wb_end)
                            + __b_dir.clone().slice_dim(0, #rb_start..#rb_end),
                    ));
                    #gate_owner.#gate.hidden_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #wb_start..#wb_end).zeros_like(),
                    ));
                },
            });
        }

        directions.extend(quote! {
            {
                let __w_dir = __w.clone().select_dim::<2>(0, #index);
                let __r_dir = __r.clone().select_dim::<2>(0, #index);
                #bias_dir
                #gates
            }
        });
    }

    quote! {
        let __w = #w;
        let __r = #r;
        #bind_bias
        #directions
    }
}

/// The Burn module an RNN-family forward pass runs through.
///
/// Static weights live in a struct field and need no setup. Runtime weights are
/// loaded into a module built inside `forward`, which `setup` binds.
pub(crate) struct Module {
    /// Statements binding the module. Empty when it is a struct field.
    pub setup: TokenStream,
    /// Expression naming the module: `self.gru1`, or the local `gru1`.
    pub expr: TokenStream,
}
