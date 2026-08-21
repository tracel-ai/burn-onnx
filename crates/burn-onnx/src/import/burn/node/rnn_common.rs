//! Shared codegen for the RNN family: GRU, LSTM and RNN.
//!
//! All three pack their gate weights into three ONNX tensors that Burn spreads
//! across per-gate `Linear` modules:
//!
//! - `W`: `[num_directions, gates * hidden_size, input_size]`
//! - `R`: `[num_directions, gates * hidden_size, hidden_size]`
//! - `B`: `[num_directions, 2 * gates * hidden_size]`, all of `Wb` then all of `Rb`
//!
//! When those arrive as initializers the split runs at build time in each op's
//! `collect_*_snapshots` and the result is written into the `.bpk`. When they arrive
//! as graph inputs there is nothing to snapshot, so the same split is emitted as
//! generated code and applied to a module built inside `forward`.
//!
//! The [`GateLayout`] each op declares names its gate order for both paths. Its
//! [`BiasLayout`] drives the runtime path only: each `collect_*_snapshots` still encodes
//! the same bias policy by hand, so the two have to be changed together.

use super::prelude::*;

/// How ONNX's packed `B` input maps onto Burn's two per-gate `Linear` biases.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BiasLayout {
    /// `Wb[g]` on `input_transform`, `Rb[g]` on `hidden_transform`.
    ///
    /// GRU needs this for the `h` gate when `linear_before_reset != 0`, where the reset
    /// gate is applied between the two: `Xt*Wh + rt * (Ht-1*Rh + Rbh) + Wbh`. Folding
    /// would be wrong there. It stays exact for the other gates and for
    /// `linear_before_reset = 0`, so GRU uses it unconditionally.
    Split,
    /// `Wb[g] + Rb[g]` on `input_transform`, zeros on `hidden_transform`.
    ///
    /// LSTM and RNN only ever add the two, so folding them is exact.
    Merged,
}

/// Packed-weight layout of one RNN-family operator.
#[derive(Debug)]
pub(crate) struct GateLayout {
    gates: &'static [(&'static str, usize)],
    bias: BiasLayout,
}

impl GateLayout {
    /// Declare an op's gate layout: each of Burn's per-gate field names paired with the
    /// ONNX gate index it holds, in Burn's own declaration order.
    ///
    /// The indices must be a permutation of `0..gates.len()`, which is what lets
    /// [`count`](Self::count) stand for both "Burn fields" and "ONNX gates" in the
    /// stride arithmetic. A duplicate or missing index would otherwise produce silently
    /// wrong weights, so it is checked here: these are consts, so a mistake is a
    /// compile error in this crate rather than a wrong number at inference time.
    pub const fn new(gates: &'static [(&'static str, usize)], bias: BiasLayout) -> Self {
        assert!(
            !gates.is_empty(),
            "an RNN-family op packs at least one gate"
        );
        let mut i = 0;
        while i < gates.len() {
            assert!(
                gates[i].1 < gates.len(),
                "ONNX gate index is out of range for this op"
            );
            let mut j = 0;
            while j < i {
                assert!(
                    gates[i].1 != gates[j].1,
                    "two Burn gates claim the same ONNX gate index"
                );
                j += 1;
            }
            i += 1;
        }
        Self { gates, bias }
    }

    /// Burn's per-gate field names paired with the ONNX gate index each holds.
    pub fn gates(&self) -> &'static [(&'static str, usize)] {
        self.gates
    }

    /// How `B` splits across the two `Linear` modules of a gate.
    pub fn bias(&self) -> BiasLayout {
        self.bias
    }

    /// How many gates the op packs into `W`, `R` and `B`.
    ///
    /// True of both the Burn side and the ONNX side, by the permutation invariant
    /// [`new`](Self::new) enforces.
    pub fn count(&self) -> usize {
        self.gates.len()
    }
}

/// The Burn module an RNN-family forward pass runs through.
pub(crate) struct ModuleExpr {
    /// Statements binding the module. Empty when it is a struct field.
    pub setup: TokenStream,
    /// Expression naming the module: `self.gru1`, or the local `gru1`.
    ///
    /// Kept separate from `setup` so that it stays a bare path, cheap to splice more
    /// than once. Folding the two together would make every splice carry the whole
    /// weight-loading block, and a caller that spliced it into two branches would
    /// emit the `let __w = ...;` bindings twice - which `scope.arg` has already
    /// handed out as a move, and which no snapshot test would catch.
    pub expr: TokenStream,
}

/// Whether this node's weights arrive at forward time rather than as initializers.
///
/// `lift_constants` lifts `W`, `R` and `B` together or not at all, so one dynamic
/// weight puts all of them on the runtime path, where each is referenced by name.
pub(crate) fn weights_are_runtime(inputs: &[Argument]) -> bool {
    inputs.iter().skip(1).take(3).any(|arg| arg.is_dynamic())
}

/// The struct field holding this node's module, or `None` when the weights are runtime.
///
/// A field on the runtime path would declare `Param`s that no snapshot can fill, which
/// is what made `Model::from_file` panic before the weights were consumed.
pub(crate) fn field(
    name: &str,
    inputs: &[Argument],
    ty: TokenStream,
    init: TokenStream,
) -> Option<Field> {
    if weights_are_runtime(inputs) {
        return None;
    }

    let ident = Ident::new(name, Span::call_site());
    Some(Field::new(name, ty, quote! { let #ident = #init; }))
}

/// The module the forward pass runs through, plus any statements that build it.
///
/// `init` is only called on the runtime path, where the module is built locally rather
/// than read from a struct field.
pub(crate) fn module(
    name: &str,
    inputs: &[Argument],
    scope: &mut ScopeAtPosition<'_>,
    layout: &GateLayout,
    hidden_size: usize,
    num_directions: usize,
    init: impl FnOnce() -> TokenStream,
) -> ModuleExpr {
    let ident = Ident::new(name, Span::call_site());
    if !weights_are_runtime(inputs) {
        return ModuleExpr {
            setup: quote! {},
            expr: quote! { self.#ident },
        };
    }

    let init = init();
    let load = load_runtime_weights(&ident, scope, inputs, layout, hidden_size, num_directions);
    ModuleExpr {
        setup: quote! {
            let mut #ident = #init;
            #load
        },
        expr: quote! { #ident },
    }
}

/// The axis ONNX's `num_directions` occupies in `Y`.
///
/// `layout=0` gives `[seq, dirs, batch, hidden]`, `layout=1` gives
/// `[batch, seq, dirs, hidden]`.
pub(crate) fn y_direction_axis(batch_first: bool) -> usize {
    if batch_first { 2 } else { 1 }
}

/// The axis ONNX's `num_directions` occupies in `Y_h` and `Y_c`.
///
/// `layout=0` gives `[dirs, batch, hidden]`, `layout=1` gives `[batch, dirs, hidden]`.
pub(crate) fn state_direction_axis(batch_first: bool) -> usize {
    if batch_first { 1 } else { 0 }
}

/// Emit the statements that load runtime `W`/`R`/`B` into `module`'s gate parameters.
///
/// `module` must already be bound as `mut`. Only the parameters the Burn config
/// actually created are assigned: without a `B` input every `Linear` holds
/// `bias: None`, and the bias arms are skipped.
fn load_runtime_weights(
    module: &Ident,
    scope: &mut ScopeAtPosition<'_>,
    inputs: &[Argument],
    layout: &GateLayout,
    hidden_size: usize,
    num_directions: usize,
) -> TokenStream {
    // `to_static` clears an argument's name, so a lifted sibling here would reach
    // `Ident::new("")` and panic inside proc-macro2 with nothing to point at.
    // `lift_all_or_none` keeps the group uniform and `validate_uniform_group` rejects a
    // model that arrives mixed, so this is a backstop rather than a reachable path.
    for (index, arg) in inputs.iter().enumerate().skip(1).take(3) {
        assert!(
            !arg.is_static(),
            "weight input #{index} was lifted to a static initializer while a sibling \
             weight is a runtime graph input; the group must be lifted together or not \
             at all"
        );
    }

    let w = scope.arg(&inputs[1]);
    let r = scope.arg(&inputs[2]);
    let b = inputs
        .get(3)
        .filter(|arg| !arg.is_optional())
        .map(|arg| scope.arg(arg));

    let gate_count = layout.count();
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
        // Each packed tensor is read once per direction, so the last direction moves it.
        let reuse = (direction + 1 < num_directions).then(|| quote! { .clone() });
        let zero_bias = (has_bias && matches!(layout.bias(), BiasLayout::Merged)).then(|| {
            let hidden = hidden_size.to_tokens();
            // Every gate's zeroed hidden bias has the same shape, so build one.
            quote! { let __b_zero = __b_dir.clone().slice_dim(0, 0..#hidden).zeros_like(); }
        });
        let bias_dir = has_bias.then(|| {
            quote! { let __b_dir = __b #reuse.select_dim::<1>(0, #index); }
        });

        let mut gates = quote! {};
        for (gate, onnx_gate) in layout.gates().iter().copied() {
            let gate = Ident::new(gate, Span::call_site());
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

            let rb_start = ((gate_count + onnx_gate) * hidden_size).to_tokens();
            let rb_end = ((gate_count + onnx_gate + 1) * hidden_size).to_tokens();

            gates.extend(match layout.bias() {
                BiasLayout::Split => quote! {
                    #gate_owner.#gate.input_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #start..#end),
                    ));
                    #gate_owner.#gate.hidden_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #rb_start..#rb_end),
                    ));
                },
                BiasLayout::Merged => quote! {
                    #gate_owner.#gate.input_transform.bias = Some(burn::module::Param::from_tensor(
                        __b_dir.clone().slice_dim(0, #start..#end)
                            + __b_dir.clone().slice_dim(0, #rb_start..#rb_end),
                    ));
                    #gate_owner.#gate.hidden_transform.bias =
                        Some(burn::module::Param::from_tensor(__b_zero.clone()));
                },
            });
        }

        directions.extend(quote! {
            {
                let __w_dir = __w #reuse.select_dim::<2>(0, #index);
                let __r_dir = __r #reuse.select_dim::<2>(0, #index);
                #bias_dir
                #zero_bias
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

/// Mark an RNN-family node's `W`/`R`/`B` as lifted initializers.
///
/// `Argument::new` defaults to `ValueSource::Dynamic`, which is the state of a weight
/// supplied as a graph input, so tests covering the static path have to say so. The
/// fabricated `DataId` resolves in no store: these tests read `field`/`forward`, never
/// `value()`. A test that calls `collect_snapshots` on one of these nodes will panic on
/// the unresolvable weight - back the argument with a real store instead of reaching for
/// this helper there.
#[cfg(test)]
pub(crate) fn weights_as_initializers(inputs: &mut [Argument]) {
    for arg in inputs.iter_mut().skip(1).take(3) {
        arg.value_source = onnx_ir::ir::ValueSource::Static(0);
    }
}

/// Mark an RNN-family node's `W`/`R`/`B` as graph inputs, which is how every RNN test
/// in the upstream ONNX suite supplies them.
#[cfg(test)]
pub(crate) fn weights_as_graph_inputs(inputs: &mut [Argument]) {
    for arg in inputs.iter_mut().skip(1).take(3) {
        arg.value_source = onnx_ir::ir::ValueSource::Dynamic;
    }
}
