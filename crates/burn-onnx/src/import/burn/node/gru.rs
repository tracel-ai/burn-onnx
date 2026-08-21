//! ONNX GRU node import implementation.
//!
//! ## Supported ONNX Features
//!
//! - Forward, reverse, and bidirectional directions
//! - Batch-first and sequence-first layouts (`layout` attribute)
//! - Initial hidden state
//! - `linear_before_reset` attribute (maps to Burn's `reset_after`)
//!
//! ## Unsupported ONNX Features
//!
//! - **Variable sequence lengths**: ONNX input `sequence_lens` with shape `[batch_size]` specifies
//!   the actual length of each sequence in a batch. Currently, all sequences in a batch must have
//!   the same length.
//!
//! - **Cell state clipping**: The `clip` attribute is not supported by Burn's GRU module.
//!
//! - **Custom activations**: Burn's GRU uses fixed Sigmoid (gates) and Tanh (hidden) activations.

use super::prelude::*;
use super::rnn_common::{
    self, BiasLayout, GateLayout, ModuleExpr, state_direction_axis, y_direction_axis,
};
use burn_store::TensorSnapshot;
use onnx_ir::gru::{GruActivationFunction, GruDirection};

/// Collect tensor snapshots for GRU burnpack serialization.
///
/// ONNX GRU weight layout:
/// - W: `[num_directions, 3*hidden_size, input_size]` - gates ordered as [z, r, h]
/// - R: `[num_directions, 3*hidden_size, hidden_size]` - gates ordered as [z, r, h]
/// - B: `[num_directions, 6*hidden_size]` - Wb[z,r,h] then Rb[z,r,h]
///
/// Burn GRU structure (per direction):
/// - update_gate.input_transform: weight `[input_size, hidden_size]`, bias `[hidden_size]`
/// - update_gate.hidden_transform: weight `[hidden_size, hidden_size]`, bias `[hidden_size]`
/// - reset_gate, new_gate: same structure
#[allow(clippy::single_range_in_vec_init)]
fn collect_gru_snapshots(
    field_name: &str,
    inputs: &[Argument],
    config: &onnx_ir::gru::GruConfig,
) -> Vec<TensorSnapshot> {
    use crate::burn::node_traits::extract_node_data;
    use burn::tensor::Tensor;

    let hidden_size = config.hidden_size;
    let input_size = config.input_size;

    let data_w = extract_node_data(inputs, 1);
    let data_r = extract_node_data(inputs, 2);
    let data_b = extract_node_data(inputs, 3);

    // `field()` emitted a module, so every weight was supposed to resolve.
    // `validate_uniform_group` in onnx-ir rejects the model-shaped ways this can fail (a
    // missing W/R, or a group split across initializers and graph inputs), so reaching
    // here means the tensor store lost a value we were promised. Returning an empty list
    // instead would rebuild the bug this path exists to fix: a struct full of gate
    // `Param`s that no snapshot fills, which `from_file` reports as missing tensors and
    // `Model::new` silently fills with random values.
    let (Some(data_w), Some(data_r)) = (data_w, data_r) else {
        panic!(
            "GRU '{field_name}': W/R are build-time weights but their data did not resolve. \
             The generated model would load with uninitialized gate weights."
        );
    };
    assert_eq!(
        data_b.is_some(),
        config.has_bias,
        "GRU '{field_name}': config says has_bias={} but B data resolved to {}",
        config.has_bias,
        data_b.is_some()
    );

    let dtype = data_w.dtype;
    let device = Default::default();

    let gate_count = GATE_LAYOUT.count();

    let direction_prefixes: Vec<&str> = match config.direction {
        GruDirection::Forward | GruDirection::Reverse => vec![""],
        GruDirection::Bidirectional => vec!["forward.", "reverse."],
    };

    let mut snapshots = Vec::new();

    // Create tensors from data, pinning the runtime dtype to the ONNX weight
    // dtype via `(device, dtype)`. A bare `&device` would let
    // `Tensor::from_data` resolve the dtype from the device default and
    // silently truncate f64 weights before they enter the snapshot pipeline.
    let w_tensor: Tensor<3> = Tensor::from_data(data_w.clone(), (&device, dtype));
    let r_tensor: Tensor<3> = Tensor::from_data(data_r.clone(), (&device, dtype));
    let b_tensor: Option<Tensor<2>> = data_b
        .clone()
        .map(|b| Tensor::from_data(b, (&device, dtype)));

    for (dir_idx, dir_prefix) in direction_prefixes.iter().enumerate() {
        // W shape: [num_directions, gates*hidden_size, input_size]
        let w_dir = w_tensor
            .clone()
            .slice([
                dir_idx..dir_idx + 1,
                0..gate_count * hidden_size,
                0..input_size,
            ])
            .squeeze::<2>(); // [gates*hidden_size, input_size]

        // R shape: [num_directions, gates*hidden_size, hidden_size]
        let r_dir = r_tensor
            .clone()
            .slice([
                dir_idx..dir_idx + 1,
                0..gate_count * hidden_size,
                0..hidden_size,
            ])
            .squeeze::<2>(); // [gates*hidden_size, hidden_size]

        // B shape: [num_directions, 2*gates*hidden_size]
        let b_dir = b_tensor.as_ref().map(|b| {
            b.clone()
                .slice([dir_idx..dir_idx + 1, 0..2 * gate_count * hidden_size])
                .squeeze::<1>() // [2*gates*hidden_size]
        });

        for (gate_name, onnx_gate_idx) in GATE_LAYOUT.gates().iter().copied() {
            let start = onnx_gate_idx * hidden_size;
            let end = start + hidden_size;

            // Input transform weight: ONNX [hidden_size, input_size] -> Burn [input_size, hidden_size]
            let w_gate = w_dir.clone().slice([start..end, 0..input_size]).transpose();
            let w_gate_data = w_gate.into_data();

            let path = format!(
                "{}.{}{}.input_transform.weight",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(
                w_gate_data,
                &path,
                "Linear",
                dtype,
            ));

            // Input transform bias: Wb for this gate
            if let Some(ref b) = b_dir {
                let wb_start = onnx_gate_idx * hidden_size;
                let wb_end = wb_start + hidden_size;

                let wb: Tensor<1> = b.clone().slice([wb_start..wb_end]);
                let bias_data = wb.into_data();

                let path = format!(
                    "{}.{}{}.input_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
            }

            // Hidden transform weight: ONNX [hidden_size, hidden_size] -> Burn [hidden_size, hidden_size]
            let r_gate = r_dir
                .clone()
                .slice([start..end, 0..hidden_size])
                .transpose();
            let r_gate_data = r_gate.into_data();

            let path = format!(
                "{}.{}{}.hidden_transform.weight",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(
                r_gate_data,
                &path,
                "Linear",
                dtype,
            ));

            // Hidden transform bias: Rb for this gate
            if let Some(b) = &b_dir {
                let rb_start = (gate_count + onnx_gate_idx) * hidden_size;
                let rb_end = rb_start + hidden_size;

                let rb: Tensor<1> = b.clone().slice([rb_start..rb_end]);
                let bias_data = rb.into_data();

                let path = format!(
                    "{}.{}{}.hidden_transform.bias",
                    field_name, dir_prefix, gate_name
                );
                snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
            }
        }
    }

    snapshots
}

/// Create a TensorSnapshot from TensorData.
fn create_snapshot_from_data(
    data: burn::tensor::TensorData,
    path: &str,
    container_type: &str,
    dtype: burn::tensor::DType,
) -> TensorSnapshot {
    use burn::module::ParamId;
    use burn_store::TensorSnapshotError;

    let data = data.convert_dtype(dtype);

    let shape = data.shape.clone();
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    let data_fn = TensorSnapshot::data_fn(
        move || -> Result<burn::tensor::TensorData, TensorSnapshotError> { Ok(data.clone()) },
    );

    TensorSnapshot::from_closure(
        data_fn,
        dtype,
        shape,
        path_stack,
        container_stack,
        ParamId::new(),
    )
}

/// Generate forward code for unidirectional (forward/reverse) GRU.
fn forward_unidirectional(
    node: &onnx_ir::gru::GruNode,
    scope: &mut ScopeAtPosition<'_>,
    input: TokenStream,
    module: ModuleExpr,
    output_y: Option<Ident>,
    output_y_h: Option<Ident>,
) -> TokenStream {
    let has_initial_h = node.config.has_initial_h;
    let is_reverse = matches!(node.config.direction, GruDirection::Reverse);
    let batch_first = node.config.batch_first;

    // Build the initial state expression. ONNX initial_h carries the direction axis in
    // the same place as Y_h, so layout=1 puts it at 1 rather than 0. Burn wants
    // [batch_size, hidden_size].
    let initial_state_expr = if has_initial_h {
        let h_input = scope.arg(&node.inputs[5]);
        let axis = state_direction_axis(batch_first).to_tokens();
        quote! { Some(#h_input.squeeze_dim(#axis)) }
    } else {
        quote! { None }
    };

    // Burn GRU expects [batch_size, seq_length, input_size] (always batch-first)
    let input_transform = if batch_first {
        quote! { #input }
    } else {
        quote! { #input.swap_dims(0, 1) }
    };

    // For reverse: flip the sequence dimension (dim 1 in batch-first layout)
    let input_with_direction = if is_reverse {
        quote! {
            {
                let batch_first_input = #input_transform;
                batch_first_input.flip([1])
            }
        }
    } else {
        quote! { #input_transform }
    };

    let ModuleExpr { setup, expr } = module;
    let forward_call = quote! {
        #setup
        let gru_output = #expr.forward(#input_with_direction, #initial_state_expr);
    };

    // For reverse: flip output back
    let output_with_direction = if is_reverse {
        quote! { gru_output.flip([1]) }
    } else {
        quote! { gru_output }
    };

    // Extract Y_h (final hidden state) from the sequence output.
    let y_h_step = if is_reverse {
        quote! { 0..1 }
    } else {
        quote! { (seq_len - 1)..seq_len }
    };
    let y_h_axis = state_direction_axis(batch_first).to_tokens();
    let y_h_expr = quote! {
        {
            let [_batch, seq_len, _hidden] = batch_first_output.dims();
            let step = batch_first_output.clone().slice([0.._batch, #y_h_step, 0.._hidden]);
            step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[#y_h_axis])
        }
    };

    let y_axis = y_direction_axis(batch_first).to_tokens();
    // Burn's unidirectional Gru is always batch-first, so layout=0 also needs the
    // sequence and batch axes swapped back.
    let y_output_expr = if batch_first {
        quote! { batch_first_output.clone().unsqueeze_dims::<4>(&[#y_axis]) }
    } else {
        quote! { batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[#y_axis]) }
    };

    match (output_y, output_y_h) {
        (Some(y), Some(y_h)) => {
            quote! {
                let (#y, #y_h) = {
                    #forward_call
                    let batch_first_output = #output_with_direction;
                    (
                        #y_output_expr,
                        #y_h_expr
                    )
                };
            }
        }
        (Some(y), None) => {
            quote! {
                let #y = {
                    #forward_call
                    let batch_first_output = #output_with_direction;
                    #y_output_expr
                };
            }
        }
        (None, Some(y_h)) => {
            quote! {
                let #y_h = {
                    #forward_call
                    let batch_first_output = #output_with_direction;
                    #y_h_expr
                };
            }
        }
        (None, None) => {
            quote! {
                {
                    #forward_call
                }
            }
        }
    }
}

/// Generate forward code for bidirectional GRU.
///
/// BiGru.forward() returns (output_seq, final_state):
///   output_seq: [batch, seq, 2*hidden] (batch_first) or [seq, batch, 2*hidden]
///   final_state: [2, batch, hidden] (already matches ONNX Y_h format)
fn forward_bidirectional(
    node: &onnx_ir::gru::GruNode,
    scope: &mut ScopeAtPosition<'_>,
    input: TokenStream,
    module: ModuleExpr,
    output_y: Option<Ident>,
    output_y_h: Option<Ident>,
) -> TokenStream {
    let has_initial_h = node.config.has_initial_h;
    let hidden_size = node.config.hidden_size;

    // BiGru wants [2, batch, hidden], which is ONNX layout=0. layout=1 hands it
    // [batch, 2, hidden].
    let initial_state_expr = if has_initial_h {
        let h_input = scope.arg(&node.inputs[5]);
        if node.config.batch_first {
            quote! { Some(#h_input.swap_dims(0, 1)) }
        } else {
            quote! { Some(#h_input) }
        }
    } else {
        quote! { None }
    };

    // BiGru's final state is [num_directions, batch, hidden], which is ONNX layout=0.
    let y_h_expr = if node.config.batch_first {
        quote! { final_state.swap_dims(0, 1) }
    } else {
        quote! { final_state }
    };

    // Y output transformation: split concatenated hidden states
    let y_output_expr = if node.config.batch_first {
        // Burn: [batch, seq, 2*hidden] -> reshape [batch, seq, 2, hidden] (ONNX layout=1)
        quote! {
            {
                let [batch_size, seq_len, _] = output_seq.dims();
                output_seq.reshape([batch_size, seq_len, 2, #hidden_size])
            }
        }
    } else {
        // Burn: [seq, batch, 2*hidden] -> reshape [seq, batch, 2, hidden]
        // -> swap_dims(1, 2) -> [seq, 2, batch, hidden] (ONNX layout=0)
        quote! {
            {
                let [seq_len, batch_size, _] = output_seq.dims();
                let reshaped = output_seq.reshape([seq_len, batch_size, 2, #hidden_size]);
                reshaped.swap_dims(1, 2)
            }
        }
    };

    let ModuleExpr { setup, expr } = module;

    // Vary the destructuring to avoid unused-variable warnings in generated code
    match (output_y, output_y_h) {
        (Some(y), Some(y_h)) => {
            quote! {
                let (#y, #y_h) = {
                    #setup
                    let (output_seq, final_state) = #expr.forward(#input, #initial_state_expr);
                    (#y_output_expr, #y_h_expr)
                };
            }
        }
        (Some(y), None) => {
            quote! {
                let #y = {
                    #setup
                    let (output_seq, _final_state) = #expr.forward(#input, #initial_state_expr);
                    #y_output_expr
                };
            }
        }
        (None, Some(y_h)) => {
            quote! {
                let #y_h = {
                    #setup
                    let (_output_seq, final_state) = #expr.forward(#input, #initial_state_expr);
                    #y_h_expr
                };
            }
        }
        (None, None) => {
            quote! {
                {
                    #setup
                    let _ = #expr.forward(#input, #initial_state_expr);
                }
            }
        }
    }
}

/// ONNX GRU packs its gates as [z, r, h], which is Burn's own update/reset/new order.
const GATE_LAYOUT: GateLayout = GateLayout::new(
    &[("update_gate", 0), ("reset_gate", 1), ("new_gate", 2)],
    BiasLayout::Split,
);

/// The module's type, and the expression that builds it on `device`.
///
/// `device` is the only thing that differs between the two paths: `device` names the
/// parameter of the generated `new()`, `&self.device` the field read inside `forward()`.
fn module_parts(node: &onnx_ir::gru::GruNode, device: TokenStream) -> (TokenStream, TokenStream) {
    if node.config.clip.is_some() {
        panic!(
            "GRU clip attribute is not supported. Burn's GRU module does not support cell state clipping."
        );
    }
    if node.config.gate_activation != GruActivationFunction::Sigmoid
        || node.config.hidden_activation != GruActivationFunction::Tanh
    {
        panic!(
            "Custom GRU activations are not supported. Burn's GRU uses fixed Sigmoid (gates) and Tanh (hidden). Got gate: {:?}, hidden: {:?}",
            node.config.gate_activation, node.config.hidden_activation
        );
    }

    let d_input = node.config.input_size.to_tokens();
    let d_hidden = node.config.hidden_size.to_tokens();
    let bias = node.config.has_bias;
    // ONNX linear_before_reset maps to Burn reset_after
    let reset_after = node.config.linear_before_reset;

    match node.config.direction {
        GruDirection::Forward | GruDirection::Reverse => (
            quote! { burn::nn::gru::Gru },
            quote! {
                burn::nn::gru::GruConfig::new(#d_input, #d_hidden, #bias)
                    .with_reset_after(#reset_after)
                    .init(#device)
            },
        ),
        GruDirection::Bidirectional => {
            let batch_first = node.config.batch_first;
            (
                quote! { burn::nn::gru::BiGru },
                quote! {
                    burn::nn::gru::BiGruConfig::new(#d_input, #d_hidden, #bias)
                        .with_reset_after(#reset_after)
                        .with_batch_first(#batch_first)
                        .init(#device)
                },
            )
        }
    }
}

impl NodeCodegen for onnx_ir::gru::GruNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let (ty, init) = module_parts(self, quote! { device });
        rnn_common::field(&self.name, &self.inputs, ty, init)
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        collect_gru_snapshots(field_name, &self.inputs, &self.config)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let module = rnn_common::module(
            &self.name,
            &self.inputs,
            scope,
            &GATE_LAYOUT,
            self.config.hidden_size,
            self.config.direction.num_directions(),
            || module_parts(self, quote! { &self.device }).1,
        );

        let output_y = self
            .outputs
            .first()
            .filter(|a| !a.name.is_empty())
            .map(arg_to_ident);
        let output_y_h = self
            .outputs
            .get(1)
            .filter(|a| !a.name.is_empty())
            .map(arg_to_ident);

        if matches!(self.config.direction, GruDirection::Bidirectional) {
            forward_bidirectional(self, scope, input, module, output_y, output_y_h)
        } else {
            forward_unidirectional(self, scope, input, module, output_y, output_y_h)
        }
    }

    fn register_imports(&self, _imports: &mut BurnImports) {
        // GRU types are accessed via full path in field(), so no extra imports needed
    }
}

#[cfg(test)]
mod tests {
    use super::super::rnn_common::{weights_as_graph_inputs, weights_as_initializers};
    use super::super::test_helpers::*;
    use crate::burn::node::NodeCodegen;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::gru::{GruActivationFunction, GruConfig, GruDirection, GruNode};
    use onnx_ir::ir::{ArgType, Argument, TensorType};

    fn create_gru_node(
        name: &str,
        direction: GruDirection,
        batch_first: bool,
        has_initial_h: bool,
        num_outputs: usize,
    ) -> GruNode {
        let config = GruConfig::new(
            4, // input_size
            8, // hidden_size
            direction,
            true, // has_bias
            has_initial_h,
            batch_first,
            None,                           // clip
            false,                          // linear_before_reset
            GruActivationFunction::Sigmoid, // gate_activation
            GruActivationFunction::Tanh,    // hidden_activation
            None,                           // activation_alpha
            None,                           // activation_beta
        );

        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let w = Argument::new("W", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let r = Argument::new("R", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let b = Argument::new("B", ArgType::Tensor(TensorType::new(DType::F32, 2, None)));

        let mut inputs = vec![input, w, r, b];
        weights_as_initializers(&mut inputs);

        if has_initial_h {
            // sequence_lens (unused optional placeholder)
            inputs.push(Argument::new(
                "sequence_lens",
                ArgType::ScalarNative(DType::I64),
            ));
            // initial_h
            inputs.push(Argument::new(
                "initial_h",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }

        let mut outputs = vec![];
        if num_outputs > 0 {
            outputs.push(Argument::new(
                "Y",
                ArgType::Tensor(TensorType::new(DType::F32, 4, None)),
            ));
        }
        if num_outputs > 1 {
            outputs.push(Argument::new(
                "Y_h",
                ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
            ));
        }

        GruNode {
            name: name.to_string(),
            inputs,
            outputs,
            config,
        }
    }

    #[test]
    fn test_gru_forward_basic() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let gru_output = self.gru1.forward(input.swap_dims(0, 1), None);
                let batch_first_output = gru_output;
                (
                    batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, (seq_len - 1)..seq_len, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_reverse() {
        let node = create_gru_node("gru1", GruDirection::Reverse, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let gru_output = self
                    .gru1
                    .forward(
                        {
                            let batch_first_input = input.swap_dims(0, 1);
                            batch_first_input.flip([1])
                        },
                        None,
                    );
                let batch_first_output = gru_output.flip([1]);
                (
                    batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, 0..1, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_y_only() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let Y = {
                let gru_output = self.gru1.forward(input.swap_dims(0, 1), None);
                let batch_first_output = gru_output;
                batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1])
            };
            Y
        }
        ");
    }

    #[test]
    fn test_gru_field_forward() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, false, 2);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let gru1 = burn::nn::gru::GruConfig::new(4, 8, true)
            .with_reset_after(false)
            .init(device);
        ");
    }

    #[test]
    fn test_gru_field_reverse() {
        let node = create_gru_node("gru1", GruDirection::Reverse, false, false, 2);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let gru1 = burn::nn::gru::GruConfig::new(4, 8, true)
            .with_reset_after(false)
            .init(device);
        ");
    }

    // Note: Y_h-only output branch (None, Some(y_h)) cannot be tested via codegen_forward_default
    // because the test helper panics on empty-named outputs. This branch is covered by integration
    // tests and by the forward() logic which filters empty names via .filter(|a| !a.name.is_empty()).

    #[test]
    fn test_gru_forward_batch_first() {
        let node = create_gru_node("gru1", GruDirection::Forward, true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let gru_output = self.gru1.forward(input, None);
                let batch_first_output = gru_output;
                (
                    batch_first_output.clone().unsqueeze_dims::<4>(&[2]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, (seq_len - 1)..seq_len, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[1])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_with_initial_h() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            sequence_lens: i64,
            initial_h: Tensor<3>,
        ) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let gru_output = self
                    .gru1
                    .forward(input.swap_dims(0, 1), Some(initial_h.squeeze_dim(0)));
                let batch_first_output = gru_output;
                (
                    batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, (seq_len - 1)..seq_len, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_bidirectional() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.gru1.forward(input, None);
                (
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                        reshaped.swap_dims(1, 2)
                    },
                    final_state,
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_bidirectional_batch_first() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, true, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.gru1.forward(input, None);
                (
                    {
                        let [batch_size, seq_len, _] = output_seq.dims();
                        output_seq.reshape([batch_size, seq_len, 2, 8usize])
                    },
                    final_state.swap_dims(0, 1),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_bidirectional_y_only() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> Tensor<4> {
            let Y = {
                let (output_seq, _final_state) = self.gru1.forward(input, None);
                {
                    let [seq_len, batch_size, _] = output_seq.dims();
                    let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                    reshaped.swap_dims(1, 2)
                }
            };
            Y
        }
        ");
    }

    #[test]
    fn test_gru_forward_bidirectional_with_initial_h() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            sequence_lens: i64,
            initial_h: Tensor<3>,
        ) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.gru1.forward(input, Some(initial_h));
                (
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                        reshaped.swap_dims(1, 2)
                    },
                    final_state,
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_field_bidirectional() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, false, false, 2);
        let code = codegen_field_init(&node);
        assert_snapshot!(code, @r"
        let gru1 = burn::nn::gru::BiGruConfig::new(4, 8, true)
            .with_reset_after(false)
            .with_batch_first(false)
            .init(device);
        ");
    }

    #[test]
    fn test_gru_field_runtime_weights() {
        let mut node = create_gru_node("gru1", GruDirection::Forward, false, false, 2);
        weights_as_graph_inputs(&mut node.inputs);
        assert!(
            NodeCodegen::field(&node).is_none(),
            "runtime weights must not declare a struct field, or `from_file` fails on \
             tensors no snapshot can supply"
        );
    }

    #[test]
    fn test_gru_forward_runtime_weights() {
        let mut node = create_gru_node("gru1", GruDirection::Forward, false, false, 2);
        weights_as_graph_inputs(&mut node.inputs);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            W: Tensor<3>,
            R: Tensor<3>,
            B: Tensor<2>,
        ) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let mut gru1 = burn::nn::gru::GruConfig::new(4, 8, true)
                    .with_reset_after(false)
                    .init(&self.device);
                let __w = W;
                let __r = R;
                let __b = B;
                {
                    let __w_dir = __w.select_dim::<2>(0, 0);
                    let __r_dir = __r.select_dim::<2>(0, 0);
                    let __b_dir = __b.select_dim::<1>(0, 0);
                    gru1
                        .update_gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    gru1
                        .update_gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    gru1
                        .update_gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 0..8)),
                    );
                    gru1
                        .update_gate
                        .hidden_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 24..32)),
                    );
                    gru1
                        .reset_gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 8..16).transpose(),
                    );
                    gru1
                        .reset_gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 8..16).transpose(),
                    );
                    gru1
                        .reset_gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 8..16)),
                    );
                    gru1
                        .reset_gate
                        .hidden_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 32..40)),
                    );
                    gru1
                        .new_gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 16..24).transpose(),
                    );
                    gru1
                        .new_gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 16..24).transpose(),
                    );
                    gru1
                        .new_gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 16..24)),
                    );
                    gru1
                        .new_gate
                        .hidden_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(__b_dir.clone().slice_dim(0, 40..48)),
                    );
                }
                let gru_output = gru1.forward(input.swap_dims(0, 1), None);
                let batch_first_output = gru_output;
                (
                    batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, (seq_len - 1)..seq_len, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    /// ONNX layout=1 moves the direction axis of `initial_h` too, not just of the outputs.
    #[test]
    fn test_gru_forward_batch_first_initial_h() {
        let node = create_gru_node("gru1", GruDirection::Forward, true, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            sequence_lens: i64,
            initial_h: Tensor<3>,
        ) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let gru_output = self.gru1.forward(input, Some(initial_h.squeeze_dim(1)));
                let batch_first_output = gru_output;
                (
                    batch_first_output.clone().unsqueeze_dims::<4>(&[2]),
                    {
                        let [_batch, seq_len, _hidden] = batch_first_output.dims();
                        let step = batch_first_output
                            .clone()
                            .slice([0.._batch, (seq_len - 1)..seq_len, 0.._hidden]);
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[1])
                    },
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_gru_forward_bidirectional_batch_first_initial_h() {
        let node = create_gru_node("gru1", GruDirection::Bidirectional, true, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<3>,
            sequence_lens: i64,
            initial_h: Tensor<3>,
        ) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self
                    .gru1
                    .forward(input, Some(initial_h.swap_dims(0, 1)));
                (
                    {
                        let [batch_size, seq_len, _] = output_seq.dims();
                        output_seq.reshape([batch_size, seq_len, 2, 8usize])
                    },
                    final_state.swap_dims(0, 1),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    /// A weight group that is only partly constant stays unlifted, so the whole group
    /// takes the runtime path rather than half of it being dropped.
    #[test]
    fn test_gru_field_partial_runtime_weights() {
        let mut node = create_gru_node("gru1", GruDirection::Forward, false, false, 2);
        // W and R are constants that `lift_all_or_none` declined to lift; B is a graph input.
        node.inputs[3].value_source = onnx_ir::ir::ValueSource::Dynamic;
        assert!(
            NodeCodegen::field(&node).is_none(),
            "a runtime B must put the whole group on the runtime path, or its bias is dropped"
        );
    }
}
