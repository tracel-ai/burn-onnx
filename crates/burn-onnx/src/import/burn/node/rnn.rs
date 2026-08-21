//! ONNX Rnn node import implementation.
//!
//! ## Supported ONNX Features
//!
//! - Forward, reverse, and bidirectional directions
//! - Batch-first and sequence-first layouts (`layout` attribute)
//! - Initial hidden state
//! - Custom activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus
//! - Cell state clipping (`clip` attribute)
//!
//! ## Unsupported ONNX Features
//!
//! - **Variable sequence lengths**: ONNX input `sequence_lens` with shape `[batch_size]` specifies
//!   the actual length of each sequence in a batch. Currently, all sequences in a batch must have
//!   the same length.

use super::prelude::*;
use super::rnn_common::{
    self, BiasLayout, GateLayout, ModuleExpr, state_direction_axis, weights_are_runtime,
    y_direction_axis,
};
use burn::nn::activation::ActivationConfig;
use burn_store::TensorSnapshot;
use onnx_ir::rnn::{RnnActivationFunction, RnnDirection};

/// Convert ONNX activation function to Burn ActivationConfig.
///
/// # Panics
///
/// Panics if the ONNX activation function is not supported by burn-nn.
/// Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus.
fn to_burn_activation(onnx_activation: RnnActivationFunction) -> ActivationConfig {
    match onnx_activation {
        RnnActivationFunction::Sigmoid => ActivationConfig::Sigmoid,
        RnnActivationFunction::Tanh => ActivationConfig::Tanh,
        RnnActivationFunction::Relu => ActivationConfig::Relu,
        RnnActivationFunction::HardSigmoid => {
            ActivationConfig::HardSigmoid(burn::nn::HardSigmoidConfig::new())
        }
        RnnActivationFunction::LeakyRelu => {
            ActivationConfig::LeakyRelu(burn::nn::LeakyReluConfig::new())
        }
        RnnActivationFunction::Softplus => {
            ActivationConfig::Softplus(burn::nn::SoftplusConfig::new())
        }
        unsupported => panic!(
            "RNN activation '{:?}' is not supported by burn-nn. \
             Supported activations: Sigmoid, Tanh, Relu, HardSigmoid, LeakyRelu, Softplus. \
             Consider using a supported activation or implementing support in burn-nn.",
            unsupported
        ),
    }
}

/// Collect tensor snapshots for Rnn burnpack serialization.
///
/// This function handles the weight transformation from ONNX's packed RNN format
/// to Burn's Rnn weight structure using the Flex CPU backend for tensor ops.
///
/// ONNX Rnn weight layout:
/// - W: `[num_directions, hidden_size, input_size]`
/// - R: `[num_directions, hidden_size, hidden_size]`
/// - B: `[num_directions, 2*hidden_size]` - Wb then Rb
///
/// Burn Rnn structure (per direction):
/// - input_transform: weight `[input_size, hidden_size]`, bias `[hidden_size]`
/// - hidden_transform: weight `[hidden_size, hidden_size]`, bias `[hidden_size]`
#[allow(clippy::single_range_in_vec_init)]
fn collect_rnn_snapshots(
    field_name: &str,
    inputs: &[Argument],
    config: &onnx_ir::rnn::RnnConfig,
) -> Vec<TensorSnapshot> {
    use crate::burn::node_traits::extract_node_data;
    use burn::tensor::Tensor;

    let hidden_size = config.hidden_size;
    let input_size = config.input_size;

    // Extract weight tensors from inputs
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
            "RNN '{field_name}': W/R are build-time weights but their data did not resolve. \
             The generated model would load with uninitialized gate weights."
        );
    };
    assert_eq!(
        data_b.is_some(),
        config.has_bias,
        "RNN '{field_name}': config says has_bias={} but B data resolved to {}",
        config.has_bias,
        data_b.is_some()
    );

    let dtype = data_w.dtype;
    let device = Default::default();

    let gate_count = GATE_LAYOUT.count();

    // Determine direction prefixes based on Rnn type
    let direction_prefixes: Vec<&str> = match config.direction {
        RnnDirection::Forward | RnnDirection::Reverse => vec![""],
        RnnDirection::Bidirectional => vec!["forward.", "reverse."],
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
        // Select direction slice from W and R
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

        let (gate_name, onnx_gate_idx) = GATE_LAYOUT.gates()[0];
        let start = onnx_gate_idx * hidden_size;
        let end = start + hidden_size;

        // Input transform weight: slice from W and transpose
        // ONNX: [hidden_size, input_size] -> Burn: [input_size, hidden_size]
        let w_gate = w_dir.clone().slice([start..end, 0..input_size]).transpose(); // [input_size, hidden_size]
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

        // Input transform bias: Wb + Rb for this gate
        if let Some(ref b) = b_dir {
            let wb_start = onnx_gate_idx * hidden_size;
            let wb_end = wb_start + hidden_size;
            let rb_start = (gate_count + onnx_gate_idx) * hidden_size;
            let rb_end = rb_start + hidden_size;

            let wb: Tensor<1> = b.clone().slice([wb_start..wb_end]);
            let rb: Tensor<1> = b.clone().slice([rb_start..rb_end]);
            let bias = wb.add(rb);
            let bias_data = bias.into_data();

            let path = format!(
                "{}.{}{}.input_transform.bias",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(bias_data, &path, "Linear", dtype));
        }

        // Hidden transform weight: slice from R and transpose
        // ONNX: [hidden_size, hidden_size] -> Burn: [hidden_size, hidden_size]
        let r_gate = r_dir
            .clone()
            .slice([start..end, 0..hidden_size])
            .transpose(); // [hidden_size, hidden_size]
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

        // Hidden transform bias: zeros (combined bias is in input_transform)
        if b_dir.is_some() {
            let zeros: Tensor<1> = Tensor::zeros([hidden_size], (&device, dtype));
            let zeros_data = zeros.into_data();

            let path = format!(
                "{}.{}{}.hidden_transform.bias",
                field_name, dir_prefix, gate_name
            );
            snapshots.push(create_snapshot_from_data(
                zeros_data, &path, "Linear", dtype,
            ));
        }
    }

    snapshots
}

/// Create a TensorSnapshot from TensorData.
///
/// Normalizes the data back to the target dtype as a safety net. The upstream
/// weight-slicing pipeline already pins the dtype when building the intermediate
/// `Tensor<_>` (via `from_data(data, (device, dtype))`),
/// so this `convert_dtype` is ordinarily a no-op. It stays in place to guarantee
/// the snapshot's dtype tag matches the tensor data even if a future refactor
/// introduces a path that produces data in a different dtype.
fn create_snapshot_from_data(
    data: burn::tensor::TensorData,
    path: &str,
    container_type: &str,
    dtype: burn::tensor::DType,
) -> TensorSnapshot {
    use burn::module::ParamId;
    use burn_store::TensorSnapshotError;

    // No-op when `data` already has the declared dtype; defensive otherwise.
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

/// Convert ActivationConfig to tokens for code generation
fn activation_to_tokens(activation: &ActivationConfig) -> TokenStream {
    match activation {
        ActivationConfig::Sigmoid => quote! { ActivationConfig::Sigmoid },
        ActivationConfig::Tanh => quote! { ActivationConfig::Tanh },
        ActivationConfig::Relu => quote! { ActivationConfig::Relu },
        ActivationConfig::HardSigmoid(_) => {
            quote! { ActivationConfig::HardSigmoid(burn::nn::HardSigmoidConfig::new()) }
        }
        ActivationConfig::LeakyRelu(_) => {
            quote! { ActivationConfig::LeakyRelu(burn::nn::LeakyReluConfig::new()) }
        }
        ActivationConfig::Softplus(_) => {
            quote! { ActivationConfig::Softplus(burn::nn::SoftplusConfig::new()) }
        }
        _ => panic!("Unsupported activation config for RNN"),
    }
}

/// RNN has a single gate, so ONNX's packing is already Burn's.
const GATE_LAYOUT: GateLayout = GateLayout::new(&[("gate", 0)], BiasLayout::Merged);

/// The module's type, and the expression that builds it on `device`.
///
/// `device` is the only thing that differs between the two paths: `device` names the
/// parameter of the generated `new()`, `&self.device` the field read inside `forward()`.
fn module_parts(node: &onnx_ir::rnn::RnnNode, device: TokenStream) -> (TokenStream, TokenStream) {
    let d_input = node.config.input_size.to_tokens();
    let d_hidden = node.config.hidden_size.to_tokens();
    let bias = node.config.has_bias;
    let batch_first = node.config.batch_first;

    // Convert activations to tokens
    let hidden_act = to_burn_activation(node.config.hidden_activation);
    let hidden_activation = activation_to_tokens(&hidden_act);

    // Generate clip config if present
    let clip_config = if let Some(clip) = node.config.clip {
        let clip_val = clip as f64;
        quote! { .with_clip(Some(#clip_val)) }
    } else {
        quote! {}
    };

    // Only add non-default activations to config
    let activations_config = {
        let mut tokens = quote! {};
        if !matches!(hidden_act, ActivationConfig::Tanh) {
            tokens = quote! { #tokens .with_hidden_activation(#hidden_activation) };
        }
        tokens
    };

    match node.config.direction {
        RnnDirection::Forward => (
            quote! { Rnn },
            quote! {
                RnnConfig::new(#d_input, #d_hidden, #bias)
                    .with_batch_first(#batch_first)
                    #clip_config
                    #activations_config
                    .init(#device)
            },
        ),
        RnnDirection::Reverse => (
            quote! { Rnn },
            quote! {
                RnnConfig::new(#d_input, #d_hidden, #bias)
                    .with_batch_first(#batch_first)
                    .with_reverse(true)
                    #clip_config
                    #activations_config
                    .init(#device)
            },
        ),
        RnnDirection::Bidirectional => (
            quote! { BiRnn },
            quote! {
                BiRnnConfig::new(#d_input, #d_hidden, #bias)
                    .with_batch_first(#batch_first)
                    #clip_config
                    #activations_config
                    .init(#device)
            },
        ),
    }
}

impl NodeCodegen for onnx_ir::rnn::RnnNode {
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
        collect_rnn_snapshots(field_name, &self.inputs, &self.config)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let ModuleExpr { setup, expr } = rnn_common::module(
            &self.name,
            &self.inputs,
            scope,
            &GATE_LAYOUT,
            self.config.hidden_size,
            self.config.direction.num_directions(),
            || module_parts(self, quote! { &self.device }).1,
        );

        // Get output variable names
        let output_y = self.outputs.first().map(arg_to_ident);
        let output_y_h = self.outputs.get(1).map(arg_to_ident);

        // Handle initial states if provided
        let has_initial_h = self.config.has_initial_h;

        // Get initial state inputs if present
        // Input indices: 0=X, 1=W, 2=R, 3=B, 4=sequence_lens, 5=initial_h
        // ONNX initial states: [num_directions, batch_size, hidden_size]
        // Burn expects: [batch_size, hidden_size] for unidirectional
        // initial_h carries the direction axis in the same place as Y_h, so layout=1 puts
        // it at 1 rather than 0.
        let state_axis = state_direction_axis(self.config.batch_first).to_tokens();
        let initial_state_expr = if has_initial_h {
            let h_input = scope.arg(&self.inputs[5]);
            match self.config.direction {
                RnnDirection::Forward | RnnDirection::Reverse => {
                    // Drop the direction axis: Burn wants [batch_size, hidden_size]
                    quote! { Some(RnnState::new(#h_input.squeeze_dim(#state_axis))) }
                }
                RnnDirection::Bidirectional => {
                    // BiRnn wants [2, batch, hidden]; layout=1 hands it [batch, 2, hidden].
                    if self.config.batch_first {
                        quote! { Some(RnnState::new(#h_input.swap_dims(0, 1))) }
                    } else {
                        quote! { Some(RnnState::new(#h_input)) }
                    }
                }
            }
        } else {
            quote! { None }
        };

        // The Rnn module now handles batch_first and reverse internally via config,
        // so no input/output transformation is needed here
        let forward_call = quote! {
            #setup
            let (output_seq, final_state) = #expr.forward(#input, #initial_state_expr);
        };

        // Transform outputs to ONNX format
        // Burn output shape depends on batch_first config:
        //   batch_first=true:  [batch_size, seq_length, hidden_size] or [batch_size, seq_length, 2*hidden_size] for bidirectional
        //   batch_first=false: [seq_length, batch_size, hidden_size] or [seq_length, batch_size, 2*hidden_size] for bidirectional
        // ONNX Y: [seq, dirs, batch, hidden] under layout=0, [batch, seq, dirs, hidden]
        // under layout=1. Y_h: [dirs, batch, hidden] or [batch, dirs, hidden].

        // For unidirectional Rnn:
        //   - Burn final_state.hidden: [batch_size, hidden_size] (2D)
        //   - Need to unsqueeze to add num_directions dimension
        //   - Burn output: [seq, batch, hidden] -> ONNX Y, direction axis per layout
        // For bidirectional Rnn:
        //   - Burn final_state.hidden: [2, batch_size, hidden_size] (already 3D)
        //   - No unsqueeze needed
        //   - Burn output: [seq, batch, 2*hidden] -> ONNX Y: [seq, 2, batch, hidden]
        //     This requires reshape + transpose
        let is_bidirectional = matches!(self.config.direction, RnnDirection::Bidirectional);
        let hidden_size = self.config.hidden_size;

        let batch_first = self.config.batch_first;
        let state_axis = state_direction_axis(batch_first);
        let hidden_expr = if is_bidirectional {
            // Burn produces [num_directions, batch, hidden]; layout=1 wants batch first.
            if state_axis == 0 {
                quote! { final_state.hidden }
            } else {
                quote! { final_state.hidden.swap_dims(0, 1) }
            }
        } else {
            let axis = state_axis.to_tokens();
            quote! { final_state.hidden.unsqueeze_dims::<3>(&[#axis]) }
        };

        // Y output transformation
        // For unidirectional: unsqueeze at the layout's direction axis
        // For bidirectional: reshape to split the concatenated hidden states, then reorder dims
        //   ONNX layout=0 (batch_first=false): Y is [seq, num_dirs, batch, hidden]
        //   ONNX layout=1 (batch_first=true):  Y is [batch, seq, num_dirs, hidden]
        let y_output_expr = if is_bidirectional {
            if batch_first {
                // Burn output: [batch, seq, 2*hidden]
                // Reshape to: [batch, seq, 2, hidden] - already matches ONNX layout=1
                quote! {
                    {
                        let [batch_size, seq_len, _] = output_seq.dims();
                        output_seq.reshape([batch_size, seq_len, 2, #hidden_size])
                    }
                }
            } else {
                // Burn output: [seq, batch, 2*hidden]
                // Reshape to: [seq, batch, 2, hidden]
                // Then swap dims 1 and 2 to get: [seq, 2, batch, hidden] for ONNX layout=0
                quote! {
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, #hidden_size]);
                        reshaped.swap_dims(1, 2)
                    }
                }
            }
        } else {
            let axis = y_direction_axis(batch_first).to_tokens();
            quote! { output_seq.unsqueeze_dims::<4>(&[#axis]) }
        };

        // Build output assignments based on which outputs are used
        // Use block scoping to contain temporary variables
        match (output_y, output_y_h) {
            (Some(y), Some(y_h)) => {
                quote! {
                    let (#y, #y_h) = {
                        #forward_call
                        (
                            #y_output_expr,
                            #hidden_expr
                        )
                    };
                }
            }
            (Some(y), None) => {
                quote! {
                    let #y = {
                        #forward_call
                        #y_output_expr
                    };
                }
            }
            (None, Some(y_h)) => {
                quote! {
                    let #y_h = {
                        #forward_call
                        #hidden_expr
                    };
                }
            }
            _ => {
                // Handle remaining cases - just run the forward pass
                quote! {
                    {
                        #forward_call
                    }
                }
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Check if we need to import ActivationConfig (for non-default activations)
        let hidden_act = to_burn_activation(self.config.hidden_activation);

        let needs_activation_import = !matches!(hidden_act, ActivationConfig::Tanh);

        if needs_activation_import {
            imports.register("burn::nn::ActivationConfig");
        }

        // The module type is only named by the struct field. On the runtime-weight
        // path there is no field, and importing it would warn in generated code.
        let needs_module_type = !weights_are_runtime(&self.inputs);
        // RnnState is only named when an initial state is passed in.
        let needs_state = self.config.has_initial_h;

        match self.config.direction {
            RnnDirection::Forward | RnnDirection::Reverse => {
                if needs_module_type {
                    imports.register("burn::nn::Rnn");
                }
                imports.register("burn::nn::RnnConfig");
                if needs_state {
                    imports.register("burn::nn::RnnState");
                }
            }
            RnnDirection::Bidirectional => {
                if needs_module_type {
                    imports.register("burn::nn::BiRnn");
                }
                imports.register("burn::nn::BiRnnConfig");
                if needs_state {
                    imports.register("burn::nn::RnnState");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::rnn_common::{weights_as_graph_inputs, weights_as_initializers};
    use super::super::test_helpers::*;
    use crate::burn::node::NodeCodegen;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType};
    use onnx_ir::rnn::{RnnActivationFunction, RnnConfig, RnnDirection, RnnNode};

    fn create_rnn_node(
        name: &str,
        direction: RnnDirection,
        batch_first: bool,
        num_outputs: usize,
    ) -> RnnNode {
        let config = RnnConfig::new(
            4, // input_size
            8, // hidden_size
            direction,
            true,  // has_bias
            false, // has_initial_h
            batch_first,
            None,                        // clip
            RnnActivationFunction::Tanh, // hidden_activation
        );

        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        );
        let w = Argument::new("W", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let r = Argument::new("R", ArgType::Tensor(TensorType::new(DType::F32, 3, None)));
        let b = Argument::new("B", ArgType::Tensor(TensorType::new(DType::F32, 2, None)));

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

        if num_outputs > 2 {
            panic!("RnnNode can only have up to 2 outputs (Y and Y_h)");
        }

        let mut inputs = vec![input, w, r, b];
        weights_as_initializers(&mut inputs);

        RnnNode {
            name: name.to_string(),
            inputs,
            outputs,
            config,
        }
    }

    #[test]
    fn test_rnn_forward_basic() {
        let node = create_rnn_node("Rnn1", RnnDirection::Forward, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.Rnn1.forward(input, None);
                (
                    output_seq.unsqueeze_dims::<4>(&[1]),
                    final_state.hidden.unsqueeze_dims::<3>(&[0]),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_rnn_forward_bidirectional() {
        let node = create_rnn_node("Rnn1", RnnDirection::Bidirectional, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.Rnn1.forward(input, None);
                (
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                        reshaped.swap_dims(1, 2)
                    },
                    final_state.hidden,
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_rnn_forward_reverse() {
        let node = create_rnn_node("Rnn1", RnnDirection::Reverse, false, 2);
        let code = codegen_forward_default(&node);
        // Note: reverse is now handled by the Rnn module's config, not by flip() in codegen
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.Rnn1.forward(input, None);
                (
                    output_seq.unsqueeze_dims::<4>(&[1]),
                    final_state.hidden.unsqueeze_dims::<3>(&[0]),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_rnn_field_runtime_weights() {
        let mut node = create_rnn_node("rnn1", RnnDirection::Forward, false, 2);
        weights_as_graph_inputs(&mut node.inputs);
        assert!(
            NodeCodegen::field(&node).is_none(),
            "runtime weights must not declare a struct field, or `from_file` fails on \
             tensors no snapshot can supply"
        );
    }

    #[test]
    fn test_rnn_forward_runtime_weights() {
        let mut node = create_rnn_node("rnn1", RnnDirection::Forward, false, 2);
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
                let mut rnn1 = RnnConfig::new(4, 8, true)
                    .with_batch_first(false)
                    .init(&self.device);
                let __w = W;
                let __r = R;
                let __b = B;
                {
                    let __w_dir = __w.select_dim::<2>(0, 0);
                    let __r_dir = __r.select_dim::<2>(0, 0);
                    let __b_dir = __b.select_dim::<1>(0, 0);
                    let __b_zero = __b_dir.clone().slice_dim(0, 0..8).zeros_like();
                    rnn1
                        .gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(
                            __b_dir.clone().slice_dim(0, 0..8)
                                + __b_dir.clone().slice_dim(0, 8..16),
                        ),
                    );
                    rnn1
                        .gate
                        .hidden_transform
                        .bias = Some(burn::module::Param::from_tensor(__b_zero.clone()));
                }
                let (output_seq, final_state) = rnn1.forward(input, None);
                (
                    output_seq.unsqueeze_dims::<4>(&[1]),
                    final_state.hidden.unsqueeze_dims::<3>(&[0]),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    /// Covers the two-direction branch of the shared weight loader: RNN has one gate,
    /// so the same coverage costs a fraction of GRU's four-assignment-per-gate snapshot.
    #[test]
    fn test_rnn_forward_runtime_weights_bidirectional() {
        let mut node = create_rnn_node("rnn1", RnnDirection::Bidirectional, false, 2);
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
                let mut rnn1 = BiRnnConfig::new(4, 8, true)
                    .with_batch_first(false)
                    .init(&self.device);
                let __w = W;
                let __r = R;
                let __b = B;
                {
                    let __w_dir = __w.clone().select_dim::<2>(0, 0);
                    let __r_dir = __r.clone().select_dim::<2>(0, 0);
                    let __b_dir = __b.clone().select_dim::<1>(0, 0);
                    let __b_zero = __b_dir.clone().slice_dim(0, 0..8).zeros_like();
                    rnn1
                        .forward
                        .gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .forward
                        .gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .forward
                        .gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(
                            __b_dir.clone().slice_dim(0, 0..8)
                                + __b_dir.clone().slice_dim(0, 8..16),
                        ),
                    );
                    rnn1
                        .forward
                        .gate
                        .hidden_transform
                        .bias = Some(burn::module::Param::from_tensor(__b_zero.clone()));
                }
                {
                    let __w_dir = __w.select_dim::<2>(0, 1);
                    let __r_dir = __r.select_dim::<2>(0, 1);
                    let __b_dir = __b.select_dim::<1>(0, 1);
                    let __b_zero = __b_dir.clone().slice_dim(0, 0..8).zeros_like();
                    rnn1
                        .reverse
                        .gate
                        .input_transform
                        .weight = burn::module::Param::from_tensor(
                        __w_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .reverse
                        .gate
                        .hidden_transform
                        .weight = burn::module::Param::from_tensor(
                        __r_dir.clone().slice_dim(0, 0..8).transpose(),
                    );
                    rnn1
                        .reverse
                        .gate
                        .input_transform
                        .bias = Some(
                        burn::module::Param::from_tensor(
                            __b_dir.clone().slice_dim(0, 0..8)
                                + __b_dir.clone().slice_dim(0, 8..16),
                        ),
                    );
                    rnn1
                        .reverse
                        .gate
                        .hidden_transform
                        .bias = Some(burn::module::Param::from_tensor(__b_zero.clone()));
                }
                let (output_seq, final_state) = rnn1.forward(input, None);
                (
                    {
                        let [seq_len, batch_size, _] = output_seq.dims();
                        let reshaped = output_seq.reshape([seq_len, batch_size, 2, 8usize]);
                        reshaped.swap_dims(1, 2)
                    },
                    final_state.hidden,
                )
            };
            (Y, Y_h)
        }
        ");
    }

    /// ONNX layout=1 moves the direction axis of `initial_h` too.
    #[test]
    fn test_rnn_forward_batch_first_initial_h() {
        let node = with_initial_state(create_rnn_node("rnn1", RnnDirection::Forward, true, 2));
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
                    .rnn1
                    .forward(input, Some(RnnState::new(initial_h.squeeze_dim(1))));
                (
                    output_seq.unsqueeze_dims::<4>(&[2]),
                    final_state.hidden.unsqueeze_dims::<3>(&[1]),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    #[test]
    fn test_rnn_forward_bidirectional_batch_first() {
        let node = create_rnn_node("rnn1", RnnDirection::Bidirectional, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<3>) -> (Tensor<4>, Tensor<3>) {
            let (Y, Y_h) = {
                let (output_seq, final_state) = self.rnn1.forward(input, None);
                (
                    {
                        let [batch_size, seq_len, _] = output_seq.dims();
                        output_seq.reshape([batch_size, seq_len, 2, 8usize])
                    },
                    final_state.hidden.swap_dims(0, 1),
                )
            };
            (Y, Y_h)
        }
        ");
    }

    /// Give a node the optional inputs an initial state needs, mirroring the ONNX input
    /// order `X, W, R, B, sequence_lens, initial_h`.
    fn with_initial_state(mut node: RnnNode) -> RnnNode {
        node.config.has_initial_h = true;
        node.inputs.push(Argument::new(
            "sequence_lens",
            ArgType::ScalarNative(DType::I64),
        ));
        node.inputs.push(Argument::new(
            "initial_h",
            ArgType::Tensor(TensorType::new(DType::F32, 3, None)),
        ));
        node
    }
}
