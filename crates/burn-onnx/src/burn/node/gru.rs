//! ONNX GRU node import implementation.
//!
//! ## Supported ONNX Features
//!
//! - Forward and reverse directions
//! - Batch-first and sequence-first layouts (`layout` attribute)
//! - Initial hidden state
//! - `linear_before_reset` attribute (maps to Burn's `reset_after`)
//!
//! ## Unsupported ONNX Features
//!
//! - **Bidirectional**: Burn does not have a BiGru module. Bidirectional GRU models will cause
//!   a panic during code generation.
//!
//! - **Variable sequence lengths**: ONNX input `sequence_lens` with shape `[batch_size]` specifies
//!   the actual length of each sequence in a batch. Currently, all sequences in a batch must have
//!   the same length.
//!
//! - **Cell state clipping**: The `clip` attribute is not supported by Burn's GRU module.
//!
//! - **Custom activations**: Burn's GRU uses fixed Sigmoid (gates) and Tanh (hidden) activations.

use super::prelude::*;
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
    use crate::burn::node_traits::{SerializationBackend, extract_node_data};
    use burn::tensor::Tensor;

    let hidden_size = config.hidden_size;
    let input_size = config.input_size;

    let data_w = extract_node_data(inputs, 1);
    let data_r = extract_node_data(inputs, 2);
    let data_b = extract_node_data(inputs, 3);

    let Some(data_w) = data_w else {
        return vec![];
    };
    let Some(data_r) = data_r else {
        return vec![];
    };

    let dtype = data_w.dtype;
    let device = Default::default();

    // ONNX gate order: z(update), r(reset), h(new/hidden)
    // Burn gate names: update_gate, reset_gate, new_gate
    let gate_indices = [0usize, 1, 2]; // z, r, h -> update, reset, new
    let gate_names = ["update_gate", "reset_gate", "new_gate"];

    let direction_prefixes: Vec<&str> = match config.direction {
        GruDirection::Forward | GruDirection::Reverse => vec![""],
        GruDirection::Bidirectional => vec!["forward.", "reverse."],
    };

    let mut snapshots = Vec::new();

    let w_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_w.clone(), &device);
    let r_tensor: Tensor<SerializationBackend, 3> = Tensor::from_data(data_r.clone(), &device);
    let b_tensor: Option<Tensor<SerializationBackend, 2>> =
        data_b.clone().map(|b| Tensor::from_data(b, &device));

    for (dir_idx, dir_prefix) in direction_prefixes.iter().enumerate() {
        // W shape: [num_directions, 3*hidden_size, input_size]
        let w_dir = w_tensor
            .clone()
            .slice([dir_idx..dir_idx + 1, 0..3 * hidden_size, 0..input_size])
            .squeeze::<2>(); // [3*hidden_size, input_size]

        // R shape: [num_directions, 3*hidden_size, hidden_size]
        let r_dir = r_tensor
            .clone()
            .slice([dir_idx..dir_idx + 1, 0..3 * hidden_size, 0..hidden_size])
            .squeeze::<2>(); // [3*hidden_size, hidden_size]

        // B shape: [num_directions, 6*hidden_size]
        let b_dir = b_tensor.as_ref().map(|b| {
            b.clone()
                .slice([dir_idx..dir_idx + 1, 0..6 * hidden_size])
                .squeeze::<1>() // [6*hidden_size]
        });

        for (gate_idx, gate_name) in gate_names.iter().enumerate() {
            let onnx_gate_idx = gate_indices[gate_idx];
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

                let wb: Tensor<SerializationBackend, 1> = b.clone().slice([wb_start..wb_end]);
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
                let rb_start = 3 * hidden_size + onnx_gate_idx * hidden_size;
                let rb_end = rb_start + hidden_size;

                let rb: Tensor<SerializationBackend, 1> = b.clone().slice([rb_start..rb_end]);
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
    use std::rc::Rc;

    let data = data.convert_dtype(dtype);

    let shape = data.shape.clone();
    let path_stack: Vec<String> = path.split('.').map(String::from).collect();
    let container_stack = vec![format!("Struct:{}", container_type)];

    let data_fn = Rc::new(
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

impl NodeCodegen for onnx_ir::gru::GruNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        if self.config.clip.is_some() {
            panic!(
                "GRU clip attribute is not supported. Burn's GRU module does not support cell state clipping."
            );
        }
        if self.config.gate_activation != GruActivationFunction::Sigmoid
            || self.config.hidden_activation != GruActivationFunction::Tanh
        {
            panic!(
                "Custom GRU activations are not supported. Burn's GRU uses fixed Sigmoid (gates) and Tanh (hidden). Got gate: {:?}, hidden: {:?}",
                self.config.gate_activation, self.config.hidden_activation
            );
        }

        let name = Ident::new(&self.name, Span::call_site());
        let d_input = self.config.input_size.to_tokens();
        let d_hidden = self.config.hidden_size.to_tokens();
        let bias = self.config.has_bias;
        // ONNX linear_before_reset maps to Burn reset_after
        let reset_after = self.config.linear_before_reset;

        match self.config.direction {
            GruDirection::Forward | GruDirection::Reverse => Some(Field::new(
                self.name.clone(),
                quote! { burn::nn::gru::Gru<B> },
                quote! {
                    let #name = burn::nn::gru::GruConfig::new(#d_input, #d_hidden, #bias)
                        .with_reset_after(#reset_after)
                        .init(device);
                },
            )),
            GruDirection::Bidirectional => {
                panic!("Bidirectional GRU is not supported. Burn does not have a BiGru module.");
            }
        }
    }

    fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
        collect_gru_snapshots(field_name, &self.inputs, &self.config)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

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

        let has_initial_h = self.config.has_initial_h;
        let is_reverse = matches!(self.config.direction, GruDirection::Reverse);
        let batch_first = self.config.batch_first;

        // Build the initial state expression
        // Input indices: 0=X, 1=W, 2=R, 3=B, 4=sequence_lens, 5=initial_h
        // ONNX initial_h: [num_directions, batch_size, hidden_size]
        // Burn expects: [batch_size, hidden_size] (2D)
        let initial_state_expr = if has_initial_h {
            let h_input = scope.arg(&self.inputs[5]);
            // Squeeze out the direction dimension (index 0)
            quote! { Some(#h_input.squeeze_dim(0)) }
        } else {
            quote! { None }
        };

        // Burn GRU expects [batch_size, seq_length, input_size] (always batch-first)
        // ONNX default (layout=0): [seq_length, batch_size, input_size]
        // ONNX layout=1: [batch_size, seq_length, input_size]
        let input_transform = if batch_first {
            // Already batch-first, no transform needed
            quote! { #input }
        } else {
            // seq-first -> batch-first: swap dims 0 and 1
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

        // Forward call: Burn GRU returns [batch_size, seq_length, hidden_size]
        let forward_call = quote! {
            let gru_output = self.#field.forward(#input_with_direction, #initial_state_expr);
        };

        // For reverse: flip output back
        let output_with_direction = if is_reverse {
            quote! { gru_output.flip([1]) }
        } else {
            quote! { gru_output }
        };

        // Transform output from Burn format to ONNX format
        // Burn: [batch_size, seq_length, hidden_size]
        // ONNX Y (layout=0): [seq_length, num_directions, batch_size, hidden_size]
        // ONNX Y (layout=1): [batch_size, seq_length, num_directions, hidden_size]
        // ONNX Y_h: [num_directions, batch_size, hidden_size]

        // Extract Y_h (final hidden state) from the sequence output.
        // For forward: the last timestep is the final state.
        // For reverse: the final state is the Burn GRU's last output BEFORE flipping,
        // which becomes the first timestep after flipping.
        let y_h_step = if is_reverse {
            // After flip: first timestep = final state from reverse processing
            quote! { 0..1 }
        } else {
            quote! { (seq_len - 1)..seq_len }
        };
        let y_h_expr = quote! {
            {
                let [_batch, seq_len, _hidden] = batch_first_output.dims();
                let step = batch_first_output.clone().slice([0.._batch, #y_h_step, 0.._hidden]);
                // [batch, 1, hidden] -> squeeze seq dim -> [batch, hidden] -> unsqueeze dir dim -> [1, batch, hidden]
                step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
            }
        };

        let y_output_expr = if batch_first {
            // Burn: [batch, seq, hidden] -> ONNX layout=1: [batch, seq, 1, hidden]
            quote! { batch_first_output.clone().unsqueeze_dims::<4>(&[2]) }
        } else {
            // Burn: [batch, seq, hidden] -> swap to [seq, batch, hidden] -> [seq, 1, batch, hidden]
            quote! { batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1]) }
        };

        // Build output assignments
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

    fn register_imports(&self, _imports: &mut BurnImports) {
        // GRU types are accessed via full path in field(), so no extra imports needed
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
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

        if has_initial_h {
            // sequence_lens (unused optional placeholder)
            inputs.push(Argument::new("sequence_lens", ArgType::Scalar(DType::I64)));
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
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
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
        "#);
    }

    #[test]
    fn test_gru_forward_reverse() {
        let node = create_gru_node("gru1", GruDirection::Reverse, false, false, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
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
        "#);
    }

    #[test]
    fn test_gru_forward_y_only() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, false, 1);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> Tensor<B, 4> {
            let Y = {
                let gru_output = self.gru1.forward(input.swap_dims(0, 1), None);
                let batch_first_output = gru_output;
                batch_first_output.clone().swap_dims(0, 1).unsqueeze_dims::<4>(&[1])
            };
            Y
        }
        "#);
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
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
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
                        step.squeeze_dim::<2>(1).unsqueeze_dims::<3>(&[0])
                    },
                )
            };
            (Y, Y_h)
        }
        "#);
    }

    #[test]
    fn test_gru_forward_with_initial_h() {
        let node = create_gru_node("gru1", GruDirection::Forward, false, true, 2);
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r#"
        pub fn forward(
            &self,
            input: Tensor<B, 3>,
            W: Tensor<B, 3>,
            R: Tensor<B, 3>,
            B: Tensor<B, 2>,
            sequence_lens: i64,
            initial_h: Tensor<B, 3>,
        ) -> (Tensor<B, 4>, Tensor<B, 3>) {
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
        "#);
    }
}
