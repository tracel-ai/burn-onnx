use onnx_ir::node::non_max_suppression::{BoxFormat, NonMaxSuppressionNode};

use super::prelude::*;

fn compile_error(message: String) -> TokenStream {
    quote! {
        compile_error!(#message);
    }
}

fn optional_scalar(
    argument: Option<&Argument>,
    scope: &mut ScopeAtPosition<'_>,
) -> Result<Option<TokenStream>, String> {
    let Some(argument) = argument.filter(|argument| !argument.is_optional()) else {
        return Ok(None);
    };

    let value = scope.arg(argument);
    match &argument.ty {
        ArgType::ScalarNative(_) => Ok(Some(value)),
        ArgType::ScalarTensor(dtype) => Ok(Some(on_device_to_native(value, dtype))),
        ArgType::Tensor(tensor) if tensor.rank == 1 => {
            Ok(Some(on_device_to_native(value, &tensor.dtype)))
        }
        other => Err(format!(
            "NonMaxSuppression scalar input validation was bypassed for type {other}"
        )),
    }
}

fn corner_boxes(format: &BoxFormat) -> TokenStream {
    match format {
        BoxFormat::Corner => quote! {{
            let __first_corner: Tensor<2> = __boxes_batch
                .clone()
                .slice_dim(1, 0..2)
                .flip([1]);
            let __second_corner: Tensor<2> = __boxes_batch
                .slice_dim(1, 2..4)
                .flip([1]);
            Tensor::cat(
                alloc::vec![
                    __first_corner.clone().min_pair(__second_corner.clone()),
                    __first_corner.max_pair(__second_corner),
                ],
                1,
            )
        }},
        BoxFormat::Center => quote! {{
            let __center: Tensor<2> = __boxes_batch
                .clone()
                .slice_dim(1, 0..2);
            let __half_size: Tensor<2> = __boxes_batch
                .slice_dim(1, 2..4)
                / 2.0f32;
            Tensor::cat(
                alloc::vec![
                    __center.clone() - __half_size.clone(),
                    __center + __half_size,
                ],
                1,
            )
        }},
    }
}

impl NodeCodegen for NonMaxSuppressionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::vision::Nms");
        imports.register("burn::vision::NmsOptions");
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let [boxes_argument, scores_argument, ..] = self.inputs.as_slice() else {
            return compile_error(format!(
                "NonMaxSuppression node '{}' requires at least two inputs",
                self.name
            ));
        };
        let [output_argument] = self.outputs.as_slice() else {
            return compile_error(format!(
                "NonMaxSuppression node '{}' requires exactly one output",
                self.name
            ));
        };

        let boxes = scope.arg(boxes_argument);
        let scores = scope.arg(scores_argument);
        let output = arg_to_ident(output_argument);

        let max_output = match optional_scalar(self.inputs.get(2), scope) {
            Ok(Some(value)) => value,
            Ok(None) => quote! { 0i64 },
            Err(message) => return compile_error(message),
        };
        let iou_threshold = match optional_scalar(self.inputs.get(3), scope) {
            Ok(Some(value)) => quote! { #value as f32 },
            Ok(None) => quote! { 0.0f32 },
            Err(message) => return compile_error(message),
        };
        let score_threshold = match optional_scalar(self.inputs.get(4), scope) {
            Ok(Some(value)) => quote! { Some(#value as f32) },
            Ok(None) => quote! { None },
            Err(message) => return compile_error(message),
        };
        let corner_boxes = corner_boxes(
            self.config
                .center_point_box
                .as_ref()
                .unwrap_or(&BoxFormat::Corner),
        );

        // Burn's NMS primitive operates on one set of boxes and scores. ONNX
        // applies NMS independently to each batch/class pair and returns
        // [batch_index, class_index, box_index] triples.
        quote! {
            let #output = {
                let __max_output_boxes_per_class: i64 = #max_output;
                let __iou_threshold: f32 = #iou_threshold;
                let __score_threshold: Option<f32> = #score_threshold;
                // ONNX reference runtimes keep scores strictly greater than
                // the threshold, while Burn NMS keeps scores greater than or
                // equal to it. Advance one f32 step to preserve ONNX behavior.
                let __burn_score_threshold = match __score_threshold {
                    Some(threshold) if threshold == f32::INFINITY => f32::NAN,
                    Some(threshold) => threshold.next_up(),
                    None => f32::NEG_INFINITY,
                };
                let __device = #boxes.device();
                let [__num_batches, _, _] = #boxes.dims();
                let [_, __num_classes, _] = #scores.dims();
                let mut __selected: alloc::vec::Vec<Tensor<2, Int>> = alloc::vec::Vec::new();

                // ONNX defines zero as no output, whereas Burn uses zero as
                // unlimited, so skip the primitive entirely in that case.
                if __max_output_boxes_per_class > 0 {
                    let __max_output_boxes = usize::try_from(
                        __max_output_boxes_per_class,
                    )
                    .unwrap_or(usize::MAX);

                    for __batch in 0..__num_batches {
                        let __boxes_batch: Tensor<2> = #boxes
                            .clone()
                            .select_dim(0, __batch);
                        let __corner_boxes: Tensor<2> = #corner_boxes;
                        let __scores_batch: Tensor<2> = #scores
                            .clone()
                            .select_dim(0, __batch);
                        for __class in 0..__num_classes {
                            let __class_scores: Tensor<1> = __scores_batch
                                .clone()
                                .select_dim(0, __class);
                            let __kept: Tensor<1, Int> = __corner_boxes.clone().nms(
                                __class_scores,
                                NmsOptions {
                                    iou_threshold: __iou_threshold,
                                    score_threshold: __burn_score_threshold,
                                    max_output_boxes: __max_output_boxes,
                                },
                            );
                            let [__num_kept] = __kept.dims();

                            if __num_kept > 0 {
                                let __batch_indices = Tensor::<1, Int>::full(
                                    [__num_kept],
                                    __batch as i64,
                                    (&__device, burn::tensor::DType::I64),
                                );
                                let __class_indices = Tensor::<1, Int>::full(
                                    [__num_kept],
                                    __class as i64,
                                    (&__device, burn::tensor::DType::I64),
                                );
                                __selected.push(Tensor::stack(
                                    alloc::vec![
                                        __batch_indices,
                                        __class_indices,
                                        __kept.cast(burn::tensor::DType::I64),
                                    ],
                                    1,
                                ));
                            }
                        }
                    }
                }

                if __selected.is_empty() {
                    Tensor::<2, Int>::empty(
                        [0, 3],
                        (&__device, burn::tensor::DType::I64),
                    )
                } else {
                    Tensor::cat(__selected, 0)
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use insta::assert_snapshot;
    use onnx_ir::node::non_max_suppression::{
        NonMaxSuppressionConfig, NonMaxSuppressionNodeBuilder,
    };

    fn base_node(format: Option<BoxFormat>) -> NonMaxSuppressionNodeBuilder {
        NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .input_tensor("scores", 3, DType::F32)
            .config(NonMaxSuppressionConfig::new(format))
    }

    fn generated_section(node: &NonMaxSuppressionNode, start: &str, end: &str) -> String {
        let code = codegen_forward_default(node);
        let (_, section) = code.split_once(start).expect("missing section start");
        let (section, _) = section.split_once(end).expect("missing section end");

        format!("{start}{section}")
            .lines()
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn scalar_setup(node: &NonMaxSuppressionNode) -> String {
        generated_section(
            node,
            "let __max_output_boxes_per_class",
            "let __burn_score_threshold",
        )
    }

    fn box_conversion(node: &NonMaxSuppressionNode) -> String {
        generated_section(node, "let __corner_boxes", "let __scores_batch")
    }

    #[test]
    fn corner_with_native_scalars() {
        let node = base_node(None)
            .input_scalar("max_output", DType::I64)
            .input_scalar("iou_threshold", DType::F32)
            .input_scalar("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .build();

        assert_snapshot!(scalar_setup(&node), @"
        let __max_output_boxes_per_class: i64 = max_output;
        let __iou_threshold: f32 = iou_threshold as f32;
        let __score_threshold: Option<f32> = Some(score_threshold as f32);
        ");
        assert_snapshot!(box_conversion(&node), @"
        let __corner_boxes: Tensor<2> = {
        let __first_corner: Tensor<2> = __boxes_batch
        .clone()
        .slice_dim(1, 0..2)
        .flip([1]);
        let __second_corner: Tensor<2> = __boxes_batch
        .slice_dim(1, 2..4)
        .flip([1]);
        Tensor::cat(
        alloc::vec![
        __first_corner.clone().min_pair(__second_corner.clone()),
        __first_corner.max_pair(__second_corner),
        ],
        1,
        )
        };
        ");
    }

    #[test]
    fn center_with_scalar_tensors() {
        let scalar_tensor_node = base_node(Some(BoxFormat::Center))
            .input_scalar_tensor("max_output", DType::I64)
            .input_scalar_tensor("iou_threshold", DType::F32)
            .input_scalar_tensor("score_threshold", DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .build();
        let rank_one_tensor_node = base_node(Some(BoxFormat::Center))
            .input_tensor("max_output", 1, DType::I64)
            .input_tensor("iou_threshold", 1, DType::F32)
            .input_tensor("score_threshold", 1, DType::F32)
            .output_tensor("selected_indices", 2, DType::I64)
            .build();

        assert_snapshot!(box_conversion(&scalar_tensor_node), @"
        let __corner_boxes: Tensor<2> = {
        let __center: Tensor<2> = __boxes_batch.clone().slice_dim(1, 0..2);
        let __half_size: Tensor<2> = __boxes_batch.slice_dim(1, 2..4)
        / 2.0f32;
        Tensor::cat(
        alloc::vec![
        __center.clone() - __half_size.clone(), __center +
        __half_size,
        ],
        1,
        )
        };
        ");
        assert_snapshot!(scalar_setup(&scalar_tensor_node), @"
        let __max_output_boxes_per_class: i64 = (max_output).into_scalar::<i64>();
        let __iou_threshold: f32 = (iou_threshold).into_scalar::<f32>() as f32;
        let __score_threshold: Option<f32> = Some(
        (score_threshold).into_scalar::<f32>() as f32,
        );
        ");

        assert_eq!(
            scalar_setup(&scalar_tensor_node),
            scalar_setup(&rank_one_tensor_node)
        );
    }

    #[test]
    fn omitted_optional_inputs() {
        let node = base_node(Some(BoxFormat::Corner))
            .output_tensor("selected_indices", 2, DType::I64)
            .build();

        assert_snapshot!(scalar_setup(&node), @"
        let __max_output_boxes_per_class: i64 = 0i64;
        let __iou_threshold: f32 = 0.0f32;
        let __score_threshold: Option<f32> = None;
        ");
    }

    #[test]
    fn missing_required_input_emits_compile_error() {
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F32)
            .config(NonMaxSuppressionConfig::new(None))
            .output_tensor("selected_indices", 2, DType::I64)
            .build();

        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(&self, boxes: Tensor<3>) -> Tensor<2, Int> {
            compile_error!("NonMaxSuppression node 'nms' requires at least two inputs");
            selected_indices
        }
        "#);
    }

    #[test]
    fn wrong_output_count_emits_compile_error() {
        let node = base_node(None)
            .output_tensor("selected_indices", 2, DType::I64)
            .output_tensor("extra_output", 2, DType::I64)
            .build();

        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(
            &self,
            boxes: Tensor<3>,
            scores: Tensor<3>,
        ) -> (Tensor<2, Int>, Tensor<2, Int>) {
            compile_error!("NonMaxSuppression node 'nms' requires exactly one output");
            (selected_indices, extra_output)
        }
        "#);
    }

    #[test]
    fn invalid_scalar_type_emits_compile_error() {
        let node = base_node(None)
            .input_tensor("max_output", 2, DType::I64)
            .output_tensor("selected_indices", 2, DType::I64)
            .build();

        assert_snapshot!(codegen_forward_default(&node), @r#"
        pub fn forward(
            &self,
            boxes: Tensor<3>,
            scores: Tensor<3>,
            max_output: Tensor<2, Int>,
        ) -> Tensor<2, Int> {
            compile_error!(
                "NonMaxSuppression scalar input validation was bypassed for type I64[?, ?]"
            );
            selected_indices
        }
        "#);
    }
}
