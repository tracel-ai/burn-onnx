use crate::{
    ir::{ArgType, Argument, AttributeValue, DType, Node, RawNode, TensorType},
    processor::{InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError},
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Kernel function type for the SVMRegressor operator.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SVMKernelType {
    /// Linear kernel: K(x, sv) = x · sv
    #[default]
    Linear,
    /// Polynomial kernel: K(x, sv) = (gamma * x · sv + coef0)^degree
    Poly,
    /// Radial basis function kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
    Rbf,
    /// Sigmoid kernel: K(x, sv) = tanh(gamma * x · sv + coef0)
    Sigmoid,
}

impl std::str::FromStr for SVMKernelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LINEAR" => Ok(SVMKernelType::Linear),
            "POLY" => Ok(SVMKernelType::Poly),
            "RBF" => Ok(SVMKernelType::Rbf),
            "SIGMOID" => Ok(SVMKernelType::Sigmoid),
            _ => Err(format!("Invalid kernel type: {s}")),
        }
    }
}

/// Post-transform function applied to the raw SVM output.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SVMPostTransform {
    /// No post-transform; return raw scores.
    #[default]
    None,
    /// Apply logistic sigmoid: 1 / (1 + exp(-y))
    Logistic,
    /// Apply softmax over all class scores.
    Softmax,
    /// Apply softmax with an implicit zero class appended.
    SoftmaxZero,
    /// Apply probit (inverse normal CDF) — not supported in Burn.
    Probit,
}

impl std::str::FromStr for SVMPostTransform {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "NONE" => Ok(SVMPostTransform::None),
            "LOGISTIC" => Ok(SVMPostTransform::Logistic),
            "SOFTMAX" => Ok(SVMPostTransform::Softmax),
            "SOFTMAX_ZERO" => Ok(SVMPostTransform::SoftmaxZero),
            "PROBIT" => Ok(SVMPostTransform::Probit),
            _ => Err(format!("Invalid post transform: {s}")),
        }
    }
}

/// Configuration for the SVMRegressor operator.
///
/// Performs regression using Support Vector Machine (SVM) with various kernel types.
#[allow(clippy::too_many_arguments)]
#[derive(Debug, Clone, Default, new)]
pub struct SVMRegressorConfig {
    /// Coefficients for the support vector in the decision function.
    pub coefficients: Option<Vec<f32>>,
    /// Parameters for the kernel function (gamma, coef0, degree).
    pub kernel_params: Option<Vec<f32>>,
    /// Type of kernel function. Default: [`SVMKernelType::Linear`].
    pub kernel_type: SVMKernelType,
    /// Number of support vectors. Default: `0`.
    pub n_supports: Option<i64>,
    /// Number of features per support vector (= support_vectors.len() / n_supports).
    pub n_features: Option<usize>,
    /// Flag indicating one-class SVM anomaly detection mode. Default: `0` (disabled).
    pub one_class: Option<i64>,
    /// How to transform the output. Default: [`SVMPostTransform::None`].
    pub post_transform: SVMPostTransform,
    /// Bias term(s) in decision function.
    pub rho: Option<Vec<f32>>,
    /// Support vectors.
    pub support_vectors: Option<Vec<f32>>,
}

/// SVMRegressor ONNX operator.
///
/// Performs regression using Support Vector Machine (SVM) with configurable kernel types:
/// - LINEAR: K(x, sv) = x · sv
/// - RBF: K(x, sv) = exp(-gamma * ||x - sv||^2)
/// - POLY: K(x, sv) = (gamma * x · sv + coef0)^degree
/// - SIGMOID: K(x, sv) = tanh(gamma * x · sv + coef0)
#[derive(Debug, Clone, new, NodeBuilder)]
pub struct SVMRegressorNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: SVMRegressorConfig,
}

pub(crate) struct SVMRegressorProcessor;

impl NodeProcessor for SVMRegressorProcessor {
    type Config = SVMRegressorConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate input is a tensor with an allowed dtype.
        // Per the ONNX spec: T may be tensor(float), tensor(double), tensor(int32), tensor(int64).
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.clone(),
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        };

        match input_tensor.dtype {
            DType::F32 | DType::F64 | DType::I32 | DType::I64 => {}
            other => {
                return Err(ProcessError::TypeMismatch {
                    expected: "f32, f64, i32, or i64".to_string(),
                    actual: format!("{other:?}"),
                });
            }
        }

        // The Burn codegen assumes a 2-D input [N, C] (matmul, sum_dim(1), etc.).
        // Reject other ranks with a clear error rather than generating non-compiling code.
        if input_tensor.rank != 2 {
            return Err(ProcessError::Custom(format!(
                "SVMRegressor currently supports only rank-2 input tensors [N, C]; got rank {}",
                input_tensor.rank
            )));
        }

        // Output is always tensor(float) with shape [N] — one score per sample.
        let output_shape = input_tensor
            .static_shape
            .as_ref()
            .map(|shape| shape.iter().take(1).cloned().collect());

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::F32,
            rank: 1,
            static_shape: output_shape,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        use std::str::FromStr;

        let mut coefficients: Option<Vec<f32>> = None;
        let mut kernel_params: Option<Vec<f32>> = None;
        let mut kernel_type = SVMKernelType::default();
        let mut n_supports: Option<i64> = None;
        let mut one_class: Option<i64> = None;
        let mut n_features: Option<usize> = None;
        let mut post_transform = SVMPostTransform::default();
        let mut rho: Option<Vec<f32>> = None;
        let mut support_vectors: Option<Vec<f32>> = None;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "coefficients" => match value {
                    AttributeValue::Float32s(floats) => coefficients = Some(floats.clone()),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "coefficients".to_string(),
                            reason: "expected Float32s".to_string(),
                        });
                    }
                },
                "kernel_params" => match value {
                    AttributeValue::Float32s(floats) => kernel_params = Some(floats.clone()),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "kernel_params".to_string(),
                            reason: "expected Float32s".to_string(),
                        });
                    }
                },
                "kernel_type" => match value {
                    AttributeValue::String(s) => {
                        kernel_type = SVMKernelType::from_str(s).map_err(|e| {
                            ProcessError::InvalidAttribute {
                                name: "kernel_type".to_string(),
                                reason: e,
                            }
                        })?;
                    }
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "kernel_type".to_string(),
                            reason: "expected String".to_string(),
                        });
                    }
                },
                "n_supports" => match value {
                    AttributeValue::Int64(n) => n_supports = Some(*n),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "n_supports".to_string(),
                            reason: "expected Int64".to_string(),
                        });
                    }
                },
                "one_class" => match value {
                    AttributeValue::Int64(n) => one_class = Some(*n),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "one_class".to_string(),
                            reason: "expected Int64".to_string(),
                        });
                    }
                },
                "post_transform" => match value {
                    AttributeValue::String(s) => {
                        post_transform = SVMPostTransform::from_str(s).map_err(|e| {
                            ProcessError::InvalidAttribute {
                                name: "post_transform".to_string(),
                                reason: e,
                            }
                        })?;
                    }
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "post_transform".to_string(),
                            reason: "expected String".to_string(),
                        });
                    }
                },
                "rho" => match value {
                    AttributeValue::Float32s(floats) => rho = Some(floats.clone()),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "rho".to_string(),
                            reason: "expected Float32s".to_string(),
                        });
                    }
                },
                "support_vectors" => match value {
                    AttributeValue::Float32s(floats) => support_vectors = Some(floats.clone()),
                    _ => {
                        return Err(ProcessError::InvalidAttribute {
                            name: "support_vectors".to_string(),
                            reason: "expected Float32s".to_string(),
                        });
                    }
                },
                _ => {}
            }
        }

        if one_class == Some(1) {
            return Err(ProcessError::InvalidAttribute {
                name: "one_class".to_string(),
                reason: "one-class SVM anomaly detection mode (one_class=1) is not yet supported; \
                         only standard regression mode (one_class=0) is implemented"
                    .to_string(),
            });
        }

        if post_transform == SVMPostTransform::Probit {
            return Err(ProcessError::InvalidAttribute {
                name: "post_transform".to_string(),
                reason: "PROBIT post_transform requires the inverse normal CDF (erfinv) \
                         which is not available in Burn's tensor API; \
                         use a different post_transform"
                    .to_string(),
            });
        }

        // SOFTMAX over a single-target output ([N, 1]) reduces to a tensor of ones,
        // because softmax(x) over a size-1 dim is always 1. Until multi-target
        // SVMRegressor is supported, reject SOFTMAX here so users get a clear error
        // instead of silently-broken predictions. SOFTMAX_ZERO is still meaningful
        // (it appends an implicit zero class), so we don't reject it.
        if post_transform == SVMPostTransform::Softmax {
            return Err(ProcessError::InvalidAttribute {
                name: "post_transform".to_string(),
                reason: "SOFTMAX post_transform is degenerate for single-target SVMRegressor \
                         (softmax over a size-1 dim is always 1); \
                         multi-target output is not yet supported. \
                         Use NONE, LOGISTIC, or SOFTMAX_ZERO instead"
                    .to_string(),
            });
        }

        // RBF, POLY, and SIGMOID all require kernel_params [gamma, coef0, degree].
        // A missing or empty kernel_params for these kernels indicates a malformed model.
        match kernel_type {
            SVMKernelType::Rbf | SVMKernelType::Poly | SVMKernelType::Sigmoid => {
                if kernel_params.as_ref().is_none_or(|p| p.is_empty()) {
                    return Err(ProcessError::MissingAttribute(format!(
                        "kernel_params (gamma, coef0, degree) is required for {:?} kernel",
                        kernel_type
                    )));
                }
            }
            SVMKernelType::Linear => {}
        }

        // Validate attribute consistency when support vectors are present.
        // The current codegen hardcodes n_targets=1 (single-output regression).
        if let Some(n) = n_supports {
            let n = n as usize;

            if n == 0 {
                return Err(ProcessError::InvalidAttribute {
                    name: "n_supports".to_string(),
                    reason: "n_supports must be greater than 0".to_string(),
                });
            }

            if let Some(ref coef) = coefficients
                && coef.len() != n
            {
                return Err(ProcessError::InvalidAttribute {
                    name: "coefficients".to_string(),
                    reason: format!(
                        "multi-target SVMRegressor is not yet supported; \
                         expected coefficients.len() == n_supports ({n}), got {}",
                        coef.len()
                    ),
                });
            }

            if let Some(ref sv) = support_vectors
                && sv.len() % n != 0
            {
                return Err(ProcessError::InvalidAttribute {
                    name: "support_vectors".to_string(),
                    reason: format!(
                        "support_vectors.len() ({}) is not divisible by n_supports ({n})",
                        sv.len()
                    ),
                });
            }
        }

        if let Some(ref r) = rho
            && r.len() != 1
        {
            return Err(ProcessError::InvalidAttribute {
                name: "rho".to_string(),
                reason: format!(
                    "multi-target SVMRegressor is not yet supported; \
                     expected rho.len() == 1, got {}",
                    r.len()
                ),
            });
        }

        // If coefficients or support_vectors are present, n_supports must also be present
        // and the vectors must be non-empty so that codegen can construct valid tensor shapes.
        let has_weights = coefficients.is_some() || support_vectors.is_some();
        if has_weights {
            if n_supports.is_none() {
                return Err(ProcessError::MissingAttribute(
                    "n_supports is required when coefficients or support_vectors are provided"
                        .to_string(),
                ));
            }
            if coefficients.as_ref().is_none_or(|v| v.is_empty()) {
                return Err(ProcessError::InvalidAttribute {
                    name: "coefficients".to_string(),
                    reason: "coefficients must not be empty".to_string(),
                });
            }
            if support_vectors.as_ref().is_none_or(|v| v.is_empty()) {
                return Err(ProcessError::InvalidAttribute {
                    name: "support_vectors".to_string(),
                    reason: "support_vectors must not be empty".to_string(),
                });
            }
            // Compute and stash n_features so codegen doesn't re-derive it.
            // Divisibility was already validated above.
            if let (Some(n), Some(sv)) = (n_supports, &support_vectors) {
                n_features = Some(sv.len() / n as usize);
            }
        }

        Ok(SVMRegressorConfig::new(
            coefficients,
            kernel_params,
            kernel_type,
            n_supports,
            n_features,
            one_class,
            post_transform,
            rho,
            support_vectors,
        ))
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");
        Node::SVMRegressor(SVMRegressorNode::new(
            builder.name,
            builder.inputs,
            builder.outputs,
            config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svmregressor_config() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0, -0.5]),
            None,
            SVMKernelType::Linear,
            Some(2),
            Some(2),
            None,
            SVMPostTransform::None,
            Some(vec![0.5]),
            Some(vec![1.0, 2.0, 3.0, 4.0]),
        );
        assert_eq!(config.coefficients, Some(vec![1.0, -0.5]));
        assert_eq!(config.kernel_type, SVMKernelType::Linear);
        assert_eq!(config.n_supports, Some(2));
        assert_eq!(config.n_features, Some(2));
        assert_eq!(config.rho, Some(vec![0.5]));
    }

    #[test]
    fn test_svmregressor_node_builder() {
        let config = SVMRegressorConfig::new(
            Some(vec![1.0]),
            None,
            SVMKernelType::Linear,
            Some(1),
            Some(2),
            None,
            SVMPostTransform::None,
            Some(vec![0.0]),
            Some(vec![1.0, 2.0]),
        );
        let node = SVMRegressorNode::new("test_svm".to_string(), vec![], vec![], config);

        assert_eq!(node.name, "test_svm");
        assert_eq!(node.inputs.len(), 0);
        assert_eq!(node.outputs.len(), 0);
        assert!(node.config.coefficients.is_some());
        assert_eq!(node.config.kernel_type, SVMKernelType::Linear);
    }
}
