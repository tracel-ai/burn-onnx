use super::prelude::*;

impl NodeCodegen for onnx_ir::svmregressor::SVMRegressorNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Store coefficients and support vectors as a tuple of tensors
        let coefficients = self.config.coefficients.as_ref()?;
        let support_vectors = self.config.support_vectors.as_ref()?;
        let n_supports = self.config.n_supports.unwrap_or(0) as usize;

        if coefficients.is_empty() || support_vectors.is_empty() || n_supports == 0 {
            return None;
        }

        let n_features = support_vectors.len() / n_supports;
        let name = Ident::new(&self.name, Span::call_site());

        let coef_data: Vec<_> = coefficients.to_vec();
        let sv_data: Vec<_> = support_vectors.to_vec();

        // Store as a tuple (coefficients, support_vectors)
        // Store coefficients as [n_supports, 1] for direct matmul compatibility
        Some(Field::new(
            &self.name,
            quote! { (Tensor<B, 2>, Tensor<B, 2>) },
            quote! {
                let #name = (
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(alloc::vec![#(#coef_data),*], [#n_supports, 1]),
                        (device, burn::tensor::DType::F32),
                    ),
                    Tensor::<B, 2>::from_data(
                        burn::tensor::TensorData::new(alloc::vec![#(#sv_data),*], [#n_supports, #n_features]),
                        (device, burn::tensor::DType::F32),
                    )
                );
            },
        ))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(&self.inputs[0]);
        let output = arg_to_ident(&self.outputs[0]);

        // Extract configuration
        let kernel_type = self.config.kernel_type.as_deref().unwrap_or("LINEAR");
        let post_transform = self.config.post_transform.as_deref().unwrap_or("NONE");

        let has_data = self.config.coefficients.is_some() && self.config.support_vectors.is_some();
        let rho = self
            .config
            .rho
            .as_ref()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        let n_supports = self.config.n_supports.unwrap_or(0) as usize;

        // Reference the stored tensors (tuple access .0 and .1)
        let field_name = Ident::new(&self.name, Span::call_site());

        // Cast to F32: the ONNX spec allows T in {double, float, int32, int64}.
        // Weight tensors are always F32, so the input must be cast before matmul.
        let input_dtype = match &self.inputs[0].ty {
            ArgType::Tensor(t) => t.dtype,
            _ => DType::F32,
        };
        let cast_expr = match input_dtype {
            DType::F32 => quote! { #input },
            DType::F64 => quote! { #input.cast(burn::tensor::DType::F32) },
            _ => quote! { #input.float().cast(burn::tensor::DType::F32) },
        };

        // Generate kernel computation based on kernel type
        let kernel_computation = if !has_data {
            quote! { #cast_expr }
        } else {
            match kernel_type {
                "LINEAR" => {
                    quote! {
                        {
                            let x = #cast_expr;
                            let rho = #rho;
                            let (coef, sv) = &self.#field_name;

                            // Compute linear kernel: K(x, sv) = x · sv
                            // Compute kernel matrix: x @ sv^T → [batch, n_supports]
                            let kernel_values = x.matmul(sv.clone().transpose());

                            // prediction = kernel_values @ coefficients + rho → [batch, 1]
                            kernel_values.matmul(coef.clone()) + rho
                        }
                    }
                }
                "RBF" => {
                    let gamma = self
                        .config
                        .kernel_params
                        .as_ref()
                        .and_then(|p| p.first())
                        .copied()
                        .unwrap_or(0.1);

                    quote! {
                        {
                            let x = #cast_expr;
                            let rho = #rho;
                            let gamma = #gamma;
                            let (coef, sv) = &self.#field_name;

                            // Compute RBF kernel: K(x, sv) = exp(-gamma * ||x - sv||^2)
                            let [batch_size, _] = x.dims();

                            // Vectorized squared distance using:
                            // ||x - s||^2 = ||x||^2 + ||s||^2 - 2 * x·s
                            let dot_products = x.clone().matmul(sv.clone().transpose());
                            let input_norms = x.clone().powf_scalar(2.0).sum_dim(1).reshape([batch_size, 1]);
                            let sv_norms = sv.clone().powf_scalar(2.0).sum_dim(1).reshape([1, #n_supports]);
                            let sq_distances = input_norms + sv_norms - dot_products * 2.0;
                            let kernel_values = (sq_distances * (-gamma)).exp();

                            kernel_values.matmul(coef.clone()) + rho
                        }
                    }
                }
                "POLY" => {
                    let params = self.config.kernel_params.as_ref();
                    let gamma = params.and_then(|p| p.first()).copied().unwrap_or(1.0);
                    let coef0 = params.and_then(|p| p.get(1)).copied().unwrap_or(0.0);
                    let degree = params.and_then(|p| p.get(2)).copied().unwrap_or(3.0);

                    quote! {
                        {
                            let x = #cast_expr;
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            let degree = #degree;
                            let (coef, sv) = &self.#field_name;

                            // Compute polynomial kernel: K(x, sv) = (gamma * x · sv + coef0)^degree
                            let dot_products = x.matmul(sv.clone().transpose());
                            let kernel_values = (dot_products * gamma + coef0).powf_scalar(degree);

                            kernel_values.matmul(coef.clone()) + rho
                        }
                    }
                }
                "SIGMOID" => {
                    let params = self.config.kernel_params.as_ref();
                    let gamma = params.and_then(|p| p.first()).copied().unwrap_or(1.0);
                    let coef0 = params.and_then(|p| p.get(1)).copied().unwrap_or(0.0);

                    quote! {
                        {
                            let x = #cast_expr;
                            let rho = #rho;
                            let gamma = #gamma;
                            let coef0 = #coef0;
                            let (coef, sv) = &self.#field_name;

                            // Compute sigmoid kernel: K(x, sv) = tanh(gamma * x · sv + coef0)
                            let dot_products = x.matmul(sv.clone().transpose());
                            let kernel_values = (dot_products * gamma + coef0).tanh();

                            kernel_values.matmul(coef.clone()) + rho
                        }
                    }
                }
                _ => quote! { #cast_expr },
            }
        };

        // Apply post-transform if needed
        let function = match post_transform {
            "NONE" => kernel_computation,
            "LOGISTIC" => quote! { { let y = #kernel_computation; (y.neg().exp() + 1.0).recip() } },
            "SOFTMAX" => {
                quote! { { let y = #kernel_computation; let e = y.exp(); e.clone() / e.sum_dim(1) } }
            }
            "SOFTMAX_ZERO" => quote! { {
                let y = #kernel_computation;
                // Append a zero logit column → [N, 2], softmax over targets, take first column.
                let [batch, _] = y.dims();
                let zero_col = burn::tensor::Tensor::<B, 2>::zeros(
                    burn::tensor::Shape::new([batch, 1usize]),
                    &y.device(),
                );
                let combined = burn::tensor::Tensor::cat(alloc::vec![y, zero_col], 1);
                let e = combined.exp();
                e.clone().narrow(1, 0, 1) / e.sum_dim(1)
            } },
            "PROBIT" => quote! {
                compile_error!(
                    "SVMRegressor: PROBIT post_transform requires the inverse normal CDF (erfinv) \
                     which is not available in Burn's tensor API. Use a different post_transform \
                     or implement erfinv manually."
                )
            },
            _ => kernel_computation,
        };

        quote! {
            let #output = { #function }.squeeze_dim::<1>(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::svmregressor::{SVMRegressorConfig, SVMRegressorNodeBuilder};

    fn make_node(
        kernel: &str,
        post_transform: Option<&str>,
        n_supports: i64,
        kernel_params: Option<Vec<f32>>,
    ) -> onnx_ir::svmregressor::SVMRegressorNode {
        SVMRegressorNodeBuilder::new("svm1")
            .input_tensor("input", 2, DType::F32)
            .output_tensor("output", 1, DType::F32)
            .config(SVMRegressorConfig::new(
                Some(vec![1.0, -0.5]),
                kernel_params,
                Some(kernel.to_string()),
                Some(n_supports),
                None,
                post_transform.map(|s| s.to_string()),
                Some(vec![0.5]),
                Some(vec![1.0, 2.0, 3.0, 4.0]),
            ))
            .build()
    }

    #[test]
    fn test_svm_linear() {
        let code = codegen_forward_default(&make_node("LINEAR", None, 2, None));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let x = input;
                    let rho = 0.5f32;
                    let (coef, sv) = &self.svm1;
                    let kernel_values = x.matmul(sv.clone().transpose());
                    kernel_values.matmul(coef.clone()) + rho
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_rbf() {
        let code = codegen_forward_default(&make_node("RBF", None, 2, Some(vec![0.1])));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let x = input;
                    let rho = 0.5f32;
                    let gamma = 0.1f32;
                    let (coef, sv) = &self.svm1;
                    let [batch_size, _] = x.dims();
                    let dot_products = x.clone().matmul(sv.clone().transpose());
                    let input_norms = x
                        .clone()
                        .powf_scalar(2.0)
                        .sum_dim(1)
                        .reshape([batch_size, 1]);
                    let sv_norms = sv.clone().powf_scalar(2.0).sum_dim(1).reshape([1, 2usize]);
                    let sq_distances = input_norms + sv_norms - dot_products * 2.0;
                    let kernel_values = (sq_distances * (-gamma)).exp();
                    kernel_values.matmul(coef.clone()) + rho
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_poly() {
        let code = codegen_forward_default(&make_node("POLY", None, 2, Some(vec![1.0, 0.0, 3.0])));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let x = input;
                    let rho = 0.5f32;
                    let gamma = 1f32;
                    let coef0 = 0f32;
                    let degree = 3f32;
                    let (coef, sv) = &self.svm1;
                    let dot_products = x.matmul(sv.clone().transpose());
                    let kernel_values = (dot_products * gamma + coef0).powf_scalar(degree);
                    kernel_values.matmul(coef.clone()) + rho
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_sigmoid() {
        let code = codegen_forward_default(&make_node("SIGMOID", None, 2, Some(vec![0.5, 1.0])));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let x = input;
                    let rho = 0.5f32;
                    let gamma = 0.5f32;
                    let coef0 = 1f32;
                    let (coef, sv) = &self.svm1;
                    let dot_products = x.matmul(sv.clone().transpose());
                    let kernel_values = (dot_products * gamma + coef0).tanh();
                    kernel_values.matmul(coef.clone()) + rho
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_linear_logistic() {
        let code = codegen_forward_default(&make_node("LINEAR", Some("LOGISTIC"), 2, None));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let y = {
                        let x = input;
                        let rho = 0.5f32;
                        let (coef, sv) = &self.svm1;
                        let kernel_values = x.matmul(sv.clone().transpose());
                        kernel_values.matmul(coef.clone()) + rho
                    };
                    (y.neg().exp() + 1.0).recip()
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_linear_softmax() {
        let code = codegen_forward_default(&make_node("LINEAR", Some("SOFTMAX"), 2, None));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let y = {
                        let x = input;
                        let rho = 0.5f32;
                        let (coef, sv) = &self.svm1;
                        let kernel_values = x.matmul(sv.clone().transpose());
                        kernel_values.matmul(coef.clone()) + rho
                    };
                    let e = y.exp();
                    e.clone() / e.sum_dim(1)
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }

    #[test]
    fn test_svm_linear_softmax_zero() {
        let code = codegen_forward_default(&make_node("LINEAR", Some("SOFTMAX_ZERO"), 2, None));
        assert_snapshot!(code, @"
        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
            let output = {
                {
                    let y = {
                        let x = input;
                        let rho = 0.5f32;
                        let (coef, sv) = &self.svm1;
                        let kernel_values = x.matmul(sv.clone().transpose());
                        kernel_values.matmul(coef.clone()) + rho
                    };
                    let [batch, _] = y.dims();
                    let zero_col = burn::tensor::Tensor::<
                        B,
                        2,
                    >::zeros(burn::tensor::Shape::new([batch, 1usize]), &y.device());
                    let combined = burn::tensor::Tensor::cat(alloc::vec![y, zero_col], 1);
                    let e = combined.exp();
                    e.clone().narrow(1, 0, 1) / e.sum_dim(1)
                }
            }
                .squeeze_dim::<1>(1);
            output
        }
        ");
    }
}
