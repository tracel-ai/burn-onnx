//! Import `custom_model.onnx`, which uses two operators from the vendor domain
//! `example.custom` that burn-onnx does not know, plus a built-in `Sigmoid`
//! that we want routed to our own kernel.
//!
//! Each hook does two jobs:
//!   * type inference, during parsing (`infer_output_types`)
//!   * code generation, emitting the call into `src/ops.rs` (`forward`)
//!
//! Tip: to discover what a model needs, run `ModelGen` with NO hooks
//! registered. The build fails with a list of every custom `(domain, op_type)`
//! pair and how many nodes use it - that list is the set of hooks to write.

use burn_onnx::ModelGen;
use burn_onnx::ext::proc_macro2::TokenStream;
use burn_onnx::ext::{
    ArgType, CodegenContext, CustomNode, CustomOp, Imports, Node, NodeType, OpOverride,
    ProcessError, arg_to_ident, quote::quote,
};

fn main() {
    println!("cargo:rerun-if-changed=src/model/custom_model.onnx");

    ModelGen::new()
        .input("src/model/custom_model.onnx")
        .out_dir("model/")
        .register_custom_op(ScaleBias)
        .register_custom_op(ChannelScale)
        .register_op_override(SigmoidOverride)
        .run_from_script();
}

// ---------------------------------------------------------------------------
// Custom op 1: example.custom::ScaleBias - attributes only
// ---------------------------------------------------------------------------

/// `y = x * scale + bias`, with `scale` and `bias` as ONNX FLOAT attributes.
struct ScaleBias;

/// Attributes parsed once and reused by both hook methods.
///
/// `infer_output_types` and `forward` run in different phases, so both need to
/// read the attributes. Parsing in one place means validation errors surface
/// during parsing (with a friendly message) rather than mid-codegen.
struct ScaleBiasConfig {
    scale: f32,
    bias: f32,
}

impl ScaleBias {
    fn config(node: &CustomNode) -> Result<ScaleBiasConfig, ProcessError> {
        let attr = |name: &str| {
            node.attrs
                .get_f32(name)
                .ok_or_else(|| ProcessError::MissingAttribute(name.to_string()))
        };
        Ok(ScaleBiasConfig {
            scale: attr("scale")?,
            bias: attr("bias")?,
        })
    }
}

impl CustomOp for ScaleBias {
    fn op_type(&self) -> &str {
        "ScaleBias"
    }

    fn domain(&self) -> &str {
        "example.custom"
    }

    fn infer_output_types(&self, node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
        // Validate the attributes here so a malformed model fails during
        // parsing, not during code generation.
        Self::config(node)?;
        let input = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("x".to_string()))?;
        // Elementwise: the output type matches the input exactly (including
        // its static shape, which keeps downstream shape inference precise).
        Ok(vec![input.ty.clone()])
    }

    fn forward(
        &self,
        node: &CustomNode,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError> {
        let config = Self::config(node)?;
        // ctx.arg handles clone tracking; use it for every INPUT.
        let x = ctx.arg(&node.inputs[0]);
        // arg_to_ident is for OUTPUTS (and host-side values) only.
        let out = arg_to_ident(&node.outputs[0]);
        let (scale, bias) = (config.scale, config.bias);
        Ok(quote! {
            let #out = ops::scale_bias(#x, #scale, #bias);
        })
    }

    fn register_imports(&self, imports: &mut Imports<'_>) {
        // `crate::` because the generated file is include!-ed into THIS crate.
        // Imports are deduplicated, so every hook can register what it needs.
        imports.register("crate::ops");
    }
}

// ---------------------------------------------------------------------------
// Custom op 2: example.custom::ChannelScale - reads a constant input
// ---------------------------------------------------------------------------

/// `y = x * scale`, where `scale` is a constant initializer input rather than
/// an attribute. Constant inputs are readable at build time through
/// `Argument::value()`, so the values are inlined into the generated code and
/// never become model weights.
struct ChannelScale;

impl ChannelScale {
    fn scale_values(node: &CustomNode) -> Result<Vec<f32>, ProcessError> {
        node.inputs
            .get(1)
            .and_then(|arg| arg.value())
            .ok_or_else(|| {
                ProcessError::Custom("ChannelScale requires a constant scale input".to_string())
            })?
            .to_vec::<f32>()
            .map_err(|_| ProcessError::Custom("ChannelScale scale must be f32".to_string()))
    }
}

impl CustomOp for ChannelScale {
    fn op_type(&self) -> &str {
        "ChannelScale"
    }

    fn domain(&self) -> &str {
        "example.custom"
    }

    fn infer_output_types(&self, node: &CustomNode) -> Result<Vec<ArgType>, ProcessError> {
        // Checking the constant here means a dynamic (non-constant) scale is
        // reported as a parse error instead of failing later in codegen.
        Self::scale_values(node)?;
        let input = node
            .inputs
            .first()
            .ok_or_else(|| ProcessError::MissingInput("x".to_string()))?;
        Ok(vec![input.ty.clone()])
    }

    fn forward(
        &self,
        node: &CustomNode,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError> {
        let scale = Self::scale_values(node)?;
        let x = ctx.arg(&node.inputs[0]);
        let out = arg_to_ident(&node.outputs[0]);
        // `self.device` is reachable: the generated forward is a method on the
        // model struct.
        Ok(quote! {
            let #out = ops::channel_scale(#x, &[#(#scale),*], &self.device);
        })
    }

    fn register_imports(&self, imports: &mut Imports<'_>) {
        imports.register("crate::ops");
    }
}

// ---------------------------------------------------------------------------
// Override: route the built-in Sigmoid to our own kernel
// ---------------------------------------------------------------------------

/// Replaces the generated code for every `Sigmoid` node. Type inference still
/// comes from the built-in processor, so an override cannot change shapes or
/// dtypes - only the emitted code.
struct SigmoidOverride;

impl OpOverride for SigmoidOverride {
    fn target(&self) -> NodeType {
        NodeType::Sigmoid
    }

    fn forward(
        &self,
        node: &Node,
        ctx: &mut CodegenContext<'_, '_>,
    ) -> Result<TokenStream, ProcessError> {
        let Node::Sigmoid(sigmoid) = node else {
            return Err(ProcessError::Custom(
                "SigmoidOverride matched a non-Sigmoid node".to_string(),
            ));
        };
        let x = ctx.arg(&sigmoid.inputs[0]);
        let out = arg_to_ident(&sigmoid.outputs[0]);
        Ok(quote! {
            let #out = ops::fast_sigmoid(#x);
        })
    }

    fn register_imports(&self, imports: &mut Imports<'_>) {
        imports.register("crate::ops");
    }
}
