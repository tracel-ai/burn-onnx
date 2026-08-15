//! Public extension surface for custom op hooks.
//!
//! This module is the single entry point for user crates that extend the
//! ONNX import with custom operators. It re-exports the `onnx-ir` types the
//! hook traits are expressed in, the token-stream crates used to generate
//! code, and thin wrappers over the internal codegen state.
//!
//! The token-stream crates (`proc_macro2`, `quote`) are re-exported so hook
//! implementations build `TokenStream` values from the same crate build that
//! `burn-onnx` links against.

pub use crate::burn::custom_op::{CustomOp, OpOverride};
pub use crate::burn::node_traits::{Field, create_lazy_snapshot};

/// Convert an argument's name to an identifier.
///
/// For OUTPUTS and host-side values only. Never use it for `Tensor` /
/// `ScalarTensor` inputs: it bypasses clone tracking, producing generated
/// code that moves a tensor still needed elsewhere. Inputs go through
/// [`CodegenContext::arg`] instead.
pub use crate::burn::node_traits::arg_to_ident;

pub use onnx_ir::{
    ArgType, Argument, AttrKind, CustomNode, DType, Node, NodeType, OpsetRange, ProcessError,
    TensorData, TensorType,
};

pub use burn_store::TensorSnapshot;

pub use proc_macro2;
pub use quote;

/// Codegen context passed to a custom op's forward implementation.
///
/// Wraps the internal scope tracking behind the one operation a hook needs:
/// resolving an input argument to a token stream.
pub struct CodegenContext<'a, 'b> {
    inner: &'a mut crate::burn::ScopeAtPosition<'b>,
}

impl<'a, 'b> CodegenContext<'a, 'b> {
    pub(crate) fn wrap(inner: &'a mut crate::burn::ScopeAtPosition<'b>) -> Self {
        Self { inner }
    }

    /// Resolve an input argument to a token stream.
    ///
    /// Handles clone tracking for on-device values (`Tensor`, `ScalarTensor`)
    /// and bare identifiers for host values (`ScalarNative`, `Shape`), exactly
    /// like the built-in nodes' `scope.arg()`.
    pub fn arg(&mut self, arg: &Argument) -> proc_macro2::TokenStream {
        self.inner.arg(arg)
    }
}

/// Import registry passed to a custom op's import registration.
///
/// Each path is emitted as a `use` statement in the generated model file,
/// deduplicated across all nodes.
pub struct Imports<'a> {
    inner: &'a mut crate::burn::BurnImports,
}

impl<'a> Imports<'a> {
    pub(crate) fn wrap(inner: &'a mut crate::burn::BurnImports) -> Self {
        Self { inner }
    }

    /// Register an import path, e.g. `my_crate::ops::fft`.
    pub fn register(&mut self, path: impl Into<String>) {
        self.inner.register(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{BurnImports, Scope};
    use onnx_ir::ir::TensorType;

    fn tensor_arg(name: &str) -> Argument {
        Argument::new(
            name,
            ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 2,
                static_shape: None,
            }),
        )
    }

    #[test]
    fn codegen_context_arg_tracks_clones() {
        let mut scope = Scope::default();
        let arg = tensor_arg("input1");
        scope.tensor_register_variable(&arg, 0);
        scope.tensor_register_future_use(&arg, 1);
        scope.tensor_register_future_use(&arg, 2);

        // Two future uses remain after the first: expect a clone
        let mut at_pos = scope.at_position(1);
        let mut ctx = CodegenContext { inner: &mut at_pos };
        assert_eq!(ctx.arg(&arg).to_string(), "input1 . clone ()");

        // Last use: moved, no clone
        let mut at_pos = scope.at_position(2);
        let mut ctx = CodegenContext { inner: &mut at_pos };
        assert_eq!(ctx.arg(&arg).to_string(), "input1");
    }

    #[test]
    fn codegen_context_arg_scalar_native_is_bare_ident() {
        let arg = Argument::new("alpha", ArgType::ScalarNative(DType::F32));
        let mut scope = Scope::default();
        let mut at_pos = scope.at_position(0);
        let mut ctx = CodegenContext { inner: &mut at_pos };
        assert_eq!(ctx.arg(&arg).to_string(), "alpha");
    }

    #[test]
    fn imports_register_dedupes() {
        let mut inner = BurnImports::default();
        {
            let mut imports = Imports { inner: &mut inner };
            imports.register("my_crate::ops");
            imports.register("my_crate::ops");
        }
        let code = inner.codegen().to_string();
        assert_eq!(code.matches("my_crate :: ops").count(), 1);
    }
}
