/// Implements NodeCodegen trait on onnx_ir::Node enum
/// Uses a simple macro to generate match arms for all supported nodes
use onnx_ir::{Argument, CustomNode, Node};
use proc_macro2::TokenStream;

use super::node_traits::NodeCodegen;
use crate::burn::custom_op::{CustomOp, HookRegistry};
use crate::burn::scope::ScopeAtPosition;
use crate::burn::{BurnImports, Field};
use crate::ext::{CodegenContext, Imports};
use burn_store::TensorSnapshot;

// ============================================================================
// Hook-aware dispatch
//
// The graph-level codegen goes through these functions instead of the plain
// NodeCodegen impl so Node::Custom can be routed to its registered hook.
// The NodeCodegen trait itself stays hook-free: per-node impls never see the
// registry, and neither do the structural accessors (inputs/outputs), because
// hooks cannot change the graph's wiring.
// ============================================================================

fn require_custom_hook<'r>(hooks: &'r HookRegistry, node: &CustomNode) -> &'r dyn CustomOp {
    hooks
        .custom_for(&node.op_type, &node.domain)
        .unwrap_or_else(|| {
            panic!(
                "Custom op '{}' (node '{}') has no registered hook; \
             register one via ModelGen::register_custom_op",
                node, node.name
            )
        })
}

/// Unwrap a hook result at the codegen boundary.
///
/// `BurnGraph::codegen` has no error channel, so hook errors surface as a
/// panic that names the node and what the hook was doing. Build scripts are
/// the caller, where an attributable panic is the failure channel.
fn expect_hook<T>(result: Result<T, onnx_ir::ProcessError>, what: &str, node_name: &str) -> T {
    result.unwrap_or_else(|e| panic!("Codegen hook failed in {what} for node '{node_name}': {e}"))
}

pub(crate) fn node_forward(
    node: &Node,
    scope: &mut ScopeAtPosition<'_>,
    hooks: &HookRegistry,
) -> TokenStream {
    if let Some(over) = hooks.override_for(&node.node_type()) {
        let mut ctx = CodegenContext::wrap(scope);
        return expect_hook(
            over.forward(node, &mut ctx),
            "OpOverride::forward",
            node.name(),
        );
    }
    if let Node::Custom(c) = node {
        let mut ctx = CodegenContext::wrap(scope);
        let hook = require_custom_hook(hooks, c);
        return expect_hook(hook.forward(c, &mut ctx), "CustomOp::forward", &c.name);
    }
    NodeCodegen::forward(node, scope)
}

pub(crate) fn node_field(node: &Node, hooks: &HookRegistry) -> Option<Field> {
    // The override wins even when the built-in declares a field: an override
    // with the default field() = Ok(None) suppresses the built-in field,
    // since the override's forward will not reference it.
    if let Some(over) = hooks.override_for(&node.node_type()) {
        return expect_hook(over.field(node), "OpOverride::field", node.name());
    }
    if let Node::Custom(c) = node {
        let hook = require_custom_hook(hooks, c);
        return expect_hook(hook.field(c), "CustomOp::field", &c.name);
    }
    NodeCodegen::field(node)
}

pub(crate) fn node_register_imports(node: &Node, imports: &mut BurnImports, hooks: &HookRegistry) {
    if let Some(over) = hooks.override_for(&node.node_type()) {
        over.register_imports(&mut Imports::wrap(imports));
        return;
    }
    if let Node::Custom(c) = node {
        require_custom_hook(hooks, c).register_imports(&mut Imports::wrap(imports));
        return;
    }
    NodeCodegen::register_imports(node, imports)
}

pub(crate) fn node_collect_snapshots(
    node: &Node,
    field_name: &str,
    hooks: &HookRegistry,
) -> Vec<TensorSnapshot> {
    if let Some(over) = hooks.override_for(&node.node_type()) {
        return expect_hook(
            over.collect_snapshots(node, field_name),
            "OpOverride::collect_snapshots",
            node.name(),
        );
    }
    if let Node::Custom(c) = node {
        let hook = require_custom_hook(hooks, c);
        return expect_hook(
            hook.collect_snapshots(c, field_name),
            "CustomOp::collect_snapshots",
            &c.name,
        );
    }
    NodeCodegen::collect_snapshots(node, field_name)
}

/// Macro to implement NodeCodegen on onnx_ir::Node by dispatching to individual node impls
///
/// `Node::Custom` is handled explicitly: its structural accessors (inputs/outputs)
/// read the CustomNode's own wiring. Hook-aware codegen goes through the free
/// dispatch functions above (`node_forward` etc.); this trait impl's `forward`
/// panics for `Node::Custom` because reaching it means hook dispatch was
/// bypassed - today that is only possible for nodes inside If/Loop/Scan
/// subgraph bodies, which `BurnGraph` rejects up front with a clearer error.
macro_rules! impl_node_codegen_dispatch {
    ($($variant:ident),* $(,)?) => {
        impl NodeCodegen for Node {
            fn inputs(&self) -> &[Argument] {
                match self {
                    Node::Custom(n) => &n.inputs,
                    $(Node::$variant(n) => n.inputs(),)*
                    _ => panic!("Unsupported node type for inputs: {:?}", self),
                }
            }

            fn outputs(&self) -> &[Argument] {
                match self {
                    Node::Custom(n) => &n.outputs,
                    $(Node::$variant(n) => n.outputs(),)*
                    _ => panic!("Unsupported node type for outputs: {:?}", self),
                }
            }

            fn forward(&self, scope: &mut crate::burn::scope::ScopeAtPosition<'_>) -> TokenStream {
                match self {
                    Node::Custom(n) => panic!(
                        "Custom op '{}' (node '{}') reached built-in codegen dispatch; \
                         custom op codegen inside If/Loop/Scan subgraph bodies is not \
                         supported yet",
                        n, n.name
                    ),
                    $(Node::$variant(n) => n.forward(scope),)*
                    _ => panic!("Unsupported node type for forward: {:?}", self),
                }
            }

            fn field(&self) -> Option<Field> {
                match self {
                    $(Node::$variant(n) => n.field(),)*
                    _ => None,
                }
            }

            fn register_imports(&self, imports: &mut BurnImports) {
                match self {
                    $(Node::$variant(n) => n.register_imports(imports),)*
                    _ => {}
                }
            }

            fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
                match self {
                    $(Node::$variant(n) => n.collect_snapshots(field_name),)*
                    _ => vec![],
                }
            }
        }
    };
}

// List all supported node types here
// Just add/remove variant names as needed - one place to maintain!
impl_node_codegen_dispatch! {
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MatMul,
    Einsum,

    // Comparison ops
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,

    // Boolean ops
    And,
    Or,
    Xor,

    // Unary ops
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Ceil,
    Cos,
    Cosh,
    Erf,
    Exp,
    Floor,
    Identity,
    Log,
    Neg,
    Not,
    Reciprocal,
    Round,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,

    // Activation ops
    Relu,
    Gelu,
    Mish,
    LeakyRelu,
    HardSigmoid,
    HardSwish,
    Softmax,
    LogSoftmax,
    PRelu,
    Celu,
    Elu,
    Selu,
    Softplus,
    Softsign,
    ThresholdedRelu,
    Swish,
    Hardmax,
    Shrink,

    // Shape ops
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Transpose,
    Shape,
    Size,

    // Tensor ops
    Concat,
    Split,
    Slice,
    Gather,
    GatherElements,
    GatherND,
    ScatterElements,
    ScatterND,
    Tile,
    Expand,
    Pad,

    // Convolution ops
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    DeformConv,
    Col2Im,

    // Pooling ops
    AveragePool1d,
    AveragePool2d,
    AveragePool3d,
    LpPool1d,
    LpPool2d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    GlobalAveragePool,

    // Normalization ops
    BatchNormalization,
    LayerNormalization,
    Lrn,
    GroupNormalization,
    InstanceNormalization,
    MeanVarianceNormalization,
    LpNormalization,

    // Other ops
    Cast,
    CastLike,
    Clip,
    CumSum,
    Dropout,
    Where,
    ArgMax,
    ArgMin,
    TopK,
    NonZero,
    OneHot,
    Pow,
    Mod,
    Trilu,

    // Bitwise ops
    BitShift,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,

    // Math ops
    Sum,
    Mean,
    Gemm,
    Linear,
    MatMulInteger,
    DequantizeLinear,
    QuantizeLinear,
    QLinearMatMul,

    // Constant ops
    Constant,
    ConstantOfShape,
    EyeLike,
    Range,

    // Random ops
    RandomNormal,
    RandomUniform,
    RandomNormalLike,
    RandomUniformLike,
    Bernoulli,

    // Spatial ops
    DepthToSpace,
    SpaceToDepth,
    Resize,
    GridSample,

    // Linear algebra ops
    Det,

    // Signal processing ops
    BlackmanWindow,
    Dft,
    HammingWindow,
    HannWindow,
    MelWeightMatrix,
    Stft,

    // Test ops
    IsInf,
    IsNaN,

    // ML ops
    Imputer,
    Scaler,
    SVMRegressor,

    // Special ops
    Attention,

    // Control flow ops
    If,
    Loop,
    Scan,

    // Recurrent neural network ops
    Lstm,
    Rnn,
    Gru,

    // Reduce ops (handled by ReduceNode in onnx-ir)
    ReduceMax,
    ReduceMin,
    ReduceMean,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
}
