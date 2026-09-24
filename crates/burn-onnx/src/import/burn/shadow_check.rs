//! Guards generated code against a temporary shadowing a graph value.
//!
//! Graph values (the `forward()` parameters and every node output) keep their
//! sanitized ONNX names, so a temporary that node codegen declares can shadow
//! one that the same scope reads afterwards: `let k = ...; input.topk(k)` is
//! wrong when the data input itself is named `k`. Nothing in the emitted
//! tokens tells a temporary apart from a graph value, so every graph value
//! reference built through `arg_ident`, `arg_to_ident` or `scope.arg()` is
//! emitted with a tag ([`value_ident`]), each slice of the assembled body is
//! walked with a scope stack ([`Checker`]), and the tag is removed before the
//! code is written out ([`strip`]).
//!
//! The load-bearing rule is that no temporary starts with the tag: a tagged
//! pattern binding is taken to be a graph value, and `strip` would rename the
//! temporary. Graph value names need no constraint, because a name that
//! already carries the tag is tagged twice and rounds back through `strip`.
//!
//! An `Ident` built directly from `arg.name` (`Ident::new`, `format_ident!`)
//! carries no tag. As a binding it counts as a temporary; as a read it is
//! reported when the name currently belongs to a graph value, which is how a
//! bypass of `scope.arg()` / `arg_to_ident()` shows up.

use core::fmt;
use onnx_ir::Argument;
use proc_macro2::{Group, Ident, Span, TokenStream, TokenTree};
use quote::quote;
use syn::visit::Visit;

const TAG: &str = "__arg_";

/// The identifier that generated code uses to refer to the graph value `name`.
///
/// Only splice it into tokens: the tag is stripped when the file is written,
/// so stringifying it or deriving another identifier from it leaks the tag.
pub(crate) fn value_ident(name: &str) -> Ident {
    assert!(
        !name.is_empty(),
        "codegen referenced a graph value with an empty name; an optional input was read \
         without an is_optional() guard"
    );
    Ident::new(&format!("{TAG}{name}"), Span::call_site())
}

/// Whether `name` is the text of a tagged identifier, i.e. was stringified
/// from one instead of taken from `arg.name`.
pub(crate) fn is_tagged_name(name: &str) -> bool {
    name.starts_with(TAG)
}

/// Replace every tagged identifier with the plain graph value name.
pub(crate) fn strip(tokens: TokenStream) -> TokenStream {
    tokens
        .into_iter()
        .map(|tree| match tree {
            TokenTree::Ident(ident) => match ident.to_string().strip_prefix(TAG) {
                Some(name) => TokenTree::Ident(Ident::new(name, ident.span())),
                None => TokenTree::Ident(ident),
            },
            TokenTree::Group(group) => {
                let mut stripped = Group::new(group.delimiter(), strip(group.stream()));
                stripped.set_span(group.span());
                TokenTree::Group(stripped)
            }
            other => other,
        })
        .collect()
}

/// What [`Checker::check`] found wrong with a slice of generated code.
///
/// `site` is where the offending read happens, e.g. "node `topk1` (TopK)" or
/// "the forward() return".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CheckError {
    /// A graph value was read while a temporary of the same name is in scope.
    Shadowed {
        site: String,
        name: String,
        /// The site that declared the temporary, when it is not `site`.
        declared_in: Option<String>,
    },
    /// A plain identifier that currently names a graph value was read, so the
    /// reference did not go through `scope.arg()` or `arg_to_ident()`.
    Untagged { site: String, name: String },
    /// The slice does not parse as Rust statements.
    Unparsable { site: String, error: String },
}

impl CheckError {
    /// The graph value involved, for tests that pin which name was caught.
    #[cfg(test)]
    pub(crate) fn name(&self) -> Option<&str> {
        match self {
            CheckError::Shadowed { name, .. } | CheckError::Untagged { name, .. } => Some(name),
            CheckError::Unparsable { .. } => None,
        }
    }
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckError::Shadowed {
                site,
                name,
                declared_in,
            } => {
                write!(
                    f,
                    "generated code for {site} reads the graph value `{name}` while a \
                     temporary named `{name}`"
                )?;
                if let Some(declared_in) = declared_in {
                    write!(f, " declared by {declared_in}")?;
                }
                write!(
                    f,
                    " is in scope. This is a burn-onnx codegen bug: the temporary needs another \
                     name in that node's codegen. Renaming the ONNX value (its sanitized name is \
                     shown) works around it."
                )
            }
            CheckError::Untagged { site, name } => write!(
                f,
                "generated code for {site} reads `{name}` as a plain identifier while `{name}` \
                 names a graph value. Graph values must be referenced through scope.arg() or \
                 arg_to_ident(), not an Ident built from the name."
            ),
            CheckError::Unparsable { site, error } => write!(
                f,
                "generated code for {site} does not parse as Rust statements: {error}"
            ),
        }
    }
}

/// What a name in scope currently resolves to.
///
/// A tagged binding (`let __arg_x = ...`) is a graph value; a plain one is a
/// temporary. Later bindings shadow earlier ones, so a graph value bound after
/// a same-named temporary makes the name safe to read again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binding {
    Value,
    Temporary,
}

/// The name an identifier binds or reads, and which kind of binding it is.
fn classify(ident: &Ident) -> (String, Binding) {
    let name = ident.to_string();
    match name.strip_prefix(TAG) {
        Some(value) => (value.to_string(), Binding::Value),
        None => (name, Binding::Temporary),
    }
}

#[derive(Debug, Clone)]
struct Bound {
    name: String,
    binding: Binding,
    /// The site whose code made the binding.
    site: String,
}

type Scope = Vec<Bound>;

/// Walks each slice of one `forward()` body in order, carrying the bindings
/// made at function scope from one slice to the next.
///
/// Slices must be checked in the order they are emitted, and every slice that
/// can read a graph value must be checked: a node's statements, the boundary
/// conversions, and the return.
#[derive(Debug, Default)]
pub(crate) struct Checker {
    function_scope: Scope,
}

impl Checker {
    /// A checker for a `forward()` taking `params`, which are graph values
    /// bound before any slice runs.
    pub(crate) fn for_params(params: &[Argument]) -> Self {
        let function_scope = params
            .iter()
            .map(|param| Bound {
                name: param.name.clone(),
                binding: Binding::Value,
                site: "the forward() parameters".to_string(),
            })
            .collect();
        Self { function_scope }
    }

    /// Check the statements of one site, which follow every site checked before.
    pub(crate) fn check(&mut self, site: &str, body: &TokenStream) -> Result<(), CheckError> {
        let block: syn::Block =
            syn::parse2(quote! { { #body } }).map_err(|error| CheckError::Unparsable {
                site: site.to_string(),
                error: error.to_string(),
            })?;
        let mut walk = Walk {
            site,
            function_scope: &mut self.function_scope,
            scopes: Vec::new(),
            error: None,
        };
        // The outer block is this site's slice of forward(), not a scope of its own.
        syn::visit::visit_block(&mut walk, &block);
        match walk.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

struct Walk<'a> {
    site: &'a str,
    function_scope: &'a mut Scope,
    scopes: Vec<Scope>,
    /// The first problem found; later reads are not inspected.
    error: Option<CheckError>,
}

impl Walk<'_> {
    fn declare(&mut self, name: String, binding: Binding) {
        let bound = Bound {
            name,
            binding,
            site: self.site.to_string(),
        };
        match self.scopes.last_mut() {
            Some(scope) => scope.push(bound),
            None => self.function_scope.push(bound),
        }
    }

    /// The innermost, latest binding of `name`, as Rust would resolve it.
    fn resolve(&self, name: &str) -> Option<&Bound> {
        self.scopes
            .iter()
            .rev()
            .chain(core::iter::once(&*self.function_scope))
            .find_map(|scope| scope.iter().rev().find(|bound| bound.name == name))
    }

    /// A tagged read must resolve to a graph value and a plain read to a
    /// temporary. An unbound name is not a read of anything tracked here.
    fn read(&mut self, ident: &Ident) {
        if self.error.is_some() {
            return;
        }
        let (name, expected) = classify(ident);
        let Some(bound) = self.resolve(&name) else {
            return;
        };
        if bound.binding == expected {
            return;
        }
        let site = self.site.to_string();
        self.error = Some(match expected {
            Binding::Value => CheckError::Shadowed {
                site,
                name,
                declared_in: (bound.site != self.site).then(|| bound.site.clone()),
            },
            Binding::Temporary => CheckError::Untagged { site, name },
        });
    }

    /// Declare every identifier the pattern binds.
    fn bind(&mut self, pat: &syn::Pat) {
        let mut names = PatNames::default();
        names.visit_pat(pat);
        for (name, binding) in names.0 {
            self.declare(name, binding);
        }
    }

    fn scoped(&mut self, f: impl FnOnce(&mut Self)) {
        self.scopes.push(Vec::new());
        f(self);
        self.scopes.pop();
    }

    /// Macro bodies are opaque to syn; scan them for reads. Bindings made
    /// inside a macro body are not tracked.
    fn read_tokens(&mut self, tokens: TokenStream) {
        for tree in tokens {
            match tree {
                TokenTree::Ident(ident) => self.read(&ident),
                TokenTree::Group(group) => self.read_tokens(group.stream()),
                _ => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for Walk<'_> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        self.scoped(|walk| syn::visit::visit_block(walk, block));
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        // The initializer runs before the binding exists.
        if let Some(init) = &local.init {
            self.visit_expr(&init.expr);
            if let Some((_, diverge)) = &init.diverge {
                self.visit_expr(diverge);
            }
        }
        self.bind(&local.pat);
    }

    fn visit_expr_let(&mut self, expr: &'ast syn::ExprLet) {
        self.visit_expr(&expr.expr);
        self.bind(&expr.pat);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        // An `if let` binding is visible in the then branch only.
        self.scoped(|walk| {
            walk.visit_expr(&expr.cond);
            walk.visit_block(&expr.then_branch);
        });
        if let Some((_, else_branch)) = &expr.else_branch {
            self.visit_expr(else_branch);
        }
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.scoped(|walk| {
            walk.visit_expr(&expr.cond);
            walk.visit_block(&expr.body);
        });
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.visit_expr(&expr.expr);
        self.scoped(|walk| {
            walk.bind(&expr.pat);
            walk.visit_block(&expr.body);
        });
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.scoped(|walk| {
            for input in &expr.inputs {
                walk.bind(input);
            }
            walk.visit_expr(&expr.body);
        });
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        self.scoped(|walk| {
            walk.bind(&arm.pat);
            // In syn 3 a match guard is part of the pattern (`Pat::Guard`), so
            // visiting the pattern reaches the reads inside it.
            walk.visit_pat(&arm.pat);
            walk.visit_expr(&arm.body);
        });
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && path.path.leading_colon.is_none()
            && path.path.segments.len() == 1
        {
            self.read(&path.path.segments[0].ident);
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.read_tokens(mac.tokens.clone());
    }
}

/// Collects the identifiers a pattern binds, tagged ones as graph values.
#[derive(Default)]
struct PatNames(Vec<(String, Binding)>);

impl<'ast> Visit<'ast> for PatNames {
    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        self.0.push(classify(&pat.ident));
        syn::visit::visit_pat_ident(self, pat);
    }

    fn visit_pat_guard(&mut self, pat: &'ast syn::PatGuard) {
        // A guard is an expression; anything it binds (closure parameters) is
        // scoped to the guard, not to the arm.
        self.visit_pat(&pat.pat);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::ir::{ArgType, TensorType};

    fn check(body: TokenStream) -> Result<(), CheckError> {
        Checker::default().check("node1", &body)
    }

    fn shadowed(name: &str) -> Result<(), CheckError> {
        Err(CheckError::Shadowed {
            site: "node1".to_string(),
            name: name.to_string(),
            declared_in: None,
        })
    }

    #[test]
    fn temporary_declared_before_graph_value_read() {
        let k = value_ident("k");
        let x = value_ident("x");
        let body = quote! {
            let out = {
                let k: usize = 3;
                #x.topk(k)
            };
        };
        assert_eq!(check(body), Ok(()));

        let body = quote! {
            let out = {
                let k: usize = 3;
                #k.topk(k)
            };
        };
        assert_eq!(check(body), shadowed("k"));
    }

    #[test]
    fn graph_value_rebound_into_temporary_of_same_name() {
        let indices = value_ident("indices");
        let body = quote! {
            let out = {
                let indices = #indices.cast(I64);
                let negative = indices.clone().lower_elem(0i64);
                indices.mask_where(negative)
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn sequential_capture_of_swapped_inputs() {
        let lhs = value_ident("lhs");
        let rhs = value_ident("rhs");
        let body = quote! {
            let out = {
                let lhs = #rhs;
                let rhs = #lhs;
                lhs.add(rhs)
            };
        };
        assert_eq!(check(body), shadowed("lhs"));

        let body = quote! {
            let out = {
                let (lhs, rhs) = (#rhs, #lhs);
                lhs.add(rhs)
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn block_scope_ends_with_the_block() {
        let dims = value_ident("dims");
        let body = quote! {
            let out1 = {
                let dims = [1usize, 2usize];
                dims[0]
            };
            let out2 = #dims.reshape([1]);
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn function_scope_temporary_reaches_later_sites() {
        let actual_idx = value_ident("actual_idx");
        let mut checker = Checker::default();
        let first = quote! {
            let actual_idx = if 1 < 0 { 0usize } else { 1usize };
            let out1 = shape[actual_idx];
        };
        assert_eq!(checker.check("gather1", &first), Ok(()));
        let second = quote! {
            let out2 = #actual_idx.abs();
        };
        assert_eq!(
            checker.check("abs1", &second),
            Err(CheckError::Shadowed {
                site: "abs1".to_string(),
                name: "actual_idx".to_string(),
                declared_in: Some("gather1".to_string()),
            })
        );
    }

    #[test]
    fn graph_value_bound_after_a_temporary_takes_the_name_back() {
        let actual_idx = value_ident("actual_idx");
        let mut checker = Checker::default();
        let first = quote! {
            let actual_idx = 1usize;
            let out1 = shape[actual_idx];
        };
        assert_eq!(checker.check("gather1", &first), Ok(()));
        let second = quote! {
            let #actual_idx = out1.abs();
            let out2 = #actual_idx.abs();
        };
        assert_eq!(checker.check("abs1", &second), Ok(()));
    }

    #[test]
    fn for_loop_binding_is_scoped_to_its_body() {
        let i = value_ident("i");
        let body = quote! {
            for i in 0..3usize {
                let _ = #i;
            }
        };
        assert_eq!(check(body), shadowed("i"));

        let body = quote! {
            for i in 0..3usize {
                let _ = i;
            }
            let after = #i;
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn closure_parameter_is_scoped_to_its_body() {
        let v = value_ident("v");
        let body = quote! {
            let mapped = [1i64].map(|v| v + #v);
        };
        assert_eq!(check(body), shadowed("v"));

        let body = quote! {
            let mapped = [1i64].map(|v| v + 1i64);
            let after = #v;
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn match_arm_binding_is_scoped_to_its_arm() {
        let t = value_ident("t");
        let body = quote! {
            let picked = match Some(1i64) {
                Some(t) => #t,
                _ => 0i64,
            };
        };
        assert_eq!(check(body), shadowed("t"));

        let body = quote! {
            let picked = match Some(1i64) {
                Some(t) => t,
                _ => #t,
            };
            let after = #t;
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn reads_inside_match_guards_are_checked() {
        let x = value_ident("x");
        let body = quote! {
            let out = {
                let x = 1i64;
                match Some(2i64) {
                    Some(v) if v > #x => v,
                    _ => x,
                }
            };
        };
        assert_eq!(check(body), shadowed("x"));
    }

    #[test]
    fn closure_inside_a_guard_does_not_bind_into_the_arm() {
        let w = value_ident("w");
        let body = quote! {
            let out = match Some(2i64) {
                Some(v) if [1i64].iter().any(|w| *w == v) => #w,
                _ => 0i64,
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn if_let_binding_is_scoped_to_then_branch() {
        let x = value_ident("x");
        let body = quote! {
            let out = if let Some(x) = Some(1i64) { x } else { #x };
            let after = #x;
        };
        assert_eq!(check(body), Ok(()));

        let body = quote! {
            let out = if let Some(x) = Some(1i64) { #x } else { 0i64 };
        };
        assert_eq!(check(body), shadowed("x"));
    }

    #[test]
    fn while_let_binding_is_scoped_to_its_body() {
        let x = value_ident("x");
        let body = quote! {
            let mut it = [1i64].into_iter();
            while let Some(x) = it.next() {
                let _ = #x;
            }
        };
        assert_eq!(check(body), shadowed("x"));

        let body = quote! {
            let mut it = [1i64].into_iter();
            while let Some(x) = it.next() {
                let _ = x;
            }
            let after = #x;
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn let_else_binds_after_its_initializer_and_diverge() {
        let x = value_ident("x");
        let body = quote! {
            let Some(x) = Some(#x) else { return #x; };
            let after = #x;
        };
        assert_eq!(check(body), shadowed("x"));

        let body = quote! {
            let Some(x) = Some(#x) else { return #x; };
            let after = x;
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn graph_value_rebinding_is_not_a_temporary() {
        let cond = value_ident("cond");
        let body = quote! {
            let out = if true {
                let #cond = #cond;
                #cond
            } else {
                #cond
            };
        };
        assert_eq!(check(body), Ok(()));
    }

    #[test]
    fn reads_inside_macros_are_checked() {
        let delta = value_ident("delta");
        let body = quote! {
            let out = {
                let delta = 1i64;
                assert!(#delta != 0);
                delta
            };
        };
        assert_eq!(check(body), shadowed("delta"));

        let body = quote! {
            let out = {
                let delta = 1i64;
                alloc::vec![(delta, [#delta])]
            };
        };
        assert_eq!(check(body), shadowed("delta"));
    }

    #[test]
    fn untagged_read_of_a_graph_value_is_reported() {
        let x = value_ident("x");
        let params = [Argument::new(
            "x",
            ArgType::Tensor(TensorType::new(burn::tensor::DType::F32, 2, None)),
        )];
        let body = quote! { let out = x.abs(); };
        assert_eq!(
            Checker::for_params(&params).check("node1", &body),
            Err(CheckError::Untagged {
                site: "node1".to_string(),
                name: "x".to_string(),
            })
        );

        // A temporary named like a parameter shadows it, so a plain read is the temporary.
        let body = quote! {
            let out = {
                let x = #x.abs();
                x.neg()
            };
        };
        assert_eq!(Checker::for_params(&params).check("node1", &body), Ok(()));
    }

    #[test]
    fn unparsable_slice_is_an_error() {
        let body = quote! { let out = ; };
        assert!(matches!(
            check(body),
            Err(CheckError::Unparsable { site, .. }) if site == "node1"
        ));
    }

    #[test]
    fn strip_removes_the_tag_everywhere() {
        let x = value_ident("x");
        let out = value_ident("out");
        let tokens = quote! {
            let #out = { alloc::vec![#x.clone(), (#x)] };
        };
        assert_eq!(
            strip(tokens).to_string(),
            quote! { let out = { alloc::vec![x.clone(), (x)] }; }.to_string()
        );
    }

    #[test]
    #[should_panic(expected = "empty name")]
    fn empty_value_name_panics_at_the_reference() {
        value_ident("");
    }
}
