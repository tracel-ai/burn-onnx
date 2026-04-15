//! Parser for `expectations.toml` — the declarative source of truth for
//! how each upstream node test is expected to behave.
//!
//! The schema intentionally over-allocates statuses: only `Pass` is
//! exercised by the runner today, but the other variants are parsed so
//! that future PRs widening coverage do not need to revisit the parser
//! shape.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Top-level expectations file. Each map key is the upstream test name
/// (e.g. `test_abs`); each value declares the expected outcome.
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct Expectations {
    pub entries: BTreeMap<String, Entry>,
}

/// One row of `expectations.toml`. Only `status` is required; the rest
/// are documented as the "optional fields" contract in the TOML header.
#[derive(Debug, Clone, Deserialize)]
pub struct Entry {
    pub status: Status,
    /// Free-form explanation of why the test is in this state.
    #[serde(default)]
    pub reason: Option<String>,
    /// Linked tracking issue or PR (e.g. `#314`).
    #[serde(default)]
    pub tracking: Option<String>,
    /// `true` means we will not fix this (out of scope, upstream-only
    /// dtype, etc.). Defaults to `false` — i.e. an intentional gap we
    /// plan to close.
    #[serde(default)]
    pub wontfix: bool,
}

/// Declared expected outcome of a single upstream test.
///
/// `Pass` is the only status the runner exercises today. The remaining
/// variants are parsed but treated as "skip this test for now"; the
/// runner enforces this by panicking if any non-`Pass` entry appears
/// before the corresponding harness branch is wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Codegen succeeds, compile succeeds, output matches reference.
    Pass,
    /// `onnx2burn` panics or refuses the model (unsupported op, etc.).
    SkipCodegen,
    /// Codegen succeeds but the generated Rust does not compile.
    SkipCompile,
    /// Compiles and runs but produces incorrect output.
    FailCompare,
    /// Intermittent — do not gate CI on it.
    Flaky,
}

impl Expectations {
    /// Parse an `expectations.toml` file from disk.
    pub fn load(path: &Path) -> Result<Self, ExpectationsError> {
        let text = std::fs::read_to_string(path).map_err(|source| ExpectationsError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        toml::from_str(&text).map_err(|source| ExpectationsError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Look up the expected outcome for a given test name. Returns
    /// `None` if the file has no entry for `test_name` (i.e. the test
    /// is unknown to the expectations file, *not* that the test exists
    /// without an opinion).
    pub fn get(&self, test_name: &str) -> Option<&Entry> {
        self.entries.get(test_name)
    }
}

#[derive(Debug, Error)]
pub enum ExpectationsError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("toml parse error in {path:?}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip every status variant plus all optional fields. Locks
    /// in the `#[serde(rename_all = "kebab-case")]` mapping so renaming
    /// a `Status` variant without updating the TOML keys fails loudly.
    #[test]
    fn parses_all_statuses_and_optional_fields() {
        let toml = r##"
[test_pass]
status = "pass"

[test_skip_codegen]
status = "skip-codegen"
reason = "unsupported op"
tracking = "#314"

[test_skip_compile]
status = "skip-compile"

[test_fail_compare]
status = "fail-compare"
reason = "wrong output"
tracking = "#311"
wontfix = false

[test_flaky]
status = "flaky"
reason = "race in init"

[test_wontfix]
status = "skip-codegen"
reason = "exotic dtype"
wontfix = true
"##;
        let parsed: Expectations = toml::from_str(toml).expect("toml parse");
        assert_eq!(parsed.entries.len(), 6);
        assert_eq!(parsed.get("test_pass").unwrap().status, Status::Pass);
        assert_eq!(
            parsed.get("test_skip_codegen").unwrap().status,
            Status::SkipCodegen
        );
        assert_eq!(
            parsed.get("test_skip_compile").unwrap().status,
            Status::SkipCompile
        );
        assert_eq!(
            parsed.get("test_fail_compare").unwrap().status,
            Status::FailCompare
        );
        assert_eq!(parsed.get("test_flaky").unwrap().status, Status::Flaky);

        let wontfix = parsed.get("test_wontfix").unwrap();
        assert!(wontfix.wontfix);
        assert_eq!(wontfix.reason.as_deref(), Some("exotic dtype"));

        let fail = parsed.get("test_fail_compare").unwrap();
        assert!(!fail.wontfix);
        assert_eq!(fail.tracking.as_deref(), Some("#311"));

        // Default values when fields are omitted.
        let pass = parsed.get("test_pass").unwrap();
        assert!(!pass.wontfix);
        assert!(pass.reason.is_none());
        assert!(pass.tracking.is_none());
    }

    /// A typo'd status (e.g. `passes` instead of `pass`) must be
    /// rejected at parse time, not silently coerced or defaulted.
    #[test]
    fn rejects_unknown_status_value() {
        let toml = r#"
[test_typo]
status = "passes"
"#;
        let result: Result<Expectations, _> = toml::from_str(toml);
        assert!(
            result.is_err(),
            "expected parse error for unknown status, got Ok"
        );
    }
}
