//! Local copy of the `onnx-official-tests` expectations.toml schema.
//!
//! This module deliberately duplicates `Expectations` / `Entry` /
//! `Status` from `crates/onnx-official-tests/src/expectations.rs` so
//! that the xtask subcommands (`diff-expectations`, `update-expectations`)
//! can parse the file without depending on the `onnx-official-tests`
//! crate — which would pull in its build script and re-run
//! `burn_onnx::ModelGen` over ~500 vendored models on every xtask
//! compile (~18s instead of ~3s).
//!
//! The schema is small and stable (four fields, five status variants)
//! so the duplication is cheap. A unit test asserts the kebab-case
//! status names the parser accepts, which catches the most likely
//! form of drift — someone renaming a variant in `onnx-official-tests`
//! without updating this module.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Top-level expectations file. One map entry per upstream test name.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Expectations {
    pub entries: BTreeMap<String, Entry>,
}

/// One row from `expectations.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Entry {
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracking: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub wontfix: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Declared expected outcome for a single upstream test. The ordering
/// of variants here is not load-bearing; the kebab-case rename
/// attribute is what the TOML file cares about. `Ord` / `PartialOrd`
/// are derived so the diff renderer can produce deterministic output
/// when sorting tuples like `(name, from_status, to_status)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    Pass,
    SkipCodegen,
    SkipCompile,
    FailCompare,
    Flaky,
}

impl Status {
    /// Human-readable name matching the TOML serialization exactly.
    /// Used for diff output so readers see the same spelling the
    /// file itself uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::SkipCodegen => "skip-codegen",
            Self::SkipCompile => "skip-compile",
            Self::FailCompare => "fail-compare",
            Self::Flaky => "flaky",
        }
    }
}

/// Errors produced by [`Expectations::from_toml`] and
/// [`Expectations::load`]. Kept narrow so callers can print the
/// source path alongside the diagnostic.
#[derive(Debug)]
pub enum ExpectationsError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

impl std::fmt::Display for ExpectationsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "io error reading {}: {source}", path.display())
            }
            Self::Parse { path, source } => {
                write!(f, "parse error in {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for ExpectationsError {}

impl Expectations {
    /// Parse from a TOML string with a caller-supplied `path` used
    /// only in error messages. This two-argument form exists so that
    /// [`diff_expectations`](crate::diff_expectations) can parse a
    /// TOML blob pulled from `git show` (which has no path on disk)
    /// while still producing a useful error if the parse fails.
    pub fn from_toml(path: PathBuf, text: &str) -> Result<Self, ExpectationsError> {
        toml::from_str(text).map_err(|source| ExpectationsError::Parse { path, source })
    }

    /// Convenience wrapper: read a file from disk and parse.
    pub fn load(path: &Path) -> Result<Self, ExpectationsError> {
        let text = std::fs::read_to_string(path).map_err(|source| ExpectationsError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml(path.to_path_buf(), &text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock in the kebab-case status strings. If someone renames a
    /// variant here without updating the TOML files, this fails
    /// first — much clearer than a parse error buried inside a
    /// subcommand run.
    #[test]
    fn status_kebab_case_roundtrip() {
        for (status, expected) in [
            (Status::Pass, "pass"),
            (Status::SkipCodegen, "skip-codegen"),
            (Status::SkipCompile, "skip-compile"),
            (Status::FailCompare, "fail-compare"),
            (Status::Flaky, "flaky"),
        ] {
            assert_eq!(status.as_str(), expected);
            let toml = format!("[t]\nstatus = \"{expected}\"\n");
            let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &toml).unwrap();
            assert_eq!(parsed.entries["t"].status, status);
        }
    }

    /// Verify all optional fields round-trip cleanly. Locks in the
    /// serde attributes (`default`, `skip_serializing_if`) so a
    /// no-op diff doesn't accidentally rewrite the file with empty
    /// `reason`/`tracking`/`wontfix` keys.
    #[test]
    fn optional_fields_roundtrip() {
        // Note the `r##"..."##` delimiter: the embedded `"#314"` would
        // end a `r#"..."#` raw string at the wrong place.
        let toml = r##"
[full]
status = "skip-codegen"
reason = "unsupported op"
tracking = "#314"
wontfix = true

[bare]
status = "pass"
"##;
        let parsed = Expectations::from_toml(PathBuf::from("t.toml"), toml).unwrap();
        let full = &parsed.entries["full"];
        assert_eq!(full.status, Status::SkipCodegen);
        assert_eq!(full.reason.as_deref(), Some("unsupported op"));
        assert_eq!(full.tracking.as_deref(), Some("#314"));
        assert!(full.wontfix);

        let bare = &parsed.entries["bare"];
        assert_eq!(bare.status, Status::Pass);
        assert!(bare.reason.is_none());
        assert!(bare.tracking.is_none());
        assert!(!bare.wontfix);
    }
}
