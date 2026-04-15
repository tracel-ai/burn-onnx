//! `cargo xtask update-expectations` — run the onnx-official-tests
//! suite and rewrite `expectations.toml` in place to reflect reality.
//!
//! This is the M5 piece of tracel-ai/burn-onnx#315. The intended
//! local workflow is:
//!
//! 1. Fix a bug in burn-onnx (say, lift a restriction on a
//!    previously-failing op).
//! 2. Manually promote any tests you believe are now fixed by
//!    editing their rows in `expectations.toml` to `status = "pass"`.
//! 3. Run `cargo xtask update-expectations`.
//! 4. Commit the updated expectations alongside your fix.
//!
//! Step 3 does two things:
//!
//! * Runs `cargo test -p onnx-official-tests --release --no-fail-fast`
//!   and parses the per-test failure lines from stdout.
//! * For every failed test that is currently marked `pass`, rewrites
//!   the entry to `fail-compare` with a generated reason pointing at
//!   the refresh run. Other statuses are left alone — the caller has
//!   to manually classify build-time failures (`skip-codegen` vs
//!   `skip-compile`) because `cargo test` output does not distinguish
//!   them.
//!
//! The command is intentionally demote-only in this direction.
//! Promotions flow through manual edits (step 2 above) because
//! automated promotion would require trying every `skip-codegen` /
//! `fail-compare` entry against the full pipeline, which is
//! prohibitively expensive at ~1600 tests.
//!
//! `--dry-run` prints the planned changes without writing the file.

use std::path::PathBuf;

use tracel_xtask::prelude::*;

use crate::expectations_schema::{Expectations, Status};

/// Arguments for the `update-expectations` subcommand.
#[derive(clap::Args)]
pub struct UpdateExpectationsArgs {
    /// Skip the actual file-system mutation and just print the planned
    /// changes. Useful for previewing what a test-run would rewrite
    /// before committing.
    #[arg(long)]
    pub dry_run: bool,

    /// Build in debug mode instead of release. Off by default because
    /// the release-mode test suite matches CI's behaviour and is what
    /// the `#311`/`#314` classifications were computed against.
    #[arg(long)]
    pub debug: bool,

    /// Optional `tracking` value to embed in each demoted entry, e.g.
    /// `"#321"` to link the demotion to a specific issue or PR. When
    /// omitted (the default), rewritten entries have no `tracking`
    /// field at all — the demotion is self-documenting via its
    /// `reason`, and hard-coding a stale issue reference would be
    /// actively misleading. Pass a new value each time the demotion
    /// batch is tied to a specific PR.
    #[arg(long)]
    pub tracking: Option<String>,
}

/// Top-level dispatcher. Runs the test suite, parses failures,
/// computes the demotion set, and either applies or reports it.
pub fn handle_command(args: UpdateExpectationsArgs) -> anyhow::Result<()> {
    let repo_root = repo_root();
    let expectations_path = repo_root.join("crates/onnx-official-tests/expectations.toml");

    info!("Expectations file: {}", expectations_path.display());
    info!("Running onnx-official-tests to capture failures...");

    let failing = run_tests_and_collect_failures(args.debug)?;
    info!("Captured {} failing test(s)", failing.len());

    // Load the current expectations as a raw string so we can apply
    // surgical line edits that preserve comments, section ordering,
    // and blank lines. Round-tripping through serde would lose all
    // of that. We also parse a typed `Expectations` separately so we
    // can identify which failing tests are currently marked `pass`
    // and thus candidates for demotion.
    let current_text = std::fs::read_to_string(&expectations_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", expectations_path.display()))?;
    let parsed = Expectations::from_toml(expectations_path.clone(), &current_text)
        .map_err(|e| anyhow::anyhow!("parse expectations: {e}"))?;

    let demote_candidates: Vec<String> = failing
        .into_iter()
        .filter(|name| {
            parsed
                .entries
                .get(name)
                .map(|e| e.status == Status::Pass)
                .unwrap_or(false)
        })
        .collect();

    if demote_candidates.is_empty() {
        info!("No pass-listed entries to demote; expectations.toml is already in sync.");
        return Ok(());
    }

    // The demotion plan goes through `info!` so verbosity can be
    // controlled consistently with the rest of the xtask output.
    info!(
        "Demoting {} pass entry/entries to fail-compare:",
        demote_candidates.len()
    );
    for name in &demote_candidates {
        info!("  - {name}");
    }

    if args.dry_run {
        warn!("--dry-run set; no files will be modified");
        return Ok(());
    }

    let new_text = apply_demotions(&current_text, &demote_candidates, args.tracking.as_deref());
    std::fs::write(&expectations_path, new_text)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", expectations_path.display()))?;
    info!(
        "Rewrote {} with {} demotion(s).",
        expectations_path.display(),
        demote_candidates.len()
    );

    Ok(())
}

/// Run the test suite and return the names of failing tests.
///
/// Runs `cargo test -p onnx-official-tests [--release] --no-fail-fast`
/// and scans stdout for lines matching `test NAME ... FAILED`. This
/// relies on libtest's stable output format; the JSON alternative
/// (`--format json --unstable-options`) is nightly-only and would
/// make the command useless for the stable toolchain CI uses.
///
/// Build-phase failures (panics in build.rs or rustc errors in
/// generated code) don't produce per-test FAILED lines; they show up
/// as an overall command failure. In that case we print a hint and
/// return an error so the user can triage manually.
fn run_tests_and_collect_failures(debug: bool) -> anyhow::Result<Vec<String>> {
    // `--no-fail-fast` is a cargo flag (not a libtest flag), so it
    // goes *before* the `--` separator. Passing it after `--` would
    // forward it to libtest, which doesn't know about it and errors
    // out before running any test.
    let mut args = vec!["test", "-p", "onnx-official-tests", "--no-fail-fast"];
    if !debug {
        args.push("--release");
    }

    // We run cargo directly (not via `run_process`) because we need
    // to capture stdout and parse it. `run_process` inherits stdout
    // from the parent, which would require piping through tee to a
    // temp file — avoidable complexity.
    let output = std::process::Command::new("cargo")
        .args(&args)
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn cargo: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let failures: Vec<String> = stdout
        .lines()
        .filter_map(parse_failure_line)
        .map(|s| s.to_string())
        .collect();

    // If the command failed but we parsed zero FAILED lines, the test
    // binary never ran — likely a build-time issue. Tell the user
    // what happened so they can fix the root cause manually.
    if !output.status.success() && failures.is_empty() {
        error!("cargo test failed before any test ran");
        error!("stderr tail:");
        for line in stderr
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .iter()
            .rev()
        {
            error!("  {line}");
        }
        return Err(anyhow::anyhow!(
            "cargo test failed during build phase; classify the affected tests as \
             skip-codegen or skip-compile manually and re-run"
        ));
    }

    // De-duplicate in case libtest's retries ever produce the same
    // test name twice, and sort so the order is reproducible.
    let mut unique: Vec<String> = failures
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    unique.sort();
    Ok(unique)
}

/// Parse a single libtest output line of the form
/// `test NAME ... FAILED`, returning the test name or `None` if the
/// line doesn't match. Supports the case where libtest prefixes the
/// name with the module path (e.g. `test test_mod::test_abs ... FAILED`)
/// by stripping every module segment up to and including the last
/// `::`.
fn parse_failure_line(line: &str) -> Option<&str> {
    let line = line.trim();
    let rest = line.strip_prefix("test ")?;
    let rest = rest.strip_suffix(" ... FAILED")?;
    // libtest might print `module::path::test_name` for nested tests
    // or `test_name` at the crate root. For our data-driven harness
    // all test fns are at the crate root (tests/test_mod.rs), so in
    // practice `rest` is already the bare name. The split is
    // defensive for future changes.
    Some(rest.rsplit("::").next().unwrap_or(rest))
}

/// Apply a set of demotions to the expectations.toml text, preserving
/// comments, blank lines, and the block structure of unrelated
/// entries. Each demoted entry is rewritten from `status = "pass"`
/// (with no reason/tracking) to a `fail-compare` block carrying an
/// auto-generated reason and, if `tracking` is `Some`, a `tracking`
/// field too. A `None` tracking value omits the field entirely
/// rather than writing a stale or ambiguous reference.
///
/// We do this with line-based editing rather than serde round-trip
/// because serde would destroy comments, reorder sections, and wipe
/// the human-curated structure of the file. Every row in the demote
/// set is a pass entry today, which means the block currently has
/// exactly one field (`status = "pass"`) — so replacing the block
/// is as simple as matching the header and the next non-blank line.
fn apply_demotions(original: &str, to_demote: &[String], tracking: Option<&str>) -> String {
    let demote_set: std::collections::BTreeSet<&str> =
        to_demote.iter().map(String::as_str).collect();

    let mut out = String::with_capacity(original.len() + to_demote.len() * 120);
    let mut lines = original.lines().peekable();
    while let Some(line) = lines.next() {
        // Look for `[test_name]` headers that match our demote set.
        if let Some(name) = parse_header(line)
            && demote_set.contains(name)
        {
            // Consume the rest of the block (until next blank line or
            // next `[header]`). We drop every line so we can emit a
            // clean replacement below.
            while let Some(&peek) = lines.peek() {
                let trimmed = peek.trim();
                if trimmed.is_empty() || trimmed.starts_with('[') {
                    break;
                }
                lines.next();
            }
            // Emit the replacement block. The trailing blank line is
            // supplied by the original file's block separator (the
            // one we did NOT consume in the loop above).
            out.push_str(&format!("[{name}]\n"));
            out.push_str("status = \"fail-compare\"\n");
            out.push_str(
                "reason = \"demoted by `cargo xtask update-expectations` after test failure\"\n",
            );
            if let Some(tracking) = tracking {
                // The tracking reference is written verbatim — the
                // caller owns escaping concerns. In practice this is
                // always an issue/PR ref like `#321` or a commit SHA.
                out.push_str(&format!("tracking = \"{tracking}\"\n"));
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Extract the test name from a line of the form `[test_name]`. The
/// matching is deliberately strict: whitespace before the `[` or a
/// trailing comment would return `None`, but the generated file
/// never has either, so the strict form keeps the logic simple.
fn parse_header(line: &str) -> Option<&str> {
    let inner = line.strip_prefix('[')?.strip_suffix(']')?;
    if inner.starts_with("test_") {
        Some(inner)
    } else {
        None
    }
}

/// Absolute path to the repository root. Mirrors the helper in
/// `diff_expectations.rs`; kept local rather than factored into a
/// shared module because the duplication is trivially small and
/// keeps each subcommand self-contained.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate has no parent directory")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_failure_line` accepts the libtest formats we care about
    /// and rejects everything else (including the success, ignored,
    /// and summary lines that share a similar shape).
    #[test]
    fn parses_failed_test_lines() {
        assert_eq!(
            parse_failure_line("test test_abs ... FAILED"),
            Some("test_abs")
        );
        assert_eq!(
            parse_failure_line("test foo::test_abs ... FAILED"),
            Some("test_abs")
        );
        assert_eq!(
            parse_failure_line("    test test_abs ... FAILED  "),
            Some("test_abs")
        );
    }

    /// Non-failure lines never produce a test name.
    #[test]
    fn rejects_non_failure_lines() {
        assert_eq!(parse_failure_line("test test_abs ... ok"), None);
        assert_eq!(parse_failure_line("test test_abs ... ignored"), None);
        assert_eq!(
            parse_failure_line("test result: FAILED. 478 passed; 93 failed"),
            None
        );
        assert_eq!(parse_failure_line(""), None);
        assert_eq!(parse_failure_line("failures:"), None);
    }

    /// The section-header parser accepts `[test_*]` and rejects
    /// everything else (top-level comments, blank lines, malformed
    /// headers).
    #[test]
    fn parses_section_headers() {
        assert_eq!(parse_header("[test_abs]"), Some("test_abs"));
        assert_eq!(parse_header("[test_add_uint64]"), Some("test_add_uint64"));
        assert_eq!(parse_header("[something_else]"), None);
        assert_eq!(parse_header("# comment"), None);
        assert_eq!(parse_header(""), None);
    }

    /// A single demotion with an explicit tracking reference replaces
    /// the `[test]`/`status` block inline, writes the tracking field,
    /// and leaves surrounding blocks / comments / blank lines alone.
    #[test]
    fn apply_demotions_single_entry_with_tracking() {
        let original = "\
# header comment
[test_abs]
status = \"pass\"

[test_add]
status = \"pass\"

[test_ceil]
status = \"skip-codegen\"
reason = \"something\"
";
        let demoted = apply_demotions(original, &["test_add".to_string()], Some("#321"));
        assert!(demoted.contains("[test_abs]\nstatus = \"pass\""));
        assert!(demoted.contains("[test_add]\nstatus = \"fail-compare\""));
        assert!(demoted.contains("tracking = \"#321\""));
        // Unrelated rows with extra fields are preserved verbatim.
        assert!(demoted.contains("[test_ceil]\nstatus = \"skip-codegen\"\nreason = \"something\""));
    }

    /// With `tracking = None`, the rewritten block has no `tracking`
    /// field at all. This is the default and avoids baking a stale
    /// issue reference into every demotion.
    #[test]
    fn apply_demotions_without_tracking() {
        let original = "[test_a]\nstatus = \"pass\"\n";
        let demoted = apply_demotions(original, &["test_a".to_string()], None);
        assert!(demoted.contains("[test_a]\nstatus = \"fail-compare\""));
        assert!(!demoted.contains("tracking"));
    }

    /// Multiple demotions in the same pass are all applied, and the
    /// order of demotees in the input vector does not matter.
    #[test]
    fn apply_demotions_multiple_entries() {
        let original = "\
[test_a]
status = \"pass\"

[test_b]
status = \"pass\"

[test_c]
status = \"pass\"
";
        let demoted = apply_demotions(
            original,
            &["test_c".to_string(), "test_a".to_string()],
            None,
        );
        assert!(demoted.contains("[test_a]\nstatus = \"fail-compare\""));
        assert!(demoted.contains("[test_b]\nstatus = \"pass\""));
        assert!(demoted.contains("[test_c]\nstatus = \"fail-compare\""));
    }

    /// A demote of an entry that doesn't exist in the file is a no-op.
    /// (Shouldn't happen in practice because the caller filters to
    /// entries marked `pass`, but a trailing safety check.)
    #[test]
    fn apply_demotions_missing_entry_is_noop() {
        let original = "[test_a]\nstatus = \"pass\"\n";
        let result = apply_demotions(original, &["test_zzz".to_string()], None);
        assert_eq!(result, original);
    }

    /// Parser round-trip: a file rewritten by `apply_demotions` must
    /// still parse cleanly through the schema. This is the catch for
    /// syntactic regressions in the emitter. Run with both tracking
    /// present and absent so both branches are exercised.
    #[test]
    fn rewritten_file_reparses() {
        let original = "\
[test_a]
status = \"pass\"

[test_b]
status = \"pass\"
";
        let demoted = apply_demotions(original, &["test_a".to_string()], Some("#321"));
        let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &demoted).unwrap();
        assert_eq!(parsed.entries["test_a"].status, Status::FailCompare);
        assert_eq!(parsed.entries["test_b"].status, Status::Pass);
        assert_eq!(parsed.entries["test_a"].tracking.as_deref(), Some("#321"));

        let demoted_no_tracking = apply_demotions(original, &["test_a".to_string()], None);
        let parsed =
            Expectations::from_toml(PathBuf::from("t.toml"), &demoted_no_tracking).unwrap();
        assert_eq!(parsed.entries["test_a"].status, Status::FailCompare);
        assert!(parsed.entries["test_a"].tracking.is_none());
    }
}
