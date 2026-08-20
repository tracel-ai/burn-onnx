//! `cargo xtask retriage`: re-run the `skip-codegen` and `skip-compile`
//! rows of `expectations.toml` against the current tree and rewrite
//! them to match reality.
//!
//! `crates/onnx-official-tests/build.rs` only exercises `pass` and
//! `fail-compare` rows. Every `skip-*` row is read as documentation and
//! never tried, so those rows drift the moment someone fixes the bug
//! that put them there. The drift is one-directional and invisible: the
//! file always claims the tree is worse than it is.
//!
//! `update-expectations` covers the other direction, demoting `pass`
//! rows that started failing, and leaves promotion to manual edits on
//! the grounds that promotion is prohibitively expensive. For codegen
//! that is not true: ~700 process spawns finish in under a minute, and
//! that alone catches the bulk of the drift. Compile and compare really
//! are expensive, which is why they are staged behind it rather than
//! run per row.
//!
//! Two stages run here:
//!
//! 1. **Codegen.** Run `onnx2burn` on every selected row. Rows that
//!    still fail keep `skip-codegen` and get a freshly captured reason
//!    (a `skip-compile` row that fails here was mislabeled). Rows that
//!    succeed are promoted to `pass` optimistically.
//! 2. **Compile.** Build the test crate. A rustc error inside a
//!    generated model demotes that row to `skip-compile` carrying the
//!    diagnostic as its reason; an error inside the generated harness is
//!    attributed to the row whose runner encloses it. Repeated until the
//!    crate builds, because one broken model can mask errors in later
//!    ones. If it never builds, the file is restored and the command
//!    fails rather than leaving rows claiming a `pass` nothing verified.
//!
//! What survives is a `pass` claim that this row compiles. It is not yet
//! a claim about output, which is `cargo xtask update-expectations`'s
//! job, and for a row that build.rs cannot harness (dynamic shape,
//! rank-0 I/O, a dtype the `.pb` loader cannot build) nothing will ever
//! check the output at all. `report` counts those separately so the
//! promotion list does not overstate what was proven.
//!
//! Rows marked `wontfix` are left alone, as are rows with no vendored
//! `model.onnx`.
//!
//! `--dry-run` reports stage 1 without touching the file.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use tracel_xtask::prelude::*;

use crate::expectations_schema::{Expectations, Status};

/// Maximum number of build attempts in stage 2. Each round removes at
/// least one broken model from the crate, so this only bounds the
/// pathological case where rustc surfaces one model at a time.
const MAX_COMPILE_ROUNDS: usize = 12;

/// Longest reason string written back to the file.
const MAX_REASON_LEN: usize = 220;

/// Arguments for the `retriage` subcommand.
#[derive(clap::Args)]
pub struct RetriageArgs {
    /// Report the planned rewrite without modifying any file. Only
    /// stage 1 runs: stage 2 needs the promotions on disk to build them.
    #[arg(long)]
    pub dry_run: bool,

    /// Stop after stage 1. Rows whose codegen succeeded stay marked
    /// `pass` with no compile check, so the file may claim more than the
    /// tree delivers. Useful for looking at codegen drift on its own.
    #[arg(long)]
    pub codegen_only: bool,

    /// Re-check only the first N eligible rows, in file order, for
    /// sampling the drift without paying for the full sweep.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Optional `tracking` value to embed in every rewritten row, e.g.
    /// `"#456"`. Omitted by default: a captured reason is
    /// self-documenting, and a stale issue reference is worse than none.
    #[arg(long)]
    pub tracking: Option<String>,
}

/// What the sweep decided about one row.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Verdict {
    status: Status,
    reason: Option<String>,
}

pub fn handle_command(args: RetriageArgs) -> anyhow::Result<()> {
    let repo_root = repo_root();
    let crate_dir = repo_root.join("crates/onnx-official-tests");
    let expectations_path = crate_dir.join("expectations.toml");

    let original = std::fs::read_to_string(&expectations_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", expectations_path.display()))?;
    let parsed = Expectations::from_toml(expectations_path.clone(), &original)
        .map_err(|e| anyhow::anyhow!("parse expectations: {e}"))?;

    let mut eligible: Vec<String> = parsed
        .entries
        .iter()
        .filter(|(_, e)| matches!(e.status, Status::SkipCodegen | Status::SkipCompile))
        .filter(|(_, e)| !e.wontfix)
        .map(|(name, _)| name.clone())
        .collect();
    if let Some(limit) = args.limit {
        eligible.truncate(limit);
    }

    if eligible.is_empty() {
        info!("No skip-codegen or skip-compile rows to re-check.");
        return Ok(());
    }
    info!("Re-checking {} skipped row(s)", eligible.len());

    // --- Stage 1: codegen ------------------------------------------
    let onnx2burn = build_onnx2burn(&repo_root)?;
    let scratch = repo_root.join("target/retriage");
    let _ = std::fs::remove_dir_all(&scratch);

    let mut verdicts: BTreeMap<String, Verdict> = BTreeMap::new();
    let mut codegen_ok: Vec<String> = Vec::new();
    for name in &eligible {
        let model = crate_dir.join("vendor/node").join(name).join("model.onnx");
        if !model.is_file() {
            warn!("{name}: no vendored model.onnx, leaving the row alone");
            continue;
        }
        match run_codegen(&onnx2burn, &model, &scratch.join(name))? {
            None => codegen_ok.push(name.clone()),
            Some(reason) => {
                verdicts.insert(
                    name.clone(),
                    Verdict {
                        status: Status::SkipCodegen,
                        reason: Some(reason),
                    },
                );
            }
        }
    }
    info!(
        "Codegen: {} succeeded, {} still fail",
        codegen_ok.len(),
        verdicts.len()
    );

    for name in &codegen_ok {
        verdicts.insert(
            name.clone(),
            Verdict {
                status: Status::Pass,
                reason: None,
            },
        );
    }

    if args.dry_run {
        report(&parsed, &verdicts, None);
        warn!("--dry-run set; no files were modified");
        if !args.codegen_only {
            warn!("stage 2 (compile) needs the promotions on disk, so it did not run");
        }
        return Ok(());
    }

    let mut text = apply_verdicts(&original, &verdicts, args.tracking.as_deref());
    std::fs::write(&expectations_path, &text)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", expectations_path.display()))?;

    // --- Stage 2: compile ------------------------------------------
    //
    // Stage 1's promotions are on disk now, because stage 2 builds the
    // test crate and can only see rows the file already marks `pass`.
    // Until stage 2 agrees, that file over-claims: it asserts rows pass
    // that nothing has compiled. Any exit from here without a clean
    // build has to put `original` back, or a failed sweep leaves the
    // repo unbuildable for everyone with no hint of why.
    if !args.codegen_only && !codegen_ok.is_empty() {
        let outcome = run_compile_rounds(
            &expectations_path,
            &mut text,
            &mut verdicts,
            &codegen_ok,
            args.tracking.as_deref(),
        );
        if let Err(e) = outcome {
            std::fs::write(&expectations_path, &original)
                .map_err(|e| anyhow::anyhow!("restore {}: {e}", expectations_path.display()))?;
            return Err(e.context(format!(
                "stage 2 did not finish; restored {} to its previous contents",
                expectations_path.display()
            )));
        }
    }

    report(&parsed, &verdicts, codegen_only_rows(&repo_root).as_ref());
    info!("Rewrote {}", expectations_path.display());
    info!(
        "Next: `cargo xtask update-expectations` to demote any promoted row whose \
         output does not match the reference tensors."
    );
    Ok(())
}

/// Build the test crate, demoting rows rustc rejects, until it is clean.
///
/// Each round removes at least one row from the candidate set, so the
/// loop terminates; `MAX_COMPILE_ROUNDS` only bounds the pathological
/// case where rustc surfaces one row at a time. Exhausting it is a
/// failure, not a warning: the file would be left asserting `pass` for
/// rows that demonstrably do not compile.
fn run_compile_rounds(
    expectations_path: &Path,
    text: &mut String,
    verdicts: &mut BTreeMap<String, Verdict>,
    codegen_ok: &[String],
    tracking: Option<&str>,
) -> anyhow::Result<()> {
    let mut remaining: BTreeSet<String> = codegen_ok.iter().cloned().collect();
    for round in 1..=MAX_COMPILE_ROUNDS {
        let broken = compile_and_collect_broken_models(&remaining)?;
        if broken.is_empty() {
            info!("Compile: clean after {round} round(s)");
            return Ok(());
        }
        info!(
            "Compile round {round}: demoting {} model(s) to skip-compile",
            broken.len()
        );

        let round_verdicts: BTreeMap<String, Verdict> = broken
            .into_iter()
            .map(|(name, diag)| {
                (
                    name,
                    Verdict {
                        status: Status::SkipCompile,
                        reason: Some(diag),
                    },
                )
            })
            .collect();
        for (name, verdict) in &round_verdicts {
            remaining.remove(name);
            verdicts.insert(name.clone(), verdict.clone());
        }

        *text = apply_verdicts(text, &round_verdicts, tracking);
        std::fs::write(expectations_path, &*text)
            .map_err(|e| anyhow::anyhow!("write {}: {e}", expectations_path.display()))?;
    }

    Err(anyhow::anyhow!(
        "still not building after {MAX_COMPILE_ROUNDS} round(s), with {} row(s) \
         still unverified",
        remaining.len()
    ))
}

/// Build the codegen binary once and return its path.
///
/// The path is verified to exist. `cargo build` succeeds regardless of
/// where the artifact lands, so with `CARGO_TARGET_DIR` set (or
/// `build.target-dir`, or a shared target directory) the conventional
/// path is simply wrong. Returning it unchecked would make every
/// subsequent spawn fail, and since a spawn failure used to be recorded
/// as that row's codegen verdict, one stray environment variable
/// rewrote the entire table to `skip-codegen`.
fn build_onnx2burn(repo_root: &Path) -> anyhow::Result<PathBuf> {
    info!("Building onnx2burn...");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "-p",
            "burn-onnx",
            "--bin",
            "onnx2burn",
        ])
        .current_dir(repo_root)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn cargo: {e}"))?;
    if !status.success() {
        return Err(anyhow::anyhow!("building onnx2burn failed"));
    }

    let path = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.join("target"))
        .join("release/onnx2burn");
    if !path.is_file() {
        return Err(anyhow::anyhow!(
            "onnx2burn built successfully but no binary at {}; \
             set CARGO_TARGET_DIR to the directory cargo is actually using",
            path.display()
        ));
    }
    Ok(path)
}

/// Run codegen for one model.
///
/// `Ok(None)` means codegen succeeded, `Ok(Some(reason))` that this
/// model was rejected, and `Err` that the sweep itself is broken. The
/// split matters: a failure to spawn the child or create its output
/// directory says nothing about the model, and recording it as that
/// row's verdict would overwrite a real diagnosis with an environment
/// problem. Those abort the run instead.
///
/// A separate process per model means a codegen panic costs an exit
/// code rather than the whole sweep.
fn run_codegen(onnx2burn: &Path, model: &Path, out_dir: &Path) -> anyhow::Result<Option<String>> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| anyhow::anyhow!("create {}: {e}", out_dir.display()))?;
    let output = Command::new(onnx2burn)
        .arg(model)
        .arg(out_dir)
        .env("RUST_LOG", "error")
        .output()
        .map_err(|e| anyhow::anyhow!("spawn {}: {e}", onnx2burn.display()))?;

    if output.status.success() {
        return Ok(None);
    }

    // A child killed by a signal (OOM, Ctrl-C) reports no exit code and
    // usually no message. That is not a verdict about the model either.
    if output.status.code().is_none() {
        return Err(anyhow::anyhow!(
            "onnx2burn was killed by a signal while processing {}",
            model.display()
        ));
    }

    Ok(Some(extract_panic_reason(&String::from_utf8_lossy(
        &output.stderr,
    ))))
}

/// Pull the operator-level complaint out of an onnx2burn panic.
///
/// The message sits on the line after `panicked at <loc>:`, wrapped in
/// framing whose shape is the same for every failure even though its
/// text is not (it names the vendored path). Stripping it keeps reasons
/// comparable across rows and close in shape to the hand-written ones
/// already in the file.
fn extract_panic_reason(stderr: &str) -> String {
    let stripped = strip_ansi(stderr);
    let lines: Vec<&str> = stripped.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("panicked at"))
        .map(|i| i + 1)
        .unwrap_or(0);

    // Take the message body: up to the backtrace note or a blank line.
    // Multi-line messages (custom-op coverage lists) are joined so the
    // reason stays a single TOML string.
    let body: Vec<&str> = lines
        .get(start..)
        .unwrap_or(&[])
        .iter()
        .map(|l| l.trim())
        .take_while(|l| !l.is_empty() && !l.starts_with("note:"))
        .collect();

    let joined = body.join(" ");
    let mut msg = joined.trim();

    // `Failed to parse ONNX file '<path>': ` names the vendored path,
    // which differs per row and would make otherwise-identical reasons
    // look distinct.
    if let Some(rest) = msg
        .strip_prefix("Failed to parse ONNX file ")
        .and_then(|r| r.split_once("': "))
        .map(|(_, rest)| rest)
    {
        msg = rest;
    }
    msg = msg.strip_prefix("Type inference failed: ").unwrap_or(msg);

    if msg.is_empty() {
        return "codegen failed with no message on stderr".to_string();
    }
    truncate_reason(msg)
}

/// Drop ANSI colour codes so they never reach the TOML file.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Consume through the final byte of the escape sequence.
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Bound a reason's length, then escape it for a TOML basic string.
///
/// The order is load-bearing. Escaping first and cutting afterwards can
/// slice between a backslash and the character it escapes, emitting a
/// dangling `\` before the ellipsis and producing a row that TOML
/// refuses to parse. rustc diagnostics are exactly the input that
/// triggers it: quote-dense and routinely past the limit. Cutting the
/// raw message first means every escape pair is written whole.
fn truncate_reason(msg: &str) -> String {
    let truncated = if msg.chars().count() > MAX_REASON_LEN {
        msg.chars().take(MAX_REASON_LEN - 3).collect::<String>() + "..."
    } else {
        msg.to_string()
    };
    truncated.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Build the test crate and map rustc errors back to the rows that
/// produced them.
///
/// Errors land in one of two generated files. A model that does not
/// compile fails inside `$OUT_DIR/model/<test_name>.rs`, so the path
/// alone names the row. A model that compiles but that the generated
/// driver cannot call (a Shape-typed graph input arriving as
/// `[i64; N]` where the driver built a `Tensor<1, Int>`) fails inside
/// `$OUT_DIR/harness.rs`, where the row is the enclosing `fn`.
///
/// Only names in `candidates` are reported: an error inside a row that
/// was already passing is a real regression and must not be quietly
/// reclassified.
fn compile_and_collect_broken_models(
    candidates: &BTreeSet<String>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let output = Command::new("cargo")
        .args([
            "build",
            "-p",
            "onnx-official-tests",
            "--tests",
            "--message-format=short",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn cargo: {e}"))?;
    if output.status.success() {
        return Ok(BTreeMap::new());
    }

    let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
    let mut broken: BTreeMap<String, String> = BTreeMap::new();
    let mut harness: Option<HarnessIndex> = None;
    for line in stderr.lines() {
        let Some((path, line_no, message)) = parse_short_diagnostic(line) else {
            continue;
        };

        let name = if let Some(name) = model_name_from_path(path) {
            Some(name)
        } else if path.ends_with("/harness.rs") {
            let index = match &harness {
                Some(index) => index,
                None => harness.insert(HarnessIndex::load(path)?),
            };
            index.enclosing_test(line_no)
        } else {
            None
        };

        let Some(name) = name else { continue };
        if candidates.contains(&name) {
            broken
                .entry(name)
                .or_insert_with(|| truncate_reason(message));
        }
    }

    if broken.is_empty() {
        // This is the one failure the tool cannot interpret, so hand
        // over the evidence instead of a shrug. Typically a build.rs
        // panic, a linker error, or a breakage in a row that was
        // already passing.
        let tail: Vec<&str> = stderr.lines().rev().take(30).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(anyhow::anyhow!(
            "onnx-official-tests failed to build but no error was attributed to a \
             promoted row; fix the build manually and re-run.\ncargo stderr (last \
             30 lines):\n{tail}"
        ));
    }
    Ok(broken)
}

/// Split one `--message-format=short` error line into its path, line
/// number, and message. The format is
/// `file:line:col: error[CODE]: message`; warnings and notes are
/// ignored.
fn parse_short_diagnostic(line: &str) -> Option<(&str, usize, &str)> {
    let (path, rest) = line.split_once(".rs:")?;
    let path_end = path.len() + 3;
    let (line_no, rest) = rest.split_once(':')?;
    let line_no = line_no.parse::<usize>().ok()?;
    let after_span = rest.split_once(": ")?.1;
    let message = after_span.strip_prefix("error")?;
    // Drop an optional `[E0433]` code and the separating colon.
    let (_, message) = message.split_once(": ")?;
    Some((&line[..path_end], line_no, message.trim()))
}

/// Line-number-to-row-name lookup over the generated `harness.rs`.
///
/// The file is a flat list of function bodies in two blocks: every pass
/// row as `fn <name>()`, then every fail-compare row as
/// `fn fail_compare_<name>() -> bool`. The row owning a diagnostic is
/// the nearest `fn` header at or above it.
///
/// Both blocks must be indexed. Indexing only the `test_` prefix leaves
/// every fail-compare body invisible, so a diagnostic inside one
/// resolves to the last pass row above it instead, and that row gets
/// demoted carrying a stranger's rustc error. Built once per compile
/// round and reused for every diagnostic.
struct HarnessIndex {
    /// `(line number of the fn header, row name)`, ascending.
    fns: Vec<(usize, String)>,
}

impl HarnessIndex {
    fn load(path: &str) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read generated harness {path}: {e}"))?;
        let fns = text
            .lines()
            .enumerate()
            .filter_map(|(i, line)| {
                let name = line
                    .trim_start()
                    .strip_prefix("fn ")?
                    .split_once('(')?
                    .0
                    .trim();
                // A fail-compare runner is named after the row it drives.
                let name = name.strip_prefix("fail_compare_").unwrap_or(name);
                name.starts_with("test_").then(|| (i + 1, name.to_string()))
            })
            .collect();
        Ok(Self { fns })
    }

    fn enclosing_test(&self, line_no: usize) -> Option<String> {
        self.fns
            .iter()
            .rev()
            .find(|(start, _)| *start <= line_no)
            .map(|(_, name)| name.clone())
    }
}

/// `.../out/model/test_foo.rs` -> `test_foo`.
fn model_name_from_path(path: &str) -> Option<String> {
    let (_, tail) = path.rsplit_once("/model/")?;
    tail.strip_suffix(".rs").map(str::to_string)
}

/// Rewrite the matching rows in place, preserving every other line.
fn apply_verdicts(
    original: &str,
    verdicts: &BTreeMap<String, Verdict>,
    tracking: Option<&str>,
) -> String {
    let mut out = String::with_capacity(original.len() + verdicts.len() * 160);
    let mut lines = original.lines().peekable();
    while let Some(line) = lines.next() {
        if let Some(name) = parse_header(line)
            && let Some(verdict) = verdicts.get(name)
        {
            // Drop the old body so the replacement is clean. The blank
            // separator line is left in place by the peek guard.
            while let Some(&peek) = lines.peek() {
                let trimmed = peek.trim();
                if trimmed.is_empty() || trimmed.starts_with('[') {
                    break;
                }
                lines.next();
            }
            out.push_str(&format!("[{name}]\n"));
            out.push_str(&format!("status = \"{}\"\n", verdict.status.as_str()));
            if let Some(reason) = &verdict.reason {
                out.push_str(&format!("reason = \"{reason}\"\n"));
            }
            if let Some(tracking) = tracking {
                out.push_str(&format!("tracking = \"{tracking}\"\n"));
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `[test_name]` -> `test_name`, ignoring anything else.
fn parse_header(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    trimmed
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .filter(|name| !name.is_empty() && !name.contains(['[', ']', '.']))
}

/// Read the rows build.rs compiled but generated no `#[test]` for.
///
/// A row can be promoted to `pass` on the strength of compiling and
/// still never have its output compared: build.rs skips harness
/// generation for dynamic shapes, rank-0 I/O, and dtypes the `.pb`
/// loader cannot construct, and `update-expectations` can only demote
/// rows whose test failed. A row with no test is therefore unfalsifiable
/// once promoted, which is worth saying out loud rather than folding
/// into a promotion count.
///
/// Returns `None` if the manifest cannot be located, in which case the
/// caller reports the promotions without the breakdown.
fn codegen_only_rows(repo_root: &Path) -> Option<BTreeSet<String>> {
    let build_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.join("target"))
        .join("debug/build");

    // One `onnx-official-tests-<hash>` directory per feature set; the
    // most recently written manifest is the one stage 2 just produced.
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&build_dir).ok()?.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("onnx-official-tests-")
        {
            continue;
        }
        let manifest = entry.path().join("out/manifest.rs");
        let Ok(modified) = manifest.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if newest.as_ref().is_none_or(|(t, _)| modified > *t) {
            newest = Some((modified, manifest));
        }
    }

    let text = std::fs::read_to_string(newest?.1).ok()?;
    let start = text.find("CODEGEN_ONLY_TESTS")?;
    let open = text[start..].find('[')? + start;
    let close = text[open..].find(']')? + open;
    Some(
        text[open..close]
            .split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect(),
    )
}

/// Summarise the sweep as a from -> to tally plus the promotion list.
fn report(
    before: &Expectations,
    verdicts: &BTreeMap<String, Verdict>,
    codegen_only: Option<&BTreeSet<String>>,
) {
    let mut transitions: BTreeMap<(Status, Status), usize> = BTreeMap::new();
    let mut promoted: Vec<&str> = Vec::new();
    for (name, verdict) in verdicts {
        let Some(entry) = before.entries.get(name) else {
            continue;
        };
        *transitions
            .entry((entry.status, verdict.status))
            .or_default() += 1;
        if verdict.status == Status::Pass {
            promoted.push(name);
        }
    }

    info!("Re-triage summary:");
    for ((from, to), count) in &transitions {
        let marker = if from == to { "  " } else { "->" };
        info!(
            "  {marker} {:>4}  {} -> {}",
            count,
            from.as_str(),
            to.as_str()
        );
    }
    if !promoted.is_empty() {
        let unverifiable: Vec<&&str> = match codegen_only {
            Some(set) => promoted.iter().filter(|n| set.contains(**n)).collect(),
            None => Vec::new(),
        };
        info!(
            "Promoted to pass ({}, of which {} compile but are never compared):",
            promoted.len(),
            unverifiable.len()
        );
        for name in &promoted {
            let mark = if unverifiable.contains(&name) {
                "  (codegen-only)"
            } else {
                ""
            };
            info!("  - {name}{mark}");
        }
        if !unverifiable.is_empty() {
            warn!(
                "{} promoted row(s) have no generated test: build.rs cannot harness \
                 them, so `update-expectations` can never demote them and their \
                 output is unchecked.",
                unverifiable.len()
            );
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level under the repo root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The common single-line case: framing stripped, operator
    /// complaint kept.
    #[test]
    fn extract_reason_strips_framing() {
        let stderr = "\
thread 'main' (123) panicked at crates/burn-onnx/src/model_gen.rs:397:33:
Failed to parse ONNX file 'vendor/node/test_x/model.onnx': Type inference failed: Node 'resize1' (Resize): Invalid attribute 'axes': custom axes attribute is not supported
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
";
        assert_eq!(
            extract_panic_reason(stderr),
            "Node 'resize1' (Resize): Invalid attribute 'axes': custom axes attribute is not supported"
        );
    }

    /// Multi-line panic bodies are joined into one TOML-safe string.
    #[test]
    fn extract_reason_joins_multiline_body() {
        let stderr = "\
thread 'main' (123) panicked at crates/burn-onnx/src/model_gen.rs:397:33:
Failed to parse ONNX file 'vendor/node/test_adagrad/model.onnx': model contains 1 custom op(s) with no covering inference hook:
  - ai.onnx.preview.training::Adagrad used by 1 node(s)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
";
        let reason = extract_panic_reason(stderr);
        assert!(
            reason.starts_with("model contains 1 custom op(s)"),
            "{reason}"
        );
        assert!(reason.contains("Adagrad used by 1 node(s)"), "{reason}");
        assert!(!reason.contains('\n'));
    }

    /// Colour codes from the logger must not reach the file.
    #[test]
    fn extract_reason_strips_ansi() {
        let stderr =
            "panicked at x.rs:1:1:\n\u{1b}[31mERROR\u{1b}[0m something went wrong\nnote: bt\n";
        let reason = extract_panic_reason(stderr);
        assert_eq!(reason, "ERROR something went wrong");
    }

    /// A panic-free failure still yields a usable reason rather than an
    /// empty string, which would produce `reason = ""`.
    #[test]
    fn extract_reason_handles_empty_stderr() {
        assert_eq!(
            extract_panic_reason(""),
            "codegen failed with no message on stderr"
        );
    }

    /// Quotes and backslashes are escaped so the rewritten row parses.
    #[test]
    fn reasons_are_toml_escaped() {
        let escaped = truncate_reason(r#"expected "Tensor", got C:\path"#);
        assert_eq!(escaped, r#"expected \"Tensor\", got C:\\path"#);
    }

    #[test]
    fn long_reasons_are_truncated() {
        let reason = truncate_reason(&"x".repeat(500));
        assert_eq!(reason.chars().count(), MAX_REASON_LEN);
        assert!(reason.ends_with("..."));
    }

    /// A long, quote-dense reason must still parse. Escaping before
    /// truncating would cut a `\"` pair in half and emit a dangling
    /// backslash, and rustc diagnostics look exactly like this.
    #[test]
    fn truncated_reasons_never_split_an_escape() {
        for pad in 210..225 {
            let raw = format!("{}\"{}", "A".repeat(pad), "B".repeat(80));
            let reason = truncate_reason(&raw);
            let toml_text = format!("[t]\nstatus = \"pass\"\nreason = \"{reason}\"\n");
            let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &toml_text)
                .unwrap_or_else(|e| panic!("pad {pad} produced unparsable TOML: {e}"));
            // The reason must survive as a real string, not a mangled one.
            assert!(parsed.entries["t"].reason.is_some());
        }

        // Same for a backslash, whose escape is also two characters.
        for pad in 210..225 {
            let raw = format!("{}\\{}", "A".repeat(pad), "B".repeat(80));
            let reason = truncate_reason(&raw);
            let toml_text = format!("[t]\nstatus = \"pass\"\nreason = \"{reason}\"\n");
            Expectations::from_toml(PathBuf::from("t.toml"), &toml_text)
                .unwrap_or_else(|e| panic!("pad {pad} produced unparsable TOML: {e}"));
        }
    }

    /// An end-to-end guard: a real rustc diagnostic, over the limit and
    /// full of backticks and quotes, must round-trip through the
    /// rewriter and back out of the parser unchanged in meaning.
    #[test]
    fn realistic_diagnostic_round_trips() {
        let diag = "mismatched types: expected `burn::Tensor<1, burn::prelude::Bool>`, \
                    found `burn::Tensor<1, burn::prelude::Int>` in this expression, \
                    note the \"expected\" type comes from the signature of \
                    `mask_where` declared elsewhere in the generated module";
        let verdicts = BTreeMap::from([(
            "test_a".to_string(),
            Verdict {
                status: Status::SkipCompile,
                reason: Some(truncate_reason(diag)),
            },
        )]);
        let out = apply_verdicts("[test_a]\nstatus = \"pass\"\n", &verdicts, None);
        let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &out).unwrap();
        let reason = parsed.entries["test_a"].reason.as_deref().unwrap();
        assert!(reason.starts_with("mismatched types: expected `burn::Tensor<1"));
    }

    #[test]
    fn short_diagnostics_parse() {
        let line =
            "/t/out/model/test_size.rs:60:27: error[E0609]: no field `shape` on type `[i64; 4]`";
        let (path, line_no, message) = parse_short_diagnostic(line).unwrap();
        assert_eq!(path, "/t/out/model/test_size.rs");
        assert_eq!(line_no, 60);
        assert_eq!(message, "no field `shape` on type `[i64; 4]`");
        assert_eq!(model_name_from_path(path).unwrap(), "test_size");
    }

    /// An error in the generated driver is attributed to the row whose
    /// `fn` encloses it, so a harness gap lands on the right entry
    /// instead of aborting the sweep.
    #[test]
    fn harness_errors_attribute_to_enclosing_test() {
        let harness = HarnessIndex {
            fns: vec![
                (10, "test_alpha".to_string()),
                (40, "test_beta".to_string()),
                (90, "test_gamma".to_string()),
            ],
        };
        assert_eq!(harness.enclosing_test(41).as_deref(), Some("test_beta"));
        assert_eq!(harness.enclosing_test(89).as_deref(), Some("test_beta"));
        assert_eq!(harness.enclosing_test(90).as_deref(), Some("test_gamma"));
        // A diagnostic above the first test belongs to the preamble.
        assert_eq!(harness.enclosing_test(3), None);
    }

    /// `HarnessIndex::load` must see both blocks build.rs emits: the
    /// pass rows as `fn <name>()`, then the fail-compare rows as
    /// `fn fail_compare_<name>() -> bool`. Indexing only the first
    /// block silently blames the last pass row for any diagnostic in
    /// the second, which is a wrong demotion carrying a stranger's
    /// error.
    #[test]
    fn harness_index_sees_fail_compare_runners() {
        let dir = std::env::temp_dir().join("retriage_harness_index_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("harness.rs");
        std::fs::write(
            &path,
            "// preamble\n\
             fn test_alpha() {\n\
             \x20   let x = 1;\n\
             }\n\
             fn test_omega() {\n\
             \x20   let y = 2;\n\
             }\n\
             fn fail_compare_test_beta() -> bool {\n\
             \x20   let z = 3;\n\
             }\n",
        )
        .unwrap();

        let index = HarnessIndex::load(path.to_str().unwrap()).unwrap();
        assert_eq!(index.enclosing_test(6).as_deref(), Some("test_omega"));
        // Line 9 sits inside the fail-compare runner, so it belongs to
        // test_beta, not to the pass row that happens to precede it.
        assert_eq!(index.enclosing_test(9).as_deref(), Some("test_beta"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Warnings share the line shape but must not demote anything.
    #[test]
    fn short_diagnostics_ignore_warnings() {
        let line = "/t/out/model/test_size.rs:60:27: warning: unused variable: `x`";
        assert!(parse_short_diagnostic(line).is_none());
    }

    /// Errors outside the generated-model directory belong to the
    /// harness, not to a row, and are left unattributed.
    #[test]
    fn diagnostics_outside_model_dir_are_unattributed() {
        assert!(model_name_from_path("/t/src/lib.rs").is_none());
    }

    #[test]
    fn apply_verdicts_rewrites_only_named_rows() {
        let original = "\
# leading comment
[test_a]
status = \"skip-compile\"
reason = \"stale\"
tracking = \"#314\"

[test_b]
status = \"pass\"
";
        let verdicts = BTreeMap::from([(
            "test_a".to_string(),
            Verdict {
                status: Status::Pass,
                reason: None,
            },
        )]);
        let out = apply_verdicts(original, &verdicts, None);
        assert!(out.contains("# leading comment"));
        assert!(out.contains("[test_a]\nstatus = \"pass\"\n"));
        // The stale reason and tracking are gone with the old body.
        assert!(!out.contains("stale"));
        assert!(!out.contains("#314"));
        // Untouched rows survive verbatim.
        assert!(out.contains("[test_b]\nstatus = \"pass\""));
    }

    /// The rewritten file must still parse, including reasons that
    /// contain quotes.
    #[test]
    fn rewritten_file_reparses() {
        let original = "[test_a]\nstatus = \"skip-compile\"\n";
        let verdicts = BTreeMap::from([(
            "test_a".to_string(),
            Verdict {
                status: Status::SkipCodegen,
                reason: Some(truncate_reason(r#"expected "Tensor for split sizes""#)),
            },
        )]);
        let out = apply_verdicts(original, &verdicts, Some("#456"));
        let parsed = Expectations::from_toml(PathBuf::from("t.toml"), &out).unwrap();
        let entry = &parsed.entries["test_a"];
        assert_eq!(entry.status, Status::SkipCodegen);
        assert_eq!(
            entry.reason.as_deref(),
            Some(r#"expected "Tensor for split sizes""#)
        );
        assert_eq!(entry.tracking.as_deref(), Some("#456"));
    }

    #[test]
    fn headers_are_recognised() {
        assert_eq!(parse_header("[test_abs]"), Some("test_abs"));
        assert_eq!(parse_header("status = \"pass\""), None);
        assert_eq!(parse_header("[[array]]"), None);
    }
}
