//! `cargo xtask diff-expectations` — compare the working-tree
//! `expectations.toml` against the same file at a given git ref
//! (default `origin/main`) and summarise the delta.
//!
//! This is the M4 piece of tracel-ai/burn-onnx#315. CI runs it on
//! pull requests and posts the markdown output as a PR comment so
//! authors see at a glance which tests they've unblocked, broken, or
//! reclassified. Locally, the text format is more useful for triage.
//!
//! ## Classification of changes
//!
//! For every test name that exists in either the base or the current
//! expectations file, the diff is bucketed as one of:
//!
//! * **Added** — present in current, missing from base. The test was
//!   vendored in this branch (e.g. via `refresh-onnx-tests`).
//! * **Removed** — present in base, missing from current. The test was
//!   removed from `vendor/node/` in this branch.
//! * **Promotion** — status changed and the new status is `pass`. The
//!   most user-visible category: the contributor unblocked something.
//! * **Regression** — the base status was `pass` and the new status
//!   is anything else. This is the row the CI gate exists to protect
//!   against.
//! * **Sideways** — both base and current are non-`pass`, but the
//!   status changed (e.g. `skip-codegen` → `skip-compile`). Usually
//!   harmless but worth surfacing because a contributor intentionally
//!   moved an entry.
//!
//! Tests whose status is unchanged are silently ignored.

use std::path::PathBuf;

use tracel_xtask::prelude::*;

use crate::expectations_schema::{Expectations, Status};

/// Arguments for the `diff-expectations` subcommand.
#[derive(clap::Args)]
pub struct DiffExpectationsArgs {
    /// Git ref to compare against. Defaults to `origin/main`, which
    /// is the right choice for a PR CI run; override with the
    /// target branch name for cross-fork PRs or with a specific
    /// commit SHA for historical comparisons.
    #[arg(long, default_value = "origin/main")]
    pub base: String,

    /// Output format. `text` is for local triage; `markdown` is the
    /// shape the PR-comment workflow step consumes via `gh pr
    /// comment --body-file -`.
    #[arg(long, default_value = "text")]
    pub format: OutputFormat,

    /// Exit with status 1 if the diff is non-empty. Intended for a
    /// future "required check" CI posture; off by default so a PR
    /// author's unrelated non-empty delta does not force them to
    /// update expectations to ship an unrelated fix.
    #[arg(long)]
    pub fail_on_diff: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Markdown,
}

/// Top-level dispatcher. Reads the base expectations via `git show`,
/// loads the current one from the working tree, runs [`diff`], and
/// renders the result in the requested format.
pub fn handle_command(args: DiffExpectationsArgs) -> anyhow::Result<()> {
    let repo_root = repo_root();
    let current_path = repo_root.join("crates/onnx-official-tests/expectations.toml");
    let relative_path = "crates/onnx-official-tests/expectations.toml";

    info!("Base ref:  {}", args.base);
    info!("Current:   {}", current_path.display());

    let current = Expectations::load(&current_path)
        .map_err(|e| anyhow::anyhow!("load current expectations: {e}"))?;

    let base = load_base(&args.base, relative_path)?;

    let changes = diff(&base, &current);

    let rendered = match args.format {
        OutputFormat::Text => render_text(&args.base, &changes),
        OutputFormat::Markdown => render_markdown(&args.base, &changes),
    };
    print!("{rendered}");

    if args.fail_on_diff && !changes.is_empty() {
        return Err(anyhow::anyhow!(
            "expectations diff is non-empty and --fail-on-diff is set"
        ));
    }

    Ok(())
}

/// Read `expectations.toml` at `base_ref` by shelling out to
/// `git show`. This is cheaper and more portable than linking a git
/// library, and matches how the rest of the codebase shells out for
/// git operations.
///
/// Returns an empty `Expectations` if the file does not exist at
/// `base_ref` (first-time introduction of the crate); any other git
/// error is surfaced as an `anyhow::Error`.
fn load_base(base_ref: &str, relative_path: &str) -> anyhow::Result<Expectations> {
    let target = format!("{base_ref}:{relative_path}");
    let output = std::process::Command::new("git")
        .args(["show", &target])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn `git show {target}`: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // A missing file at the base ref just means the expectations
        // file was added in this branch; treat that as an empty base.
        if stderr.contains("does not exist") || stderr.contains("exists on disk, but not in") {
            warn!("expectations.toml not present at {base_ref}; treating base as empty");
            return Ok(Expectations {
                entries: Default::default(),
            });
        }
        return Err(anyhow::anyhow!(
            "`git show {target}` failed: {}",
            stderr.trim()
        ));
    }

    let text = String::from_utf8(output.stdout)
        .map_err(|e| anyhow::anyhow!("git show returned non-utf8 bytes: {e}"))?;
    Expectations::from_toml(PathBuf::from(target), &text)
        .map_err(|e| anyhow::anyhow!("parse base expectations: {e}"))
}

/// Absolute path to the repository root, derived from
/// `CARGO_MANIFEST_DIR` (which for xtask resolves to `<repo>/xtask`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate has no parent directory")
        .to_path_buf()
}

/// Categorized view of a diff computed by [`diff`]. All vectors are
/// sorted by test name so the output order is stable across runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Changes {
    pub added: Vec<(String, Status)>,
    pub removed: Vec<(String, Status)>,
    pub promotions: Vec<(String, Status)>,
    pub regressions: Vec<(String, Status)>,
    pub sideways: Vec<(String, Status, Status)>,
}

impl Changes {
    /// Total number of changes across every category.
    pub fn total(&self) -> usize {
        self.added.len()
            + self.removed.len()
            + self.promotions.len()
            + self.regressions.len()
            + self.sideways.len()
    }

    /// Whether no changes were detected at all.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

/// Compute a categorized diff between two expectations sets. Pure
/// function — all of the git / filesystem work happens above. Tested
/// directly in the unit tests below.
pub fn diff(base: &Expectations, current: &Expectations) -> Changes {
    let mut out = Changes::default();

    for (name, cur_entry) in &current.entries {
        match base.entries.get(name) {
            None => out.added.push((name.clone(), cur_entry.status)),
            Some(base_entry) if base_entry.status == cur_entry.status => {
                // unchanged
            }
            Some(base_entry) => {
                let from = base_entry.status;
                let to = cur_entry.status;
                if to == Status::Pass {
                    out.promotions.push((name.clone(), from));
                } else if from == Status::Pass {
                    out.regressions.push((name.clone(), to));
                } else {
                    out.sideways.push((name.clone(), from, to));
                }
            }
        }
    }

    for (name, base_entry) in &base.entries {
        if !current.entries.contains_key(name) {
            out.removed.push((name.clone(), base_entry.status));
        }
    }

    // `entries` is a BTreeMap, so iteration order is already sorted
    // by key, but the `for name in &current.entries` loop only covers
    // tests in current — removals come from a second loop and could
    // in principle land out of order if multiple are removed. An
    // explicit sort keeps the output deterministic regardless.
    out.added.sort();
    out.removed.sort();
    out.promotions.sort();
    out.regressions.sort();
    out.sideways.sort();

    out
}

/// Render the diff as plain text for terminal output. The format is
/// deliberately wordy: a human reading this is either triaging a
/// local run or reviewing a CI log where verbose context is cheap.
pub fn render_text(base_ref: &str, changes: &Changes) -> String {
    let mut out = String::new();
    out.push_str(&format!("expectations delta against `{base_ref}`\n\n"));

    if changes.is_empty() {
        out.push_str("  (no changes)\n");
        return out;
    }

    let section = |out: &mut String, title: &str, count: usize, items: Vec<String>| {
        if count == 0 {
            return;
        }
        out.push_str(&format!("{title} ({count}):\n"));
        for line in items {
            out.push_str(&format!("  - {line}\n"));
        }
        out.push('\n');
    };

    section(
        &mut out,
        "promotions (→ pass)",
        changes.promotions.len(),
        changes
            .promotions
            .iter()
            .map(|(n, from)| format!("{n} (was {})", from.as_str()))
            .collect(),
    );
    section(
        &mut out,
        "regressions (pass → …)",
        changes.regressions.len(),
        changes
            .regressions
            .iter()
            .map(|(n, to)| format!("{n} (now {})", to.as_str()))
            .collect(),
    );
    section(
        &mut out,
        "sideways",
        changes.sideways.len(),
        changes
            .sideways
            .iter()
            .map(|(n, from, to)| format!("{n} ({} → {})", from.as_str(), to.as_str()))
            .collect(),
    );
    section(
        &mut out,
        "added",
        changes.added.len(),
        changes
            .added
            .iter()
            .map(|(n, s)| format!("{n} ({})", s.as_str()))
            .collect(),
    );
    section(
        &mut out,
        "removed",
        changes.removed.len(),
        changes
            .removed
            .iter()
            .map(|(n, s)| format!("{n} (was {})", s.as_str()))
            .collect(),
    );

    out
}

/// Render the diff as GitHub-flavored markdown suitable for a PR
/// comment. Matches the shape #315 sketches in its "Smart bits"
/// section: headline counts up top, then per-category bullets.
pub fn render_markdown(base_ref: &str, changes: &Changes) -> String {
    let mut out = String::new();
    out.push_str("## onnx-official-tests expectations delta\n\n");
    out.push_str(&format!("Compared to `{base_ref}`:\n\n"));

    if changes.is_empty() {
        out.push_str("_No changes._\n");
        return out;
    }

    // Headline counts so reviewers can eyeball the shape without
    // expanding every section.
    out.push_str(&format!(
        "- **{}** promotion{} (→ pass)\n",
        changes.promotions.len(),
        plural(changes.promotions.len())
    ));
    out.push_str(&format!(
        "- **{}** regression{} (pass → …)\n",
        changes.regressions.len(),
        plural(changes.regressions.len())
    ));
    // `sideways` is not a plural of `sideway`; it's the name of the
    // category regardless of count. Printing "1 sideway" would read
    // as a typo.
    out.push_str(&format!("- **{}** sideways\n", changes.sideways.len()));
    out.push_str(&format!("- **{}** added\n", changes.added.len()));
    out.push_str(&format!("- **{}** removed\n\n", changes.removed.len()));

    // Detailed bullets per category. Using collapsible `<details>`
    // keeps the comment short by default but lets reviewers click to
    // see the full list when a category has more than a handful of
    // entries.
    fn details(out: &mut String, title: &str, count: usize, items: impl Iterator<Item = String>) {
        if count == 0 {
            return;
        }
        out.push_str(&format!(
            "<details><summary>{title} ({count})</summary>\n\n"
        ));
        for line in items {
            out.push_str(&format!("- {line}\n"));
        }
        out.push_str("\n</details>\n\n");
    }

    details(
        &mut out,
        "Promotions",
        changes.promotions.len(),
        changes
            .promotions
            .iter()
            .map(|(n, from)| format!("`{n}` (was `{}`)", from.as_str())),
    );
    details(
        &mut out,
        "Regressions",
        changes.regressions.len(),
        changes
            .regressions
            .iter()
            .map(|(n, to)| format!("`{n}` (now `{}`)", to.as_str())),
    );
    details(
        &mut out,
        "Sideways",
        changes.sideways.len(),
        changes
            .sideways
            .iter()
            .map(|(n, from, to)| format!("`{n}` (`{}` → `{}`)", from.as_str(), to.as_str())),
    );
    details(
        &mut out,
        "Added",
        changes.added.len(),
        changes
            .added
            .iter()
            .map(|(n, s)| format!("`{n}` (`{}`)", s.as_str())),
    );
    details(
        &mut out,
        "Removed",
        changes.removed.len(),
        changes
            .removed
            .iter()
            .map(|(n, s)| format!("`{n}` (was `{}`)", s.as_str())),
    );

    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expectations_schema::Entry;
    use std::collections::BTreeMap;

    fn entries(items: &[(&str, Status)]) -> Expectations {
        let mut map = BTreeMap::new();
        for (name, status) in items {
            map.insert(
                (*name).to_string(),
                Entry {
                    status: *status,
                    reason: None,
                    tracking: None,
                    wontfix: false,
                },
            );
        }
        Expectations { entries: map }
    }

    /// Identical expectations must produce an empty diff — the base
    /// case the CI runs most often (typical PR touches no tests).
    #[test]
    fn empty_diff_on_identical_input() {
        let base = entries(&[
            ("test_abs", Status::Pass),
            ("test_pad", Status::SkipCodegen),
        ]);
        let current = base.clone();
        let d = diff(&base, &current);
        assert!(d.is_empty());
        assert_eq!(d.total(), 0);
    }

    /// A non-pass → pass transition is classified as a promotion,
    /// regardless of which non-pass status it started in.
    #[test]
    fn promotions_from_every_source_status() {
        let sources = [
            Status::SkipCodegen,
            Status::SkipCompile,
            Status::FailCompare,
            Status::Flaky,
        ];
        for from in sources {
            let base = entries(&[("test_x", from)]);
            let current = entries(&[("test_x", Status::Pass)]);
            let d = diff(&base, &current);
            assert_eq!(d.promotions, vec![("test_x".to_string(), from)]);
            assert!(d.regressions.is_empty());
            assert!(d.sideways.is_empty());
        }
    }

    /// A pass → anything transition is classified as a regression.
    /// Pass → pass is a no-op (tested above).
    #[test]
    fn regressions_from_pass() {
        let targets = [
            Status::SkipCodegen,
            Status::SkipCompile,
            Status::FailCompare,
            Status::Flaky,
        ];
        for to in targets {
            let base = entries(&[("test_x", Status::Pass)]);
            let current = entries(&[("test_x", to)]);
            let d = diff(&base, &current);
            assert_eq!(d.regressions, vec![("test_x".to_string(), to)]);
            assert!(d.promotions.is_empty());
            assert!(d.sideways.is_empty());
        }
    }

    /// Non-pass → different non-pass is classified as sideways. The
    /// realistic case is `skip-codegen` → `skip-compile` when a fix
    /// advances a test past codegen but stops short of compile.
    #[test]
    fn sideways_between_non_pass_states() {
        let base = entries(&[("test_x", Status::SkipCodegen)]);
        let current = entries(&[("test_x", Status::SkipCompile)]);
        let d = diff(&base, &current);
        assert_eq!(
            d.sideways,
            vec![(
                "test_x".to_string(),
                Status::SkipCodegen,
                Status::SkipCompile
            )]
        );
    }

    /// Tests added in the current branch but absent from base show
    /// up in `added`. Tests removed show up in `removed`.
    #[test]
    fn added_and_removed() {
        let base = entries(&[("test_keep", Status::Pass), ("test_gone", Status::Pass)]);
        let current = entries(&[
            ("test_keep", Status::Pass),
            ("test_new", Status::SkipCodegen),
        ]);
        let d = diff(&base, &current);
        assert_eq!(d.added, vec![("test_new".to_string(), Status::SkipCodegen)]);
        assert_eq!(d.removed, vec![("test_gone".to_string(), Status::Pass)]);
        assert!(d.promotions.is_empty());
        assert!(d.regressions.is_empty());
    }

    /// Text output for an empty diff produces a single "(no changes)"
    /// line — important because CI should not post a PR comment when
    /// nothing actually changed.
    #[test]
    fn text_render_empty_diff() {
        let d = Changes::default();
        let out = render_text("origin/main", &d);
        assert!(out.contains("(no changes)"));
    }

    /// Markdown output for a non-empty diff includes the headline
    /// counts and at least one `<details>` section. We don't assert
    /// exact line-by-line output because the goal is to pin the
    /// structural shape, not the exact formatting.
    #[test]
    fn markdown_render_shape() {
        let base = entries(&[("test_a", Status::SkipCodegen)]);
        let current = entries(&[("test_a", Status::Pass)]);
        let d = diff(&base, &current);
        let out = render_markdown("origin/main", &d);
        assert!(out.starts_with("## onnx-official-tests expectations delta"));
        assert!(out.contains("**1** promotion"));
        assert!(out.contains("<details><summary>Promotions"));
        assert!(out.contains("`test_a` (was `skip-codegen`)"));
    }

    /// Pluralization: 1 promotion vs N promotions. Tiny but catches
    /// the easy off-by-one mistake of printing "1 promotions".
    #[test]
    fn markdown_plural_forms() {
        assert_eq!(plural(0), "s");
        assert_eq!(plural(1), "");
        assert_eq!(plural(2), "s");
    }
}
