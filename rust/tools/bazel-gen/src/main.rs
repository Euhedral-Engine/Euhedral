//! Writes the BUILD files of the workspace from `cargo metadata`, so the two builds cannot drift apart.
//!
//! ```text
//! cargo run -p bazel-gen                 write every BUILD.bazel
//! cargo run -p bazel-gen -- -mode=diff   write nothing; fail and name the stale files
//! ```
//!
//! One feature world: every member carries the features `cargo test --workspace` gives it (its defaults, the features
//! its dependents and its own dev-dependencies ask for, and what those enable). Targets come from the metadata, never
//! from guesses: a library, its unit tests and doctests, each integration test, each binary and its unit tests.
//! Anything the generator does not model stops it with a message.

mod model;
mod render;
mod scan;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use model::{Workspace, load};

fn main() -> ExitCode {
    let diff = env::args().skip(1).any(|arg| arg == "-mode=diff");
    match run(diff) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            report(&format!("bazel-gen: {message}"));
            ExitCode::FAILURE
        }
    }
}

fn report(line: &str) {
    // A failed write to stderr has nowhere to be reported.
    let _ = writeln!(io::stderr(), "{line}");
}

/// Returns whether every BUILD file is current (always true when writing).
fn run(diff: bool) -> Result<bool, String> {
    let root = repository_root()?;
    let metadata = cargo_metadata(&root.join("rust"))?;
    let workspace = load(&metadata, &root)?;
    let files = render::files(&workspace)?;
    let mut stale = Vec::new();
    for (path, text) in &files {
        let file = root.join(path);
        let current = fs::read_to_string(&file).ok();
        if current.as_deref() == Some(text.as_str()) {
            continue;
        }
        if diff {
            stale.push(path.clone());
        } else {
            fs::write(&file, text).map_err(|error| format!("write {path}: {error}"))?;
            report(&format!("wrote {path}"));
        }
    }
    let mut problems = pins(&root)?;
    problems.extend(crate_lock(&root)?);
    problems.extend(rustee_rev(&root)?);
    problems.extend(stray_build_files(&root, &workspace, &files)?);
    if diff {
        for path in &stale {
            report(&format!("stale: {path}"));
        }
        for problem in &problems {
            report(problem);
        }
        if !stale.is_empty() || !problems.is_empty() {
            report(
                "regenerate the BUILD files with `bazel run //:gazelle` (or `cargo run -p bazel-gen`), fix what is named above, and commit",
            );
            return Ok(false);
        }
    }
    Ok(true)
}

/// The repository root: Bazel's workspace, or the nearest directory above the working directory with a MODULE.bazel.
fn repository_root() -> Result<PathBuf, String> {
    if let Some(dir) = env::var_os("BUILD_WORKSPACE_DIRECTORY") {
        return Ok(PathBuf::from(dir));
    }
    let start = env::current_dir().map_err(|error| error.to_string())?;
    start
        .ancestors()
        .find(|dir| dir.join("MODULE.bazel").is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| "no MODULE.bazel above the working directory".to_owned())
}

fn cargo_metadata(workspace_dir: &Path) -> Result<String, String> {
    let mut cargo = env::var_os("CARGO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cargo"));
    // Under `bazel run` the toolchain's cargo is a path relative to the runfiles, which is the working directory.
    if cargo.components().count() > 1 && cargo.is_relative() {
        cargo = env::current_dir().map_err(|error| error.to_string())?.join(cargo);
    }
    let output = Command::new(&cargo)
        .args(["metadata", "--format-version", "1", "--no-deps", "--offline"])
        .current_dir(workspace_dir)
        .output()
        .map_err(|error| format!("run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

/// The toolchain version is written in four places (rust-toolchain.toml, mise.toml, and twice in MODULE.bazel, which
/// takes only a literal): they must agree.
fn pins(root: &Path) -> Result<Vec<String>, String> {
    let read = |path: &str| fs::read_to_string(root.join(path)).map_err(|error| format!("read {path}: {error}"));
    // The first quoted string of the first line that starts with `prefix`.
    let first_quoted = |text: &str, prefix: &str| -> Option<String> {
        let line = text.lines().find(|line| line.trim_start().starts_with(prefix))?;
        Some(line.split('"').nth(1)?.to_owned())
    };
    let toolchain = read("rust/rust-toolchain.toml")?;
    let want = first_quoted(&toolchain, "channel").ok_or("rust/rust-toolchain.toml has no channel")?;
    let mise = read("mise.toml")?;
    let module = read("MODULE.bazel")?;
    let found = [
        ("mise.toml", first_quoted(&mise, "rust = {")),
        ("MODULE.bazel rust.toolchain", first_quoted(&module, "versions = [")),
        ("MODULE.bazel host_tools", first_quoted(&module, "version = \"1.")),
    ];
    Ok(found
        .into_iter()
        .filter(|(_, version)| version.as_deref() != Some(want.as_str()))
        .map(|(place, version)| {
            format!(
                "{place} pins {} but rust/rust-toolchain.toml says {want}",
                version.as_deref().unwrap_or("nothing")
            )
        })
        .collect())
}

/// A BUILD file in the workspace that the generator does not write is a hand-written one: refuse it.
fn stray_build_files(
    root: &Path,
    workspace: &Workspace,
    files: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    let mut strays = Vec::new();
    for package in &workspace.packages {
        let dir = root.join(&package.dir);
        for entry in walk(&dir)? {
            let name = entry.file_name().and_then(|name| name.to_str()).unwrap_or_default();
            if name == "BUILD" || name == "BUILD.bazel" {
                let relative = entry.strip_prefix(root).map_err(|error| error.to_string())?;
                let relative = relative.to_string_lossy().replace('\\', "/");
                if !files.contains_key(&relative) {
                    strays.push(format!("{relative} is a BUILD file the generator does not write"));
                }
            }
        }
    }
    Ok(strays)
}

fn walk(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in fs::read_dir(&next).map_err(|error| format!("read {}: {error}", next.display()))? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_dir() {
                if path.file_name().and_then(|name| name.to_str()) != Some("target") {
                    pending.push(path);
                }
            } else {
                found.push(path);
            }
        }
    }
    Ok(found)
}

/// crate_universe does not notice a Cargo.lock that moved on: its own lock (`rust/cargo-bazel-lock.json`) must list exactly
/// the packages of Cargo.lock, by name and version.
fn crate_lock(root: &Path) -> Result<Vec<String>, String> {
    let read = |path: &str| fs::read_to_string(root.join(path)).map_err(|error| format!("read {path}: {error}"));
    let mut cargo: BTreeSet<String> = BTreeSet::new();
    let mut name = String::new();
    for line in read("rust/Cargo.lock")?.lines() {
        if let Some(value) = line.strip_prefix("name = \"") {
            name = value.trim_end_matches('"').to_owned();
        } else if let Some(value) = line.strip_prefix("version = \"") {
            cargo.insert(format!("{name} {}", value.trim_end_matches('"')));
        }
    }
    let lock: serde_json::Value =
        serde_json::from_str(&read("rust/cargo-bazel-lock.json")?).map_err(|error| error.to_string())?;
    let bazel: BTreeSet<String> = lock
        .get("crates")
        .and_then(serde_json::Value::as_object)
        .ok_or("rust/cargo-bazel-lock.json has no crates")?
        .keys()
        .cloned()
        .collect();
    let mut problems = Vec::new();
    for missing in cargo.difference(&bazel) {
        problems.push(format!(
            "rust/cargo-bazel-lock.json lacks {missing}, which rust/Cargo.lock has"
        ));
    }
    for extra in bazel.difference(&cargo) {
        problems.push(format!(
            "rust/cargo-bazel-lock.json has {extra}, which rust/Cargo.lock does not"
        ));
    }
    if !problems.is_empty() {
        problems
            .push("repin: CARGO_BAZEL_REPIN=1 bazel build //rust/... and commit rust/cargo-bazel-lock.json".to_owned());
    }
    Ok(problems)
}

/// Cargo and Bazel must build the same RustEE: the commit of the `git_override` of `euhedral_rustee` in MODULE.bazel is the
/// `rev` of the RustEE dependencies in rust/Cargo.toml.
fn rustee_rev(root: &Path) -> Result<Vec<String>, String> {
    let read = |path: &str| fs::read_to_string(root.join(path)).map_err(|error| format!("read {path}: {error}"));
    let quoted_after = |text: &str, key: &str| -> BTreeSet<String> {
        text.lines()
            .filter_map(|line| {
                let rest = line.split(key).nth(1)?;
                Some(rest.split('"').nth(1)?.to_owned())
            })
            .collect()
    };
    let cargo = quoted_after(&read("rust/Cargo.toml")?, "rev = ");
    let module = read("MODULE.bazel")?;
    let bazel = quoted_after(&module, "commit = ");
    if cargo.len() == 1 && cargo == bazel {
        return Ok(Vec::new());
    }
    Ok(vec![format!(
        "RustEE rev: rust/Cargo.toml says {cargo:?} but the git_override in MODULE.bazel says {bazel:?}; they must be one commit"
    )])
}
