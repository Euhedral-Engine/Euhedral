//! Repository tasks.
//!
//! `cargo xtask bindings` regenerates `crates/cuda/src/sys.rs` from `native/include/euhedral_cuda.h` with the
//! `bindgen` command, which must be exactly version 0.72.1 (`cargo install bindgen-cli --version 0.72.1 --locked`) so
//! the output does not drift. bindgen also needs libclang; its version is not pinned (the generated declarations do
//! not depend on it for this header). With `--check` it regenerates into `target/` and fails when the checked-in file
//! differs.

// A build tool, not a library crate: it has no clippy.toml of the no-wait bans, and its output, its `expect` and its
// exit are its interface.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    reason = "a binary: it reports on the terminal and may abort"
)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const HEADER: &str = "native/include/euhedral_cuda.h";
const OUTPUT: &str = "rust/crates/cuda/src/sys.rs";
const CHECK_OUTPUT: &str = "rust/target/xtask/sys.rs";
const LIBRARY_STRUCT: &str = "EuhedralCuda";
const BINDGEN_VERSION: &str = "0.72.1";

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let result = match arguments.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["bindings"] => bindings(false),
        ["bindings", "--check"] => bindings(true),
        _ => Err("usage: cargo xtask bindings [--check]".to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The repository root: this crate sits at `<root>/rust/xtask`.
fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("xtask sits two levels below the repository root")
        .to_path_buf()
}

fn bindings(check: bool) -> Result<(), String> {
    let root = repository_root();
    let target = if check {
        root.join(CHECK_OUTPUT)
    } else {
        root.join(OUTPUT)
    };
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    generate(&root, &target)?;
    if !check {
        println!("wrote {}", root.join(OUTPUT).display());
        return Ok(());
    }
    let generated = fs::read(&target).map_err(|error| format!("cannot read {}: {error}", target.display()))?;
    let checked_in = fs::read(root.join(OUTPUT)).map_err(|error| format!("cannot read {OUTPUT}: {error}"))?;
    if generated == checked_in {
        println!("{OUTPUT} is up to date");
        Ok(())
    } else {
        Err(format!(
            "{OUTPUT} is stale: run `cargo xtask bindings` and commit it (diff against {})",
            target.display()
        ))
    }
}

/// Runs bindgen over the public header into `output`.
///
/// Every declared function must be exported by the library (`--dynamic-link-require-all`), so a library that lacks one
/// fails to load with the symbol's name. `euhedral_cuda_submission_stream` is declared for the library's own sources
/// and is not exported.
fn generate(root: &Path, output: &Path) -> Result<(), String> {
    require_bindgen_version()?;
    let status = Command::new("bindgen")
        .current_dir(root)
        .arg(HEADER)
        .arg("--output")
        .arg(output)
        .args([
            "--rust-target",
            "1.90",
            "--rust-edition",
            "2024",
            "--formatter",
            "prettyplease",
        ])
        .args(["--dynamic-loading", LIBRARY_STRUCT, "--dynamic-link-require-all"])
        .args(["--allowlist-function", "euhedral_cuda_.*"])
        .args(["--blocklist-function", "euhedral_cuda_submission_stream"])
        .args(["--allowlist-var", "EUHEDRAL_CUDA_.*"])
        .args(["--no-doc-comments", "--no-layout-tests"])
        .status()
        .map_err(|error| {
            format!(
                "cannot run bindgen ({error}); install it with `cargo install bindgen-cli --version 0.72.1 --locked`"
            )
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("bindgen failed: {status}"))
    }
}

/// Fails unless `bindgen --version` reports [`BINDGEN_VERSION`].
fn require_bindgen_version() -> Result<(), String> {
    let install = format!("install it with `cargo install bindgen-cli --version {BINDGEN_VERSION} --locked`");
    let output = Command::new("bindgen")
        .arg("--version")
        .output()
        .map_err(|error| format!("cannot run bindgen ({error}); {install}"))?;
    let reported = String::from_utf8_lossy(&output.stdout);
    let version = reported.split_whitespace().last().unwrap_or_default();
    if version == BINDGEN_VERSION {
        Ok(())
    } else {
        Err(format!(
            "bindgen {BINDGEN_VERSION} is required, found {reported:?}; {install}"
        ))
    }
}
