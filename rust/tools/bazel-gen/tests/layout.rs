//! The BUILD files match the Cargo manifests: a changed dependency or feature list without regenerated BUILD files
//! fails here, under `bazel test` and under `cargo test` alike.

#![cfg(unix)] // The BUILD files are checked on Linux; Windows checks out the same files.

use std::env;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn build_files_match_the_manifests() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_bazel-gen"));
    command.arg("-mode=diff");
    if let Some(root) = bazel_workspace() {
        command.env("BUILD_WORKSPACE_DIRECTORY", root);
    }
    if let (Some(dir), Some(location)) = (env::var_os("TEST_SRCDIR"), env::var_os("CARGO_RLOCATION")) {
        command.env("CARGO", PathBuf::from(dir).join(location));
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => panic!("bazel-gen did not start: {error}"),
    };
    assert!(
        output.status.success(),
        "BUILD files are stale:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Under Bazel the test runs in a tree of symlinks; the real workspace is where `MODULE.bazel` points.
fn bazel_workspace() -> Option<PathBuf> {
    env::var_os("TEST_SRCDIR")?;
    let module = PathBuf::from("MODULE.bazel").canonicalize().ok()?;
    module.parent().map(PathBuf::from)
}
