//! `cargo xtask bundle`: a runnable server bundle for one native product.
//!
//! The bundle directory holds `bin/euhedral-server` and, beside `bin/`, the native product's `lib/`,
//! `share/euhedral_cuda/` and `cuda/include/` (the tree the Gradle native build leaves under `build/native/<product>`
//! and the Java server unpacks into its native cache). The server finds its CUDA library there without
//! `host.native-cache`. The directory is archived next to itself: `.tar.gz` for Linux, `.zip` for Windows.

use std::env;
use std::fs;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use std::os::unix::fs::symlink;

use crate::repository_root;

/// The native product trees that go into a bundle, relative to the product directory.
const TREES: [&str; 3] = ["lib", "share/euhedral_cuda", "cuda/include"];

/// A platform a bundle exists for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Product {
    LinuxX64,
    WindowsX64,
}

impl Product {
    fn parse(name: &str) -> Result<Self, String> {
        match name {
            "linux-x64" => Ok(Self::LinuxX64),
            "windows-x64" => Ok(Self::WindowsX64),
            other => Err(format!("unknown product {other:?}: expected linux-x64 or windows-x64")),
        }
    }

    fn id(self) -> &'static str {
        match self {
            Self::LinuxX64 => "linux-x64",
            Self::WindowsX64 => "windows-x64",
        }
    }

    fn executable(self) -> &'static str {
        match self {
            Self::LinuxX64 => "euhedral-server",
            Self::WindowsX64 => "euhedral-server.exe",
        }
    }

    fn is_host(self) -> bool {
        cfg!(windows) == (self == Self::WindowsX64)
    }
}

/// Parses the arguments, builds the server and writes the bundle and its archive.
pub fn run(arguments: &[&str]) -> Result<(), String> {
    let (mut product, mut native, mut out) = (None, None, None);
    let mut remaining = arguments.iter();
    while let Some(&flag) = remaining.next() {
        let value = remaining.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--product" => product = Some(Product::parse(value)?),
            "--native" => native = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let product = product.ok_or("--product <linux-x64|windows-x64> is required")?;
    if !product.is_host() {
        return Err(format!("the {} bundle is built on its own platform", product.id()));
    }
    let root = repository_root();
    let native = native.unwrap_or_else(|| root.join("build/native").join(product.id()));
    let out = out.unwrap_or_else(|| root.join("build/bundle"));
    let executable = build_server(&root, product)?;
    let directory = out.join(format!("euhedral-server-{}", product.id()));
    assemble(&directory, &executable, &native, product.executable())?;
    let archive = archive(&directory, product)?;
    println!("bundle {}", directory.display());
    println!("archive {}", archive.display());
    Ok(())
}

/// Builds the server in release; returns the executable.
fn build_server(root: &Path, product: Product) -> Result<PathBuf, String> {
    let status = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root.join("rust"))
        .args(["build", "--release", "--bin", "euhedral-server"])
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if !status.success() {
        return Err(format!("building the server failed: {status}"));
    }
    let target = env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("rust/target"), PathBuf::from);
    Ok(target.join("release").join(product.executable()))
}

/// Writes `directory` from the server `executable` and the native product at `native`. An existing `directory` is
/// replaced.
fn assemble(directory: &Path, executable: &Path, native: &Path, name: &str) -> Result<(), String> {
    if !executable.is_file() {
        return Err(format!("the server is not at {}", executable.display()));
    }
    for tree in TREES {
        if !native.join(tree).is_dir() {
            return Err(format!(
                "{} is missing {tree}: build the native product first (`gradle nativeVerify`)",
                native.display()
            ));
        }
    }
    if directory.exists() {
        fs::remove_dir_all(directory).map_err(|error| io_error("remove", directory, &error))?;
    }
    let bin = directory.join("bin");
    fs::create_dir_all(&bin).map_err(|error| io_error("create", &bin, &error))?;
    fs::copy(executable, bin.join(name)).map_err(|error| io_error("copy", executable, &error))?;
    for tree in TREES {
        copy_tree(&native.join(tree), &directory.join(tree))?;
    }
    Ok(())
}

/// Copies the directory `from` to `to`, keeping symbolic links as links where the platform has them.
fn copy_tree(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|error| io_error("create", to, &error))?;
    for entry in fs::read_dir(from).map_err(|error| io_error("read", from, &error))? {
        let entry = entry.map_err(|error| io_error("read", from, &error))?;
        let (source, target) = (entry.path(), to.join(entry.file_name()));
        let kind = entry.file_type().map_err(|error| io_error("stat", &source, &error))?;
        if kind.is_symlink() {
            copy_link(&source, &target)?;
        } else if kind.is_dir() {
            copy_tree(&source, &target)?;
        } else {
            fs::copy(&source, &target).map_err(|error| io_error("copy", &source, &error))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_link(source: &Path, target: &Path) -> Result<(), String> {
    let destination = fs::read_link(source).map_err(|error| io_error("read", source, &error))?;
    symlink(destination, target).map_err(|error| io_error("link", target, &error))
}

#[cfg(not(unix))]
fn copy_link(source: &Path, target: &Path) -> Result<(), String> {
    fs::copy(source, target)
        .map(drop)
        .map_err(|error| io_error("copy", source, &error))
}

/// Archives `directory` next to itself with the system `tar` (Windows ships bsdtar, which writes zip with `-a`).
fn archive(directory: &Path, product: Product) -> Result<PathBuf, String> {
    let name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("bad bundle name")?;
    let parent = directory.parent().ok_or("bad bundle directory")?;
    let archive = parent.join(match product {
        Product::LinuxX64 => format!("{name}.tar.gz"),
        Product::WindowsX64 => format!("{name}.zip"),
    });
    let mode = match product {
        Product::LinuxX64 => "-czf",
        Product::WindowsX64 => "-acf",
    };
    let status = Command::new("tar")
        .arg(mode)
        .arg(&archive)
        .arg("-C")
        .arg(parent)
        .arg(name)
        .status()
        .map_err(|error| format!("cannot run tar: {error}"))?;
    if status.success() {
        Ok(archive)
    } else {
        Err(format!("tar failed: {status}"))
    }
}

fn io_error(action: &str, path: &Path, error: &Error) -> String {
    format!("cannot {action} {}: {error}", path.display())
}

#[cfg(test)]
mod tests {
    use std::process::id;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let base = env::var_os("EUHEDRAL_SCRATCH").map_or_else(env::temp_dir, PathBuf::from);
        let path = base.join(format!("xtask-bundle-{name}-{}", id()));
        drop(fs::remove_dir_all(&path));
        fs::create_dir_all(&path).expect("scratch");
        path
    }

    fn native_product(root: &Path) -> PathBuf {
        let native = root.join("native");
        for (file, content) in [
            ("lib/libeuhedral_cuda.so", "lib"),
            ("lib/libnvrtc-builtins.so.13.1", "nvrtc"),
            ("share/euhedral_cuda/kernel.cu", "kernel"),
            ("cuda/include/cuda.h", "header"),
        ] {
            let path = native.join(file);
            fs::create_dir_all(path.parent().expect("parent")).expect("dir");
            fs::write(path, content).expect("file");
        }
        native
    }

    #[test]
    fn the_bundle_holds_the_binary_and_the_native_trees() {
        let root = scratch("layout");
        let native = native_product(&root);
        let server = root.join("server");
        fs::write(&server, "binary").expect("server");
        let bundle = root.join("out/bundle");
        assemble(&bundle, &server, &native, "euhedral-server").expect("assemble");
        for file in [
            "bin/euhedral-server",
            "lib/libeuhedral_cuda.so",
            "lib/libnvrtc-builtins.so.13.1",
            "share/euhedral_cuda/kernel.cu",
            "cuda/include/cuda.h",
        ] {
            assert!(bundle.join(file).is_file(), "{file}");
        }
        drop(fs::remove_dir_all(root));
    }

    #[test]
    fn a_native_product_without_a_tree_is_refused() {
        let root = scratch("missing");
        let native = native_product(&root);
        fs::remove_dir_all(native.join("cuda")).expect("remove");
        let server = root.join("server");
        fs::write(&server, "binary").expect("server");
        let error = assemble(&root.join("bundle"), &server, &native, "euhedral-server").expect_err("refused");
        assert!(error.contains("cuda/include"), "{error}");
        drop(fs::remove_dir_all(root));
    }

    #[test]
    fn an_unknown_product_is_refused() {
        assert!(Product::parse("macos").is_err());
        assert_eq!(Product::parse("linux-x64"), Ok(Product::LinuxX64));
    }
}
