//! `cargo xtask bundle`: a runnable server bundle for one native product.
//!
//! The bundle directory holds `bin/euhedral-server` and, beside `bin/`, the native product's `lib/`,
//! `share/euhedral_cuda/` and `cuda/include/` (the tree `cargo xtask native` leaves under `build/native/<product>`).
//! The server finds its CUDA library there without `host.native-cache`. The directory is archived next to itself:
//! `.tar.gz` for Linux, `.zip` for Windows.
//!
//! A product is built on its own platform. The one cross-build is windows-x64 from Linux x86-64, with `cargo zigbuild`
//! for `x86_64-pc-windows-gnu` (zig is the linker, and the CUDA library comes from `cargo xtask native`, which builds
//! both products on Linux); the zip is written in process, so no host `tar` has to know the format.

use std::env;
use std::fs::{self, File};
use std::io::{self, Error};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[cfg(unix)]
use std::os::unix::fs::symlink;

use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use crate::repository_root;

/// The Rust target the Windows server is cross-built for.
const WINDOWS_TARGET: &str = "x86_64-pc-windows-gnu";

/// The native product trees that go into a bundle, relative to the product directory.
/// The manifest of the installed sources, at the root of the native product.
const MANIFEST: &str = "files.sha256";

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

    /// How this process builds the product's server, or why it cannot.
    fn build(self) -> Result<Build, String> {
        self.build_on(env::consts::OS, env::consts::ARCH)
    }

    /// `build` for a host of the given operating system and architecture (`std::env::consts` names).
    fn build_on(self, os: &str, arch: &str) -> Result<Build, String> {
        match (os, arch, self) {
            ("linux", "x86_64", Self::LinuxX64) | ("windows", "x86_64", Self::WindowsX64) => Ok(Build::Native),
            ("linux", "x86_64", Self::WindowsX64) => Ok(Build::CrossWindows),
            _ => Err(format!(
                "the {} bundle cannot be built on {os} {arch}: linux-x64 builds on Linux x86_64, windows-x64 on \
                 Windows or Linux x86_64",
                self.id()
            )),
        }
    }

    fn library(self) -> &'static str {
        match self {
            Self::LinuxX64 => "lib/libeuhedral_cuda.so",
            Self::WindowsX64 => "lib/euhedral_cuda.dll",
        }
    }
}

/// How the server is compiled for a product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Build {
    /// `cargo build`: the product is this process's platform.
    Native,
    /// `cargo zigbuild` for the Windows target, from Linux.
    CrossWindows,
}

impl Build {
    /// The cargo arguments that build the server in release and report its executable as JSON.
    fn arguments(self) -> Vec<&'static str> {
        let (command, target): (&str, &[&str]) = match self {
            Self::Native => ("build", &[]),
            Self::CrossWindows => ("zigbuild", &["--target", WINDOWS_TARGET]),
        };
        [
            command,
            "--release",
            "--bin",
            "euhedral-server",
            "--message-format=json-render-diagnostics",
        ]
        .into_iter()
        .chain(target.iter().copied())
        .collect()
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
    let build = product.build()?;
    let root = repository_root();
    let native = native.unwrap_or_else(|| root.join("build/native").join(product.id()));
    let out = out.unwrap_or_else(|| root.join("build/bundle"));
    let executable = build_server(&root, build)?;
    let directory = out.join(format!("euhedral-server-{}", product.id()));
    assemble(&directory, &executable, &native, product)?;
    let archive = archive(&directory, product)?;
    println!("bundle {}", directory.display());
    println!("archive {}", archive.display());
    Ok(())
}

/// Builds the server in release; returns the executable the compiler reports.
fn build_server(root: &Path, build: Build) -> Result<PathBuf, String> {
    let output = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root.join("rust"))
        .args(build.arguments())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if !output.status.success() {
        return Err(format!("building the server failed: {}", output.status));
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.contains(r#""reason":"compiler-artifact""#) && line.contains(r#""name":"euhedral-server""#))
        .filter_map(executable_of)
        .next_back()
        .ok_or_else(|| "cargo did not report the server executable".to_string())
}

/// The `executable` field of a cargo JSON message line, unescaped.
fn executable_of(line: &str) -> Option<PathBuf> {
    let key = r#""executable":""#;
    let rest = &line[line.find(key)? + key.len()..];
    let mut path = String::new();
    let mut characters = rest.chars();
    loop {
        match characters.next()? {
            '"' => return Some(PathBuf::from(path)),
            '\\' => path.push(characters.next()?),
            other => path.push(other),
        }
    }
}

/// Writes `directory` from the server `executable` and the native product at `native`. An existing `directory` is
/// replaced.
fn assemble(directory: &Path, executable: &Path, native: &Path, product: Product) -> Result<(), String> {
    if !is_filled(executable) {
        return Err(format!("the server is missing or empty at {}", executable.display()));
    }
    for file in [product.library(), MANIFEST] {
        if !is_filled(&native.join(file)) {
            return Err(format!(
                "{} is missing or has an empty {file}: build the native product first (`cargo xtask native`)",
                native.display()
            ));
        }
    }
    for tree in TREES {
        if !native.join(tree).is_dir() {
            return Err(format!(
                "{} is missing {tree}: build the native product first (`cargo xtask native`)",
                native.display()
            ));
        }
    }
    if directory.exists() {
        fs::remove_dir_all(directory).map_err(|error| io_error("remove", directory, &error))?;
    }
    let bin = directory.join("bin");
    fs::create_dir_all(&bin).map_err(|error| io_error("create", &bin, &error))?;
    fs::copy(executable, bin.join(product.executable())).map_err(|error| io_error("copy", executable, &error))?;
    for tree in TREES {
        copy_tree(&native.join(tree), &directory.join(tree))?;
    }
    // The kernel sources' manifest, which the library reads from the root of the installed tree.
    fs::copy(native.join(MANIFEST), directory.join(MANIFEST))
        .map_err(|error| io_error("copy", &native.join(MANIFEST), &error))?;
    Ok(())
}

/// Whether `path` is a regular file with content: an empty file is what a failed copy or build leaves.
fn is_filled(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
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

/// Archives `directory` next to itself: a `.tar.gz` made by the system `tar` for Linux, a `.zip` written here for Windows.
fn archive(directory: &Path, product: Product) -> Result<PathBuf, String> {
    let name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("bad bundle name")?;
    let parent = directory.parent().ok_or("bad bundle directory")?;
    match product {
        Product::LinuxX64 => {
            let archive = parent.join(format!("{name}.tar.gz"));
            let status = Command::new("tar")
                .arg("-czf")
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
        Product::WindowsX64 => {
            let archive = parent.join(format!("{name}.zip"));
            write_zip(directory, name, &archive)?;
            Ok(archive)
        }
    }
}

/// Writes every file under `directory` into the zip `archive`, under the top directory `name`, in name order.
fn write_zip(directory: &Path, name: &str, archive: &Path) -> Result<(), String> {
    let file = File::create(archive).map_err(|error| io_error("create", archive, &error))?;
    let mut writer = ZipWriter::new(file);
    add_tree(&mut writer, directory, name)?;
    writer
        .finish()
        .map(drop)
        .map_err(|error| format!("cannot finish {}: {error}", archive.display()))
}

/// Adds the files under `from` to `writer` as `prefix/<relative name>`.
fn add_tree(writer: &mut ZipWriter<File>, from: &Path, prefix: &str) -> Result<(), String> {
    let mut entries = fs::read_dir(from)
        .map_err(|error| io_error("read", from, &error))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| io_error("read", from, &error))?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let name = format!("{prefix}/{}", entry.file_name().to_string_lossy());
        if path.is_dir() {
            add_tree(writer, &path, &name)?;
            continue;
        }
        let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        writer
            .start_file(&name, options)
            .map_err(|error| format!("cannot add {name}: {error}"))?;
        let mut source = File::open(&path).map_err(|error| io_error("open", &path, &error))?;
        io::copy(&mut source, writer).map_err(|error| io_error("archive", &path, &error))?;
    }
    Ok(())
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
            ("files.sha256", "digests"),
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
        assemble(&bundle, &server, &native, Product::LinuxX64).expect("assemble");
        for file in [
            "bin/euhedral-server",
            "lib/libeuhedral_cuda.so",
            "lib/libnvrtc-builtins.so.13.1",
            "share/euhedral_cuda/kernel.cu",
            "cuda/include/cuda.h",
            "files.sha256",
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
        let error = assemble(&root.join("bundle"), &server, &native, Product::LinuxX64).expect_err("refused");
        assert!(error.contains("cuda/include"), "{error}");
        drop(fs::remove_dir_all(root));
    }

    #[test]
    fn each_product_is_built_only_on_the_platforms_that_can() {
        let (linux, windows) = (Product::LinuxX64, Product::WindowsX64);
        assert_eq!(linux.build_on("linux", "x86_64"), Ok(Build::Native));
        assert_eq!(windows.build_on("windows", "x86_64"), Ok(Build::Native));
        assert_eq!(windows.build_on("linux", "x86_64"), Ok(Build::CrossWindows));
        for (product, os, arch) in [
            (linux, "windows", "x86_64"),
            (linux, "macos", "x86_64"),
            (windows, "macos", "x86_64"),
            (linux, "linux", "aarch64"),
            (windows, "linux", "aarch64"),
            (windows, "windows", "aarch64"),
        ] {
            let error = product.build_on(os, arch).expect_err("refused");
            assert!(
                error.contains(product.id()) && error.contains(os) && error.contains(arch),
                "{error}"
            );
        }
    }

    #[test]
    fn the_cross_build_goes_through_zigbuild_to_the_windows_target_and_the_native_build_has_no_target() {
        let cross = Build::CrossWindows.arguments();
        assert_eq!(cross[0], "zigbuild");
        assert!(
            cross.windows(2).any(|pair| pair == ["--target", WINDOWS_TARGET]),
            "{cross:?}"
        );
        assert!(!Build::Native.arguments().contains(&"--target"));
    }

    fn windows_native_product(root: &Path) -> PathBuf {
        let native = native_product(root);
        fs::rename(
            native.join("lib/libeuhedral_cuda.so"),
            native.join("lib/euhedral_cuda.dll"),
        )
        .expect("rename");
        native
    }

    #[test]
    fn an_empty_library_or_server_is_refused() {
        let root = scratch("empty");
        let native = windows_native_product(&root);
        let server = root.join("server.exe");
        fs::write(&server, "binary").expect("server");
        fs::write(native.join("lib/euhedral_cuda.dll"), "").expect("empty");
        let error = assemble(&root.join("bundle"), &server, &native, Product::WindowsX64).expect_err("empty library");
        assert!(error.contains("lib/euhedral_cuda.dll"), "{error}");
        fs::write(native.join("lib/euhedral_cuda.dll"), "dll").expect("library");
        fs::write(&server, "").expect("empty server");
        let error = assemble(&root.join("bundle"), &server, &native, Product::WindowsX64).expect_err("empty server");
        assert!(error.contains("server"), "{error}");
        fs::write(&server, "binary").expect("server");
        fs::write(native.join("files.sha256"), "").expect("empty manifest");
        let error = assemble(&root.join("bundle"), &server, &native, Product::WindowsX64).expect_err("empty manifest");
        assert!(error.contains("files.sha256"), "{error}");
        drop(fs::remove_dir_all(root));
    }

    #[test]
    fn the_windows_archive_is_a_zip_of_the_bundle_under_its_own_name() {
        let root = scratch("zip");
        let native = windows_native_product(&root);
        let server = root.join("server.exe");
        fs::write(&server, "binary").expect("server");
        let bundle = root.join("out/euhedral-server-windows-x64");
        assemble(&bundle, &server, &native, Product::WindowsX64).expect("assemble");
        let archive = archive(&bundle, Product::WindowsX64).expect("archive");
        assert_eq!(archive, root.join("out/euhedral-server-windows-x64.zip"));
        let mut zip = zip::ZipArchive::new(File::open(&archive).expect("open")).expect("zip");
        let mut names: Vec<String> = zip.file_names().map(str::to_string).collect();
        names.sort();
        let prefix = "euhedral-server-windows-x64";
        let mut wanted: Vec<String> = [
            "bin/euhedral-server.exe",
            "cuda/include/cuda.h",
            "files.sha256",
            "lib/euhedral_cuda.dll",
            "lib/libnvrtc-builtins.so.13.1",
            "share/euhedral_cuda/kernel.cu",
        ]
        .iter()
        .map(|name| format!("{prefix}/{name}"))
        .collect();
        wanted.sort();
        assert_eq!(names, wanted);
        let mut content = String::new();
        io::Read::read_to_string(
            &mut zip
                .by_name(&format!("{prefix}/bin/euhedral-server.exe"))
                .expect("entry"),
            &mut content,
        )
        .expect("read");
        assert_eq!(content, "binary");
        drop(fs::remove_dir_all(root));
    }

    #[test]
    fn the_executable_is_read_from_the_compiler_message() {
        let line = r#"{"reason":"compiler-artifact","target":{"name":"euhedral-server"},"executable":"C:\\t\"x\"/s"}"#;
        assert_eq!(executable_of(line), Some(PathBuf::from(r#"C:\t"x"/s"#)));
    }

    #[test]
    fn an_unknown_product_is_refused() {
        assert!(Product::parse("macos").is_err());
        assert_eq!(Product::parse("linux-x64"), Ok(Product::LinuxX64));
    }
}
