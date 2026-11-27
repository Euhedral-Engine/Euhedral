//! `cargo xtask native` and `cargo xtask native-verify`: the CUDA library of each product in
//! `native/native-products.json`, built without Gradle.
//!
//! `native [--product <id>] [--force-download]` resolves the CUDA inputs of a product (the installed toolkit named by
//! `CUDA_HOME` or `CUDA_PATH` for the host's own product, otherwise the pinned redistributables downloaded into a
//! cache and unpacked under `build/cuda-dev/<id>`), checks that the headers and libraries belong together, runs
//! `zig build` in `native/` and completes the product tree under `build/native/<id>`: `lib/` (the library, the CUDA
//! runtime and NVRTC), `share/euhedral_cuda/` (the kernel sources), `cuda/include/` (the headers NVRTC compiles
//! against) and `files.sha256`. It writes no llguidance library: the Rust server links the crate.
//!
//! `native-verify [--product <id>]` checks a built tree: the kernel sources match `native/src`, the library is a
//! binary of the product's platform that imports the right runtime, the bundled runtime and headers are present and
//! `files.sha256` lists exactly the files there.
//!
//! Both take every product when `--product` is absent. The cache is `$XDG_CACHE_HOME/euhedral-cuda/<cuda version>` or,
//! without it, `xtask/cuda-cache/<cuda version>` under the Cargo target directory.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::imports::binary_imports;
use crate::redist::{self, Download};
use crate::repository_root;

const PRODUCTS: &str = "native/native-products.json";
const HASHES: &str = "files.sha256";
const COMPONENTS: [&str; 4] = ["cudart", "nvrtc", "crt", "cccl"];
const MINIMUM_CUDA: u32 = 13010;

/// The operating system a product is built for and runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Platform {
    Linux,
    Windows,
}

impl Platform {
    /// The platform of this process, when the native products run there.
    fn host() -> Option<Self> {
        match (cfg!(target_arch = "x86_64"), cfg!(windows), cfg!(target_os = "linux")) {
            (true, true, _) => Some(Self::Windows),
            (true, _, true) => Some(Self::Linux),
            _ => None,
        }
    }
}

/// A product of the manifest.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Product {
    id: String,
    zig_target: String,
    filename: String,
    cuda_platform: String,
    host_os: Platform,
}

/// A redistributable archive: its path under the download root and its SHA-256.
#[derive(Debug, Deserialize)]
struct Component {
    path: String,
    sha256: String,
}

/// `native/native-products.json`, less what only the Java build reads.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    cuda_version: String,
    products: Vec<Product>,
    redistributables: BTreeMap<String, BTreeMap<String, Component>>,
}

impl Manifest {
    fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(PRODUCTS);
        let text = fs::read_to_string(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        serde_json::from_str(&text).map_err(|error| format!("cannot parse {}: {error}", path.display()))
    }

    /// The product `id`, or every product when `id` is `None`.
    fn select(&self, id: Option<&str>) -> Result<Vec<&Product>, String> {
        let Some(id) = id else {
            return Ok(self.products.iter().collect());
        };
        match self.products.iter().find(|product| product.id == id) {
            Some(product) => Ok(vec![product]),
            None => {
                let known: Vec<&str> = self.products.iter().map(|product| product.id.as_str()).collect();
                Err(format!("unknown product {id:?}: expected one of {}", known.join(", ")))
            }
        }
    }
}

/// The arguments both commands take.
#[derive(Debug, Default, PartialEq, Eq)]
struct Options {
    product: Option<String>,
    force_download: bool,
}

fn parse_options(arguments: &[&str], force_download_allowed: bool) -> Result<Options, String> {
    let mut options = Options::default();
    let mut remaining = arguments.iter();
    while let Some(&flag) = remaining.next() {
        match flag {
            "--product" => {
                let value = remaining.next().ok_or("--product needs a value")?;
                options.product = Some((*value).to_string());
            }
            "--force-download" if force_download_allowed => options.force_download = true,
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(options)
}

/// Where a product's CUDA headers and libraries come from.
#[derive(Debug, PartialEq, Eq)]
enum Source {
    /// An installed toolkit, selected by the caller.
    Toolkit,
    /// The pinned redistributables, unpacked under `build/cuda-dev/<id>`.
    Download,
}

/// The CUDA inputs of one product build.
#[derive(Debug)]
struct Inputs {
    source: Source,
    include: PathBuf,
    library: PathBuf,
    /// The directory of the driver library to link (`libcuda.so`), or of the generated import libraries on Windows.
    driver: PathBuf,
    /// The directory of the runtime libraries the product ships.
    runtime: PathBuf,
    /// Where the Windows import libraries are generated.
    imports: PathBuf,
    /// The unpack directory of a download.
    unpacked: PathBuf,
}

impl Inputs {
    /// Plans the inputs of `product`: `toolkit` is the home of an installed toolkit to use, if any.
    fn plan(root: &Path, product: &Product, toolkit: Option<&Path>) -> Self {
        let platform = product.host_os;
        let unpacked = root.join("build/cuda-dev").join(&product.id);
        let imports = root.join("build/cuda-import").join(&product.id);
        let Some(home) = toolkit else {
            return Self {
                source: Source::Download,
                include: unpacked.join("include"),
                library: unpacked.join("lib"),
                driver: if platform == Platform::Windows {
                    imports.clone()
                } else {
                    unpacked.join("lib")
                },
                runtime: unpacked.join("runtime"),
                imports,
                unpacked,
            };
        };
        let (include, library, runtime, driver) = match platform {
            Platform::Windows => (
                home.join("include"),
                home.join("lib/x64"),
                home.join("bin/x64"),
                imports.clone(),
            ),
            Platform::Linux => {
                let tree = home.join("targets/x86_64-linux");
                (
                    tree.join("include"),
                    tree.join("lib"),
                    tree.join("lib"),
                    tree.join("lib/stubs"),
                )
            }
        };
        Self {
            source: Source::Toolkit,
            include,
            library,
            driver,
            runtime,
            imports,
            unpacked,
        }
    }
}

/// The toolkit home to build `product` against: `CUDA_HOME` or `CUDA_PATH`, for the host's own product only, and not
/// when the pinned archives are forced.
fn toolkit_home(product: &Product, force_download: bool) -> Option<PathBuf> {
    choose_toolkit(
        product,
        force_download,
        Platform::host(),
        ["CUDA_HOME", "CUDA_PATH"].iter().map(env::var_os),
    )
}

/// [`toolkit_home`] over explicit inputs: `variables` holds the values of `CUDA_HOME` and `CUDA_PATH`, in that order.
fn choose_toolkit(
    product: &Product,
    force_download: bool,
    host: Option<Platform>,
    variables: impl IntoIterator<Item = Option<OsString>>,
) -> Option<PathBuf> {
    if force_download || host != Some(product.host_os) {
        return None;
    }
    variables
        .into_iter()
        .flatten()
        .find(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn zig_program() -> OsString {
    env::var_os("ZIG").unwrap_or_else(|| "zig".into())
}

/// Fails unless running `program` with `probe` succeeds; `hint` says how to get the tool.
fn require_tool(program: &OsStr, probe: &str, hint: &str) -> Result<(), String> {
    match Command::new(program).arg(probe).output() {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "{} {probe} failed: {}; {hint}",
            program.to_string_lossy(),
            output.status
        )),
        Err(error) => Err(format!("cannot run {} ({error}); {hint}", program.to_string_lossy())),
    }
}

/// Where the downloaded archives are kept.
fn cache_directory(root: &Path, cuda_version: &str) -> PathBuf {
    match env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        Some(cache) => PathBuf::from(cache).join("euhedral-cuda").join(cuda_version),
        None => {
            let target = env::var_os("CARGO_TARGET_DIR").map_or_else(|| root.join("rust/target"), PathBuf::from);
            target.join("xtask/cuda-cache").join(cuda_version)
        }
    }
}

/// The value of the first `#define <macro> <digits>` of the header `include/<header>`.
fn header_version(include: &Path, header: &str, name: &str) -> Result<u32, String> {
    let path = include.join(header);
    let text = fs::read_to_string(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    text.lines()
        .filter_map(|line| line.trim_start().strip_prefix('#'))
        .filter_map(|line| line.trim_start().strip_prefix("define"))
        .filter(|rest| rest.starts_with(char::is_whitespace))
        .filter_map(|rest| {
            let mut words = rest.split_whitespace();
            let (macro_name, value) = (words.next()?, words.next()?);
            let digits: String = value.chars().take_while(char::is_ascii_digit).collect();
            (macro_name == name).then(|| digits.parse().ok()).flatten()
        })
        .next()
        .ok_or_else(|| format!("{} defines no numeric {name}", path.display()))
}

/// Fails unless the headers and the libraries are one CUDA 13.1 or later toolkit's: both header versions agree and are
/// at least 13.1, the headers sit beside the libraries' directory, and `nvrtc.h` is there.
fn check_coherent(include: &Path, library: &Path, platform: Platform) -> Result<(), String> {
    let runtime = header_version(include, "cuda_runtime_api.h", "CUDART_VERSION")?;
    let driver = header_version(include, "cuda.h", "CUDA_VERSION")?;
    if runtime < MINIMUM_CUDA || driver < MINIMUM_CUDA {
        return Err(format!(
            "CUDA headers in {} are older than {MINIMUM_CUDA}: CUDART_VERSION {runtime} (cuda_runtime_api.h), CUDA_VERSION {driver} (cuda.h)",
            include.display()
        ));
    }
    if runtime / 10 != driver / 10 {
        return Err(format!(
            "CUDA headers in {} disagree: CUDART_VERSION {runtime} (cuda_runtime_api.h) against CUDA_VERSION {driver} (cuda.h)",
            include.display()
        ));
    }
    let canonical = |path: &Path| {
        path.canonicalize()
            .map_err(|error| format!("cannot resolve {}: {error}", path.display()))
    };
    let library = canonical(library)?;
    let library_root =
        if platform == Platform::Windows && library.file_name().is_some_and(|name| name.eq_ignore_ascii_case("x64")) {
            library.parent().and_then(Path::parent)
        } else {
            library.parent()
        };
    if canonical(include)?.parent() != library_root {
        return Err(format!(
            "CUDA headers {} and libraries {} are not from one toolkit root",
            include.display(),
            library.display()
        ));
    }
    let nvrtc = include.join("nvrtc.h");
    if nvrtc.is_file() {
        Ok(())
    } else {
        Err(format!("missing {}", nvrtc.display()))
    }
}

/// Fails unless every file the build links is present and the headers are coherent with the libraries.
fn check_inputs(inputs: &Inputs, platform: Platform) -> Result<(), String> {
    check_coherent(&inputs.include, &inputs.library, platform)?;
    let required: Vec<PathBuf> = match platform {
        Platform::Windows => vec![
            inputs.library.join("cudart.lib"),
            inputs.library.join("nvrtc.lib"),
            inputs.driver.join("nvcuda.lib"),
            inputs.imports.join("cudart.lib"),
        ],
        Platform::Linux => vec![
            inputs.library.join("libcudart.so"),
            inputs.library.join("libnvrtc.so"),
            inputs.driver.join("libcuda.so"),
        ],
    };
    match required.iter().find(|path| !path.is_file()) {
        None => Ok(()),
        Some(missing) => {
            let hint = match inputs.source {
                Source::Toolkit => {
                    "the toolkit named by CUDA_HOME or CUDA_PATH lacks it; unset it or pass --force-download"
                }
                Source::Download => "the downloaded redistributables lack it",
            };
            Err(format!("CUDA 13.1+ input {} is missing: {hint}", missing.display()))
        }
    }
}

/// Downloads the pinned archives of `product` and unpacks them into the layout of [`redist`].
fn prepare_download(
    manifest: &Manifest,
    product: &Product,
    inputs: &Inputs,
    cache: &Path,
    download: Download,
) -> Result<(), String> {
    let components = manifest
        .redistributables
        .get(&product.cuda_platform)
        .ok_or_else(|| format!("{PRODUCTS} lists no redistributables for {}", product.cuda_platform))?;
    if inputs.unpacked.exists() {
        fs::remove_dir_all(&inputs.unpacked)
            .map_err(|error| format!("cannot clear {}: {error}", inputs.unpacked.display()))?;
    }
    // Later archives overwrite earlier ones, so the order is part of the layout.
    for name in COMPONENTS {
        let component = components
            .get(name)
            .ok_or_else(|| format!("{PRODUCTS} lists no {name} for {}", product.cuda_platform))?;
        println!("native: {} {name}", product.id);
        let archive = redist::fetch(cache, &component.path, &component.sha256, download)?;
        redist::unpack(&archive, &inputs.unpacked, product.host_os, &manifest.cuda_version)?;
    }
    if product.host_os == Platform::Linux {
        redist::link_linux_runtime(&inputs.unpacked, &manifest.cuda_version)?;
    }
    Ok(())
}

/// Runs `zig <arguments>` in `directory`.
fn run_zig(zig: &OsStr, directory: &Path, arguments: &[OsString]) -> Result<(), String> {
    let status = Command::new(zig)
        .current_dir(directory)
        .args(arguments)
        .status()
        .map_err(|error| {
            format!(
                "cannot run {} ({error}); install Zig with `mise install` or set ZIG",
                zig.to_string_lossy()
            )
        })?;
    if status.success() {
        Ok(())
    } else {
        let command = arguments
            .first()
            .map(|argument| argument.to_string_lossy().into_owned())
            .unwrap_or_default();
        Err(format!("zig {command} failed: {status}"))
    }
}

/// Generates the Windows import libraries of the driver and the runtime with `zig dlltool`.
fn generate_import_libraries(root: &Path, zig: &OsStr, imports: &Path) -> Result<(), String> {
    fs::create_dir_all(imports).map_err(|error| format!("cannot create {}: {error}", imports.display()))?;
    for stem in ["nvcuda", "cudart"] {
        let definition = root.join(format!("native/{stem}.def"));
        let arguments: Vec<OsString> = vec![
            "dlltool".into(),
            "-m".into(),
            "i386:x86-64".into(),
            "-d".into(),
            definition.into(),
            "-l".into(),
            imports.join(format!("{stem}.lib")).into(),
        ];
        run_zig(zig, root, &arguments).map_err(|error| format!("generating the {stem} import library: {error}"))?;
    }
    Ok(())
}

/// Builds `product` into `build/native/<id>` from `inputs`, which `check_inputs` has accepted.
fn build_product(
    root: &Path,
    manifest: &Manifest,
    product: &Product,
    inputs: &Inputs,
    zig: &OsStr,
) -> Result<(), String> {
    let prefix = root.join("build/native").join(&product.id);
    // Zig installs but never removes: clear the trees it owns so that a deleted source leaves no stale copy.
    for stale in ["share/euhedral_cuda", "cuda"] {
        let path = prefix.join(stale);
        if path.exists() {
            fs::remove_dir_all(&path).map_err(|error| format!("cannot clear {}: {error}", path.display()))?;
        }
    }
    let arguments: Vec<OsString> = vec![
        "build".into(),
        format!("-Dproduct-target={}", product.zig_target).into(),
        "-Doptimize=ReleaseSafe".into(),
        format!("-Dcuda-include-dir={}", inputs.include.display()).into(),
        format!("-Dcuda-lib-dir={}", inputs.library.display()).into(),
        format!("-Dcuda-driver-lib-dir={}", inputs.driver.display()).into(),
        "--prefix".into(),
        prefix.clone().into(),
    ];
    println!("native: {} zig build", product.id);
    run_zig(zig, &root.join("native"), &arguments)?;
    // The product directory is the layout the server bundle carries: the CUDA runtime and NVRTC beside the library,
    // and the headers NVRTC compiles the kernel sources against.
    let lib = prefix.join("lib");
    for name in redist::runtime_files(product.host_os, &manifest.cuda_version) {
        let from = inputs.runtime.join(&name);
        fs::copy(&from, lib.join(&name))
            .map_err(|error| format!("cannot copy {} to {}: {error}", from.display(), lib.display()))?;
    }
    copy_files(&inputs.include, &prefix.join("cuda/include"))?;
    fs::write(prefix.join(HASHES), file_hashes(&prefix)?).map_err(|error| format!("cannot write {HASHES}: {error}"))
}

/// Copies the regular files below `from` to the same relative paths below `to`.
fn copy_files(from: &Path, to: &Path) -> Result<(), String> {
    for relative in list_files(from)? {
        let target = to.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        let source = from.join(&relative);
        fs::copy(&source, &target)
            .map_err(|error| format!("cannot copy {} to {}: {error}", source.display(), target.display()))?;
    }
    Ok(())
}

/// The relative paths (with `/` separators) of the regular files below `root`, sorted.
fn list_files(root: &Path) -> Result<Vec<String>, String> {
    fn walk(directory: &Path, prefix: &str, found: &mut Vec<String>) -> Result<(), String> {
        let entries =
            fs::read_dir(directory).map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
            let path = entry.path();
            let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
            if path.is_dir() {
                walk(&path, &format!("{name}/"), found)?;
            } else if path.is_file() {
                found.push(name);
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(root, "", &mut found)?;
    found.sort();
    Ok(found)
}

/// The text of `files.sha256` for the tree at `root`: `<sha256>  <relative path>` per file, sorted by path, as
/// `sha256sum` writes them, without the manifest itself.
fn file_hashes(root: &Path) -> Result<String, String> {
    let mut text = String::new();
    for relative in list_files(root)?.into_iter().filter(|relative| relative != HASHES) {
        text.push_str(&format!("{}  {relative}\n", redist::sha256_hex(&root.join(&relative))?));
    }
    Ok(text)
}

/// `cargo xtask native [--product <id>] [--force-download]`.
pub fn run(arguments: &[&str]) -> Result<(), String> {
    let options = parse_options(arguments, true)?;
    let root = repository_root();
    let manifest = Manifest::load(&root)?;
    let products = manifest.select(options.product.as_deref())?;
    // Plan every product and check the tools before the first download: a missing tool refuses at once.
    let plans: Vec<Inputs> = products
        .iter()
        .map(|product| Inputs::plan(&root, product, toolkit_home(product, options.force_download).as_deref()))
        .collect();
    let zig = zig_program();
    require_tool(&zig, "version", "install Zig with `mise install` or set ZIG")?;
    if plans.iter().any(|inputs| inputs.source == Source::Download) {
        require_tool(
            OsStr::new("curl"),
            "--version",
            "it downloads the CUDA redistributables",
        )?;
    }
    let cache = cache_directory(&root, &manifest.cuda_version);
    for (product, inputs) in products.iter().zip(&plans) {
        if inputs.source == Source::Download {
            prepare_download(&manifest, product, inputs, &cache, &redist::download_with_curl)?;
        }
        if product.host_os == Platform::Windows {
            generate_import_libraries(&root, &zig, &inputs.imports)?;
        }
        check_inputs(inputs, product.host_os)?;
        build_product(&root, &manifest, product, inputs, &zig)?;
        println!(
            "native: built {}",
            root.join("build/native").join(&product.id).display()
        );
    }
    Ok(())
}

/// `cargo xtask native-verify [--product <id>]`.
pub fn run_verify(arguments: &[&str]) -> Result<(), String> {
    let options = parse_options(arguments, false)?;
    let root = repository_root();
    let manifest = Manifest::load(&root)?;
    for product in manifest.select(options.product.as_deref())? {
        verify_product(&root, &manifest, product)?;
        println!("native: verified {}", product.id);
    }
    Ok(())
}

fn kernel_sources(root: &Path) -> Result<BTreeSet<String>, String> {
    Ok(list_files(root)?
        .into_iter()
        .filter(|name| name.ends_with(".cu") || name.ends_with(".cuh"))
        .collect())
}

/// Checks the tree of `product` under `build/native/<id>`.
fn verify_product(root: &Path, manifest: &Manifest, product: &Product) -> Result<(), String> {
    let id = &product.id;
    let prefix = root.join("build/native").join(id);
    let binary = prefix.join("lib").join(&product.filename);
    if !binary.is_file() {
        return Err(format!("missing native product {}", binary.display()));
    }
    let sources = kernel_sources(&root.join("native/src"))?;
    let installed = kernel_sources(&prefix.join("share/euhedral_cuda"))?;
    if sources != installed {
        return Err(format!(
            "NVRTC sources of {id} differ from native/src: missing {:?}, extra {:?}",
            sources.difference(&installed).collect::<Vec<_>>(),
            installed.difference(&sources).collect::<Vec<_>>()
        ));
    }
    let windows = product.host_os == Platform::Windows;
    let mut data = Vec::new();
    File::open(&binary)
        .and_then(|mut file| file.read_to_end(&mut data))
        .map_err(|error| format!("cannot read {}: {error}", binary.display()))?;
    let signature: &[u8] = if windows { b"MZ" } else { &[0x7f, b'E'] };
    if !data.starts_with(signature) {
        return Err(format!(
            "wrong binary format for {id}: {} is not a {:?} binary",
            binary.display(),
            product.host_os
        ));
    }
    let names = binary_imports(&data, product.host_os).map_err(|error| format!("{}: {error}", binary.display()))?;
    let wanted: &[&str] = if windows {
        &["cudart64_13.dll", "nvrtc64_130_0.dll", "nvcuda.dll"]
    } else {
        &["libcudart.so.13", "libnvrtc.so.13", "libcuda.so.1"]
    };
    let foreign = if windows { ".so" } else { ".dll" };
    if let Some(missing) = wanted.iter().find(|name| !names.contains(**name)) {
        return Err(format!(
            "CUDA imports do not match target {id}: {} does not import {missing} (imports {names:?})",
            binary.display()
        ));
    }
    if let Some(wrong) = names.iter().find(|name| name.ends_with(foreign)) {
        return Err(format!(
            "CUDA imports do not match target {id}: {} imports {wrong}, a {foreign} file",
            binary.display()
        ));
    }
    let bundled = redist::runtime_files(product.host_os, &manifest.cuda_version)
        .into_iter()
        .map(|name| format!("lib/{name}"))
        .chain([
            "cuda/include/cuda_fp4.h".to_string(),
            "cuda/include/nvrtc.h".to_string(),
        ]);
    for name in bundled {
        if !prefix.join(&name).is_file() {
            return Err(format!("missing {name} in {id}"));
        }
    }
    let hashes = prefix.join(HASHES);
    let text = fs::read_to_string(&hashes).map_err(|error| format!("cannot read {}: {error}", hashes.display()))?;
    let listed: BTreeSet<&str> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.split_once("  ").map_or(line, |(_, path)| path))
        .collect();
    let files = list_files(&prefix)?;
    let present: BTreeSet<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|name| *name != HASHES)
        .collect();
    if listed != present {
        return Err(format!(
            "{HASHES} does not list the files of {id}: unlisted {:?}, listed but absent {:?}",
            present.difference(&listed).collect::<Vec<_>>(),
            listed.difference(&present).collect::<Vec<_>>()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::io::{Cursor, Write};

    use tempfile::tempdir;
    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;

    use super::*;
    use crate::imports::tests::{elf, pe};

    const VERSION: &str = "13.1.80";

    fn layout(name: &str, platform: Platform) -> Result<Option<String>, String> {
        redist::layout_path(name, platform, VERSION)
    }

    #[test]
    fn linux_archive_names_map_to_the_input_layout() {
        let cases = [
            (
                "cuda_cudart-linux-x86_64-13.1.80-archive/include/cuda_runtime_api.h",
                Some("include/cuda_runtime_api.h"),
            ),
            (
                "cuda_cccl-linux-x86_64-13.1.78-archive/include/cub/cub.cuh",
                Some("include/cub/cub.cuh"),
            ),
            ("a/lib/stubs/libcuda.so", Some("lib/libcuda.so")),
            ("a/lib/libcudart.so.13.1.80", Some("lib/libcudart.so")),
            ("a/lib/libnvrtc.so.13.1.80", Some("lib/libnvrtc.so")),
            (
                "a/lib/libnvrtc-builtins.so.13.1.80",
                Some("runtime/libnvrtc-builtins.so.13.1.80"),
            ),
            ("a/lib/libcudart_static.a", None),
            ("a/lib/libcudart.so.13", None),
            ("a/bin/x64/cudart64_13.dll", None),
            ("a/LICENSE", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                layout(name, Platform::Linux),
                Ok(expected.map(str::to_string)),
                "{name}"
            );
        }
    }

    #[test]
    fn windows_archive_names_map_to_the_input_layout() {
        let cases = [
            ("a/include/nvrtc.h", Some("include/nvrtc.h")),
            ("a/lib/x64/cuda.lib", Some("lib/cuda.lib")),
            ("a/lib/x64/cudart.lib", Some("lib/cudart.lib")),
            ("a/lib/x64/nvrtc.lib", Some("lib/nvrtc.lib")),
            ("a/lib/x64/nvrtc_static.lib", None),
            ("a/bin/x64/cudart64_13.dll", Some("runtime/cudart64_13.dll")),
            ("a/bin/x64/nvrtc64_130_0.dll", Some("runtime/nvrtc64_130_0.dll")),
            (
                "a/bin/x64/nvrtc-builtins64_131.dll",
                Some("runtime/nvrtc-builtins64_131.dll"),
            ),
            ("a/bin/x64/other.dll", None),
            ("a/lib/stubs/libcuda.so", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                layout(name, Platform::Windows),
                Ok(expected.map(str::to_string)),
                "{name}"
            );
        }
    }

    #[test]
    fn an_entry_that_escapes_the_destination_is_refused_even_where_the_layout_would_skip_it() {
        for name in [
            "a/include/../../evil.h",
            "../a/lib/libcuda.so",
            "/etc/passwd",
            "a\\..\\b",
            "C:/x/include/a.h",
            "a/unmapped/../x",
        ] {
            let error = layout(name, Platform::Linux).expect_err(name);
            assert!(error.contains(name) || error.contains(&format!("{name:?}")), "{error}");
        }
    }

    fn tar_with(entries: &[(&str, &[u8], tar::EntryType)]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data, kind) in entries {
            let mut header = tar::Header::new_gnu();
            // `set_path` refuses `..`: write the name as an archive of another origin would carry it.
            header.as_old_mut().name[..name.len()].copy_from_slice(name.as_bytes());
            header.set_entry_type(*kind);
            let linked = matches!(kind, tar::EntryType::Symlink | tar::EntryType::Link);
            let empty = linked || matches!(kind, tar::EntryType::Char | tar::EntryType::Block);
            let data = if empty { &b""[..] } else { data };
            header.set_size(data.len() as u64);
            if linked {
                header.set_link_name("target").expect("link name");
            }
            header.set_cksum();
            builder.append(&header, data).expect("entry");
        }
        builder.into_inner().expect("tar")
    }

    #[test]
    fn a_tar_stream_installs_only_the_regular_files_the_layout_names() {
        let tar = tar_with(&[
            ("a/include/h.h", b"header", tar::EntryType::Regular),
            ("a/lib/libcudart.so.13.1.80", b"cudart", tar::EntryType::Regular),
            ("a/include/sub/", b"", tar::EntryType::Directory),
            ("a/lib/libcudart.so.13", b"link", tar::EntryType::Symlink),
            ("a/lib/libcudart_static.a", b"static", tar::EntryType::Regular),
        ]);
        let destination = tempdir().expect("directory");
        redist::unpack_tar(Cursor::new(tar), destination.path(), Platform::Linux, VERSION).expect("unpack");
        let mut files = list_files(destination.path()).expect("list");
        files.sort();
        assert_eq!(files, ["include/h.h", "lib/libcudart.so"]);
        assert_eq!(
            fs::read(destination.path().join("lib/libcudart.so")).expect("read"),
            b"cudart"
        );
    }

    #[test]
    fn a_tar_stream_with_an_escaping_entry_is_refused_and_writes_nothing_outside() {
        let tar = tar_with(&[("a/include/../../escaped.h", b"x", tar::EntryType::Regular)]);
        let parent = tempdir().expect("directory");
        let destination = parent.path().join("unpacked");
        fs::create_dir(&destination).expect("destination");
        let error = redist::unpack_tar(Cursor::new(tar), &destination, Platform::Linux, VERSION).expect_err("refused");
        assert!(error.contains("a/include/../../escaped.h"), "{error}");
        assert!(!parent.path().join("escaped.h").exists());
    }

    fn zip_with(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, data) in entries {
            writer.start_file(*name, options).expect("start");
            writer.write_all(data).expect("write");
        }
        writer.finish().expect("finish").into_inner()
    }

    #[test]
    fn a_zip_archive_installs_the_windows_layout() {
        let zip = zip_with(&[
            ("a/include/nvrtc.h", b"nvrtc"),
            ("a/lib/x64/cuda.lib", b"cuda"),
            ("a/bin/x64/cudart64_13.dll", b"dll"),
            ("a/bin/x64/ignored.dll", b"no"),
        ]);
        let destination = tempdir().expect("directory");
        redist::unpack_zip(Cursor::new(zip), destination.path(), Platform::Windows, VERSION).expect("unpack");
        assert_eq!(
            list_files(destination.path()).expect("list"),
            ["include/nvrtc.h", "lib/cuda.lib", "runtime/cudart64_13.dll"]
        );
    }

    #[test]
    fn a_zip_archive_with_an_escaping_entry_is_refused() {
        let zip = zip_with(&[("a/lib/x64/../../../cuda.lib", b"x")]);
        let destination = tempdir().expect("directory");
        let error =
            redist::unpack_zip(Cursor::new(zip), destination.path(), Platform::Windows, VERSION).expect_err("refused");
        assert!(error.contains("../"), "{error}");
    }

    fn sha_of(bytes: &[u8]) -> String {
        let file = tempdir().expect("directory");
        fs::write(file.path().join("f"), bytes).expect("write");
        redist::sha256_hex(&file.path().join("f")).expect("sha")
    }

    #[test]
    fn a_download_with_the_wrong_checksum_is_refused_and_not_cached() {
        let cache = tempdir().expect("directory");
        let expected = sha_of(b"right");
        let download =
            |_: &str, destination: &Path| fs::write(destination, b"wrong").map_err(|error| error.to_string());
        let error =
            redist::fetch(cache.path(), "cuda_x/linux-x86_64/x.tar.xz", &expected, &download).expect_err("refused");
        assert!(
            error.contains("cuda_x/linux-x86_64/x.tar.xz")
                && error.contains(&expected)
                && error.contains(&sha_of(b"wrong")),
            "{error}"
        );
        assert_eq!(
            fs::read_dir(cache.path()).expect("read").count(),
            0,
            "nothing is left in the cache"
        );
    }

    #[test]
    fn a_download_with_the_right_checksum_is_cached_and_not_fetched_again() {
        let cache = tempdir().expect("directory");
        let expected = sha_of(b"right");
        let calls = Cell::new(0);
        let download = |url: &str, destination: &Path| {
            calls.set(calls.get() + 1);
            assert!(url.ends_with("/cuda_x/linux-x86_64/x.tar.xz"), "{url}");
            fs::write(destination, b"right").map_err(|error| error.to_string())
        };
        let first = redist::fetch(cache.path(), "cuda_x/linux-x86_64/x.tar.xz", &expected, &download).expect("fetch");
        let second = redist::fetch(cache.path(), "cuda_x/linux-x86_64/x.tar.xz", &expected, &download).expect("cached");
        assert_eq!((first, calls.get()), (second, 1));
    }

    #[test]
    fn a_cached_file_with_the_wrong_checksum_is_replaced_by_a_download() {
        let cache = tempdir().expect("directory");
        fs::write(cache.path().join("x.zip"), b"stale").expect("stale");
        let download =
            |_: &str, destination: &Path| fs::write(destination, b"fresh").map_err(|error| error.to_string());
        let archive = redist::fetch(cache.path(), "p/x.zip", &sha_of(b"fresh"), &download).expect("fetch");
        assert_eq!(fs::read(archive).expect("read"), b"fresh");
    }

    fn headers(include: &Path, runtime: &str, driver: &str) {
        fs::create_dir_all(include).expect("include");
        fs::write(
            include.join("cuda_runtime_api.h"),
            format!("#ifndef X\n#define CUDART_VERSION  {runtime}\n"),
        )
        .expect("runtime header");
        fs::write(include.join("cuda.h"), format!("#  define CUDA_VERSION {driver}\n")).expect("driver header");
        fs::write(include.join("nvrtc.h"), "").expect("nvrtc");
    }

    fn toolkit(runtime: &str, driver: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let root = tempdir().expect("directory");
        let (include, library) = (root.path().join("include"), root.path().join("lib"));
        headers(&include, runtime, driver);
        fs::create_dir_all(&library).expect("library");
        (root, include, library)
    }

    #[test]
    fn headers_of_one_version_beside_the_libraries_are_coherent() {
        let (_root, include, library) = toolkit("13010", "13010");
        assert_eq!(check_coherent(&include, &library, Platform::Linux), Ok(()));
    }

    #[test]
    fn headers_that_disagree_are_refused_with_both_versions() {
        let (_root, include, library) = toolkit("13010", "13020");
        let error = check_coherent(&include, &library, Platform::Linux).expect_err("refused");
        assert!(
            error.contains("13010") && error.contains("13020") && error.contains("disagree"),
            "{error}"
        );
    }

    #[test]
    fn headers_older_than_the_minimum_are_refused() {
        let (_root, include, library) = toolkit("12090", "12090");
        let error = check_coherent(&include, &library, Platform::Linux).expect_err("refused");
        assert!(error.contains("12090") && error.contains("13010"), "{error}");
    }

    #[test]
    fn a_header_without_a_version_define_is_refused_with_its_path() {
        let (_root, include, library) = toolkit("13010", "13010");
        fs::write(include.join("cuda.h"), "// no version\n").expect("rewrite");
        let error = check_coherent(&include, &library, Platform::Linux).expect_err("refused");
        assert!(error.contains("cuda.h") && error.contains("CUDA_VERSION"), "{error}");
    }

    #[test]
    fn headers_and_libraries_of_different_roots_are_refused() {
        let (_root, include, _library) = toolkit("13010", "13010");
        let other = tempdir().expect("directory");
        let error = check_coherent(&include, other.path(), Platform::Linux).expect_err("refused");
        assert!(error.contains("not from one toolkit root"), "{error}");
    }

    #[test]
    fn a_windows_toolkit_keeps_its_libraries_in_lib_x64_beside_include() {
        let root = tempdir().expect("directory");
        headers(&root.path().join("include"), "13010", "13010");
        let library = root.path().join("lib/x64");
        fs::create_dir_all(&library).expect("library");
        assert_eq!(
            check_coherent(&root.path().join("include"), &library, Platform::Windows),
            Ok(())
        );
    }

    #[test]
    fn a_missing_zig_is_refused_with_its_name() {
        let error =
            require_tool(OsStr::new("/nonexistent/zig-for-test"), "version", "install Zig").expect_err("refused");
        assert!(
            error.contains("/nonexistent/zig-for-test") && error.contains("install Zig"),
            "{error}"
        );
        let error = run_zig(
            OsStr::new("/nonexistent/zig-for-test"),
            Path::new("."),
            &["build".into()],
        )
        .expect_err("refused");
        assert!(error.contains("/nonexistent/zig-for-test"), "{error}");
    }

    #[test]
    fn file_hashes_are_sorted_by_path_and_leave_out_the_manifest() {
        let root = tempdir().expect("directory");
        for (name, data) in [("lib/b", "2"), ("a", "1"), ("lib/a", "3"), (HASHES, "old")] {
            let path = root.path().join(name);
            fs::create_dir_all(path.parent().expect("parent")).expect("directory");
            fs::write(path, data).expect("file");
        }
        let text = file_hashes(root.path()).expect("hashes");
        let paths: Vec<&str> = text
            .lines()
            .map(|line| line.split_once("  ").expect("two spaces").1)
            .collect();
        assert_eq!(paths, ["a", "lib/a", "lib/b"]);
        assert!(text.starts_with(&format!("{}  a\n", sha_of(b"1"))), "{text}");
    }

    #[test]
    fn linux_runtime_names_are_copied_from_the_unpacked_libraries() {
        let root = tempdir().expect("directory");
        for (name, data) in [
            ("lib/libcudart.so", "cudart"),
            ("lib/libnvrtc.so", "nvrtc"),
            ("runtime/libnvrtc-builtins.so.13.1.80", "builtins"),
        ] {
            let path = root.path().join(name);
            fs::create_dir_all(path.parent().expect("parent")).expect("directory");
            fs::write(path, data).expect("file");
        }
        redist::link_linux_runtime(root.path(), VERSION).expect("link");
        for (name, data) in [
            ("libcudart.so.13", "cudart"),
            ("libnvrtc.so.13", "nvrtc"),
            ("libnvrtc-builtins.so.13.1", "builtins"),
        ] {
            assert_eq!(
                fs::read_to_string(root.path().join("runtime").join(name)).expect(name),
                data
            );
        }
        assert_eq!(
            redist::runtime_files(Platform::Linux, VERSION),
            ["libcudart.so.13", "libnvrtc.so.13", "libnvrtc-builtins.so.13.1"]
        );
    }

    #[test]
    fn options_name_a_product_and_refuse_unknown_arguments() {
        assert_eq!(
            parse_options(&["--product", "linux-x64", "--force-download"], true),
            Ok(Options {
                product: Some("linux-x64".to_string()),
                force_download: true
            })
        );
        assert!(parse_options(&["--force-download"], false).is_err());
        assert!(parse_options(&["--product"], true).is_err());
        assert!(parse_options(&["--bogus"], true).is_err());
    }

    #[test]
    fn an_unknown_product_is_refused_with_the_known_ones() {
        let manifest: Manifest = serde_json::from_str(
            r#"{"cudaVersion":"13.1.80","products":[{"id":"linux-x64","zigTarget":"z","filename":"f","cudaPlatform":"p","hostOs":"linux"}],"redistributables":{}}"#,
        )
        .expect("manifest");
        let error = manifest.select(Some("macos")).expect_err("refused");
        assert!(error.contains("macos") && error.contains("linux-x64"), "{error}");
        assert_eq!(manifest.select(None).expect("all").len(), 1);
    }

    fn product_tree(root: &Path, sources: &[&str], installed: &[&str]) -> (Manifest, Product) {
        for (base, names) in [
            ("native/src", sources),
            ("build/native/linux-x64/share/euhedral_cuda", installed),
        ] {
            fs::create_dir_all(root.join(base)).expect("directory");
            for name in names {
                let path = root.join(base).join(name);
                fs::create_dir_all(path.parent().expect("parent")).expect("directory");
                fs::write(path, "").expect("file");
            }
        }
        let library = root.join("build/native/linux-x64/lib/libeuhedral_cuda.so");
        fs::create_dir_all(library.parent().expect("parent")).expect("directory");
        fs::write(library, b"\x7fELF").expect("library");
        let manifest = Manifest {
            cuda_version: VERSION.to_string(),
            products: Vec::new(),
            redistributables: BTreeMap::new(),
        };
        let product = Product {
            id: "linux-x64".to_string(),
            zig_target: "x86_64-linux-gnu".to_string(),
            filename: "libeuhedral_cuda.so".to_string(),
            cuda_platform: "linux-x86_64".to_string(),
            host_os: Platform::Linux,
        };
        (manifest, product)
    }

    #[test]
    fn a_product_missing_a_kernel_source_is_refused_naming_it() {
        let root = tempdir().expect("directory");
        let (manifest, product) = product_tree(root.path(), &["k/a.cu", "k/b.cuh"], &["k/a.cu"]);
        let error = verify_product(root.path(), &manifest, &product).expect_err("refused");
        assert!(error.contains("k/b.cuh"), "{error}");
    }

    #[test]
    fn a_product_without_its_library_is_refused() {
        let root = tempdir().expect("directory");
        let (manifest, product) = product_tree(root.path(), &[], &[]);
        fs::remove_file(root.path().join("build/native/linux-x64/lib/libeuhedral_cuda.so")).expect("remove");
        let error = verify_product(root.path(), &manifest, &product).expect_err("refused");
        assert!(error.contains("libeuhedral_cuda.so"), "{error}");
    }

    #[test]
    fn a_library_that_is_not_an_elf_is_refused_for_a_linux_product() {
        let root = tempdir().expect("directory");
        let (manifest, product) = product_tree(root.path(), &[], &[]);
        fs::write(
            root.path().join("build/native/linux-x64/lib/libeuhedral_cuda.so"),
            b"MZ",
        )
        .expect("rewrite");
        let error = verify_product(root.path(), &manifest, &product).expect_err("refused");
        assert!(error.contains("wrong binary format"), "{error}");
    }

    #[test]
    fn a_tar_entry_that_is_not_a_regular_file_at_a_used_name_is_refused_naming_it() {
        for (name, kind) in [
            ("a/lib/libnvrtc.so.13.1.80", tar::EntryType::Symlink),
            ("a/lib/libnvrtc.so.13.1.80", tar::EntryType::Link),
            ("a/lib/stubs/libcuda.so", tar::EntryType::Char),
            ("a/lib/stubs/libcuda.so", tar::EntryType::Block),
            ("a/include/h.h", tar::EntryType::Symlink),
        ] {
            let tar = tar_with(&[(name, b"", kind)]);
            let destination = tempdir().expect("directory");
            let error =
                redist::unpack_tar(Cursor::new(tar), destination.path(), Platform::Linux, VERSION).expect_err(name);
            assert!(
                error.contains(name) && error.contains("not a regular file"),
                "{kind:?}: {error}"
            );
            assert_eq!(list_files(destination.path()).expect("list"), Vec::<String>::new());
        }
    }

    fn zip_with_mode(name: &str, data: &[u8], mode: u32) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().unix_permissions(mode);
        writer.start_file(name, options).expect("start");
        writer.write_all(data).expect("write");
        writer.finish().expect("finish").into_inner()
    }

    #[test]
    fn a_zip_symlink_at_a_used_name_is_refused_naming_it_and_writes_no_file() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        writer
            .add_symlink("a/lib/x64/cuda.lib", "other.lib", SimpleFileOptions::default())
            .expect("symlink");
        let zip = writer.finish().expect("finish").into_inner();
        let destination = tempdir().expect("directory");
        let error =
            redist::unpack_zip(Cursor::new(zip), destination.path(), Platform::Windows, VERSION).expect_err("refused");
        assert!(
            error.contains("a/lib/x64/cuda.lib") && error.contains("not a regular file"),
            "{error}"
        );
        assert_eq!(list_files(destination.path()).expect("list"), Vec::<String>::new());
    }

    #[test]
    fn a_zip_file_with_permission_bits_only_is_a_regular_file() {
        let zip = zip_with_mode("a/lib/x64/cuda.lib", b"cuda", 0o644);
        let destination = tempdir().expect("directory");
        redist::unpack_zip(Cursor::new(zip), destination.path(), Platform::Windows, VERSION).expect("unpack");
        assert_eq!(list_files(destination.path()).expect("list"), ["lib/cuda.lib"]);
    }

    fn product(platform: Platform) -> Product {
        Product {
            id: "p".to_string(),
            zig_target: "z".to_string(),
            filename: "f".to_string(),
            cuda_platform: "linux-x86_64".to_string(),
            host_os: platform,
        }
    }

    fn touch(root: &Path, names: &[&str]) {
        for name in names {
            let path = root.join(name);
            fs::create_dir_all(path.parent().expect("parent")).expect("directory");
            fs::write(path, "").expect("file");
        }
    }

    /// A Linux toolkit home with headers of the given versions and the libraries the build links.
    fn linux_toolkit_home(runtime: &str, driver: &str) -> tempfile::TempDir {
        let home = tempdir().expect("directory");
        headers(&home.path().join("targets/x86_64-linux/include"), runtime, driver);
        touch(
            home.path(),
            &[
                "targets/x86_64-linux/lib/libcudart.so",
                "targets/x86_64-linux/lib/libnvrtc.so",
                "targets/x86_64-linux/lib/stubs/libcuda.so",
            ],
        );
        home
    }

    #[test]
    fn a_coherent_local_toolkit_is_the_source_and_needs_no_download() {
        let (root, home) = (tempdir().expect("root"), linux_toolkit_home("13010", "13010"));
        let inputs = Inputs::plan(root.path(), &product(Platform::Linux), Some(home.path()));
        assert_eq!(inputs.source, Source::Toolkit);
        assert_eq!(check_inputs(&inputs, Platform::Linux), Ok(()));
    }

    #[test]
    fn an_incoherent_local_toolkit_is_refused_naming_both_header_versions() {
        let (root, home) = (tempdir().expect("root"), linux_toolkit_home("13010", "13020"));
        let inputs = Inputs::plan(root.path(), &product(Platform::Linux), Some(home.path()));
        let error = check_inputs(&inputs, Platform::Linux).expect_err("refused");
        assert!(error.contains("13010") && error.contains("13020"), "{error}");
    }

    #[test]
    fn without_a_toolkit_the_source_is_the_download_under_the_product_directory() {
        let root = Path::new("/r");
        let inputs = Inputs::plan(root, &product(Platform::Linux), None);
        assert_eq!(inputs.source, Source::Download);
        assert_eq!(inputs.unpacked, root.join("build/cuda-dev/p"));
        assert_eq!(inputs.include, root.join("build/cuda-dev/p/include"));
        assert_eq!(inputs.library, root.join("build/cuda-dev/p/lib"));
        assert_eq!(inputs.runtime, root.join("build/cuda-dev/p/runtime"));
    }

    #[test]
    fn the_driver_directory_follows_the_platform_and_the_source() {
        let (root, home) = (Path::new("/r"), Path::new("/h"));
        let driver = |platform, toolkit| Inputs::plan(root, &product(platform), toolkit).driver;
        assert_eq!(driver(Platform::Linux, None), root.join("build/cuda-dev/p/lib"));
        assert_eq!(driver(Platform::Windows, None), root.join("build/cuda-import/p"));
        assert_eq!(
            driver(Platform::Linux, Some(home)),
            home.join("targets/x86_64-linux/lib/stubs")
        );
        assert_eq!(driver(Platform::Windows, Some(home)), root.join("build/cuda-import/p"));
    }

    #[test]
    fn the_toolkit_layout_follows_the_platform() {
        let (root, home) = (Path::new("/r"), Path::new("/h"));
        let windows = Inputs::plan(root, &product(Platform::Windows), Some(home));
        assert_eq!(
            (windows.include, windows.library, windows.runtime),
            (home.join("include"), home.join("lib/x64"), home.join("bin/x64"))
        );
        let linux = Inputs::plan(root, &product(Platform::Linux), Some(home));
        let tree = home.join("targets/x86_64-linux");
        assert_eq!(
            (linux.include, linux.library, linux.runtime),
            (tree.join("include"), tree.join("lib"), tree.join("lib"))
        );
    }

    #[test]
    fn the_toolkit_is_chosen_only_for_the_host_platform_and_never_when_forced() {
        let variables = || [Some(OsString::from("")), Some(OsString::from("/cuda"))];
        let linux = product(Platform::Linux);
        let choose = |force, host| choose_toolkit(&linux, force, host, variables());
        assert_eq!(choose(false, Some(Platform::Linux)), Some(PathBuf::from("/cuda")));
        assert_eq!(choose(true, Some(Platform::Linux)), None);
        assert_eq!(choose(false, Some(Platform::Windows)), None);
        assert_eq!(choose(false, None), None);
        assert_eq!(choose_toolkit(&linux, false, Some(Platform::Linux), [None, None]), None);
        assert_eq!(
            choose_toolkit(
                &linux,
                false,
                Some(Platform::Linux),
                [Some(OsString::from("/home")), Some(OsString::from("/path"))]
            ),
            Some(PathBuf::from("/home"))
        );
    }

    #[test]
    fn a_toolkit_lacking_a_library_is_refused_naming_it_and_the_variables() {
        let (root, home) = (tempdir().expect("root"), linux_toolkit_home("13010", "13010"));
        fs::remove_file(home.path().join("targets/x86_64-linux/lib/libnvrtc.so")).expect("remove");
        let inputs = Inputs::plan(root.path(), &product(Platform::Linux), Some(home.path()));
        let error = check_inputs(&inputs, Platform::Linux).expect_err("refused");
        assert!(error.contains("libnvrtc.so") && error.contains("CUDA_HOME"), "{error}");
    }

    #[test]
    fn a_download_lacking_a_library_is_refused_naming_it() {
        let root = tempdir().expect("root");
        let inputs = Inputs::plan(root.path(), &product(Platform::Linux), None);
        headers(&inputs.include, "13010", "13010");
        touch(&inputs.unpacked, &["lib/libcudart.so", "lib/libnvrtc.so"]);
        let error = check_inputs(&inputs, Platform::Linux).expect_err("refused");
        assert!(error.contains("libcuda.so") && error.contains("downloaded"), "{error}");
        touch(&inputs.unpacked, &["lib/libcuda.so"]);
        assert_eq!(check_inputs(&inputs, Platform::Linux), Ok(()));
    }

    #[test]
    fn a_windows_build_without_its_generated_import_library_is_refused() {
        let root = tempdir().expect("root");
        let inputs = Inputs::plan(root.path(), &product(Platform::Windows), None);
        headers(&inputs.include, "13010", "13010");
        touch(&inputs.unpacked, &["lib/cudart.lib", "lib/nvrtc.lib"]);
        touch(&inputs.imports, &["nvcuda.lib"]);
        let error = check_inputs(&inputs, Platform::Windows).expect_err("refused");
        assert!(error.contains("cudart.lib") && error.contains("cuda-import"), "{error}");
        touch(&inputs.imports, &["cudart.lib"]);
        assert_eq!(check_inputs(&inputs, Platform::Windows), Ok(()));
    }

    fn manifest_with(components: &[&str]) -> Manifest {
        let archives = components
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    Component {
                        path: format!("{name}/x.tar.xz"),
                        sha256: String::new(),
                    },
                )
            })
            .collect();
        Manifest {
            cuda_version: VERSION.to_string(),
            products: Vec::new(),
            redistributables: BTreeMap::from([("linux-x86_64".to_string(), archives)]),
        }
    }

    #[test]
    fn a_manifest_without_the_products_platform_is_refused_before_any_download() {
        let root = tempdir().expect("root");
        let mut item = product(Platform::Linux);
        item.cuda_platform = "windows-x86_64".to_string();
        let inputs = Inputs::plan(root.path(), &item, None);
        let download = |_: &str, _: &Path| -> Result<(), String> { panic!("downloaded") };
        let error =
            prepare_download(&manifest_with(&COMPONENTS), &item, &inputs, root.path(), &download).expect_err("refused");
        assert!(error.contains("windows-x86_64"), "{error}");
    }

    #[test]
    fn a_manifest_missing_a_component_is_refused_naming_it_before_any_download() {
        let root = tempdir().expect("root");
        let item = product(Platform::Linux);
        let inputs = Inputs::plan(root.path(), &item, None);
        let download = |_: &str, _: &Path| -> Result<(), String> { panic!("downloaded") };
        let manifest = manifest_with(&["nvrtc", "crt", "cccl"]);
        let error = prepare_download(&manifest, &item, &inputs, root.path(), &download).expect_err("refused");
        assert!(error.contains("cudart") && error.contains("linux-x86_64"), "{error}");
    }

    /// A built Linux product tree that `verify_product` accepts.
    fn built_tree(root: &Path) -> (Manifest, Product) {
        let (manifest, product) = product_tree(root, &["k/a.cu"], &["k/a.cu"]);
        let prefix = root.join("build/native/linux-x64");
        let wanted = ["libcudart.so.13", "libnvrtc.so.13", "libcuda.so.1"];
        fs::write(prefix.join("lib/libeuhedral_cuda.so"), elf(&wanted)).expect("library");
        let bundled: Vec<String> = redist::runtime_files(Platform::Linux, VERSION)
            .into_iter()
            .map(|name| format!("lib/{name}"))
            .chain([
                "cuda/include/cuda_fp4.h".to_string(),
                "cuda/include/nvrtc.h".to_string(),
            ])
            .collect();
        touch(&prefix, &bundled.iter().map(String::as_str).collect::<Vec<_>>());
        fs::write(prefix.join(HASHES), file_hashes(&prefix).expect("hashes")).expect("manifest");
        (manifest, product)
    }

    fn built_tree_error(change: impl FnOnce(&Path)) -> String {
        let root = tempdir().expect("root");
        let (manifest, product) = built_tree(root.path());
        change(&root.path().join("build/native/linux-x64"));
        verify_product(root.path(), &manifest, &product).expect_err("refused")
    }

    #[test]
    fn a_complete_product_tree_is_accepted() {
        let root = tempdir().expect("root");
        let (manifest, product) = built_tree(root.path());
        assert_eq!(verify_product(root.path(), &manifest, &product), Ok(()));
    }

    #[test]
    fn a_library_that_does_not_import_a_runtime_is_refused_naming_it() {
        let error = built_tree_error(|prefix| {
            fs::write(
                prefix.join("lib/libeuhedral_cuda.so"),
                elf(&["libcudart.so.13", "libcuda.so.1"]),
            )
            .expect("library");
        });
        assert!(error.contains("does not import libnvrtc.so.13"), "{error}");
    }

    #[test]
    fn a_linux_library_that_imports_a_dll_is_refused_naming_it() {
        let error = built_tree_error(|prefix| {
            let wanted = ["libcudart.so.13", "libnvrtc.so.13", "libcuda.so.1", "nvcuda.dll"];
            fs::write(prefix.join("lib/libeuhedral_cuda.so"), elf(&wanted)).expect("library");
        });
        assert!(error.contains("imports nvcuda.dll"), "{error}");
    }

    #[test]
    fn a_windows_library_that_imports_a_shared_object_is_refused_naming_it() {
        let root = tempdir().expect("root");
        let (manifest, mut item) = built_tree(root.path());
        item.host_os = Platform::Windows;
        let prefix = root.path().join("build/native/linux-x64");
        let wanted = ["cudart64_13.dll", "nvrtc64_130_0.dll", "nvcuda.dll", "libcuda.so"];
        fs::write(prefix.join("lib/libeuhedral_cuda.so"), pe(&wanted)).expect("library");
        let error = verify_product(root.path(), &manifest, &item).expect_err("refused");
        assert!(error.contains("imports libcuda.so"), "{error}");
    }

    #[test]
    fn a_product_missing_a_bundled_file_is_refused_naming_it() {
        for missing in ["lib/libnvrtc.so.13", "cuda/include/nvrtc.h"] {
            let error = built_tree_error(|prefix| fs::remove_file(prefix.join(missing)).expect("remove"));
            assert!(error.contains(&format!("missing {missing}")), "{error}");
        }
    }

    #[test]
    fn a_hash_manifest_that_lists_other_files_is_refused_naming_both_sides() {
        let error = built_tree_error(|prefix| {
            let text = fs::read_to_string(prefix.join(HASHES)).expect("manifest");
            let kept: String = text
                .lines()
                .filter(|line| !line.ends_with("lib/libnvrtc.so.13"))
                .map(|line| format!("{line}\n"))
                .collect();
            fs::write(prefix.join(HASHES), format!("{kept}0  lib/ghost.so\n")).expect("manifest");
        });
        assert!(
            error.contains("unlisted") && error.contains("lib/libnvrtc.so.13"),
            "{error}"
        );
        assert!(error.contains("lib/ghost.so"), "{error}");
    }
}
