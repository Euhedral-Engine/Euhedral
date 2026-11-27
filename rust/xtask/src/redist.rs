//! The pinned CUDA redistributables: download into a cache with a checksum, unpack into the input layout the native
//! build reads.
//!
//! The layout under the destination is `include/` (the headers), `lib/` (the link libraries: `cuda`, `cudart` and
//! `nvrtc`, as `.lib` on Windows and as `libcuda.so` (a driver stub), `libcudart.so` and `libnvrtc.so` on Linux) and
//! `runtime/` (the libraries the product loads at run time). Only regular files of the archives that the layout names
//! are written; an entry whose name could leave the destination, or that is a link or a device at a name the layout
//! uses, refuses the whole archive.

use std::fmt::Display;
use std::fs::{self, File};
use std::io::{self, BufReader, Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use tar::{Archive, EntryType};
use zip::ZipArchive;

use crate::native::Platform;

const DOWNLOAD_ROOT: &str = "https://developer.download.nvidia.com/compute/cuda/redist/";
const UNIX_FILE_TYPE: u32 = 0o170_000;
const UNIX_REGULAR: u32 = 0o100_000;
const WINDOWS_LINK_LIBRARIES: [&str; 3] = ["cuda.lib", "cudart.lib", "nvrtc.lib"];
const WINDOWS_RUNTIME: [&str; 3] = ["cudart64_13.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_131.dll"];

/// Fetches a download into a destination file; a fake stands in for the network in tests.
pub(crate) type Download<'a> = &'a dyn Fn(&str, &Path) -> Result<(), String>;

/// The SHA-256 of the file at `path`, in lower-case hexadecimal.
pub(crate) fn sha256_hex(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    io::copy(&mut file, &mut digest).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Downloads `url` with `curl`, which Linux images and Windows 10 and later ship.
pub(crate) fn download_with_curl(url: &str, destination: &Path) -> Result<(), String> {
    let status = Command::new("curl")
        .args(["--fail", "--silent", "--show-error", "--location"])
        .args(["--connect-timeout", "30", "--max-time", "900", "--output"])
        .arg(destination)
        .arg(url)
        .status()
        .map_err(|error| format!("cannot run curl to fetch {url}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("downloading {url} failed: {status}"))
    }
}

/// The archive `relative` (a path under the redistributable root) in `cache`, downloaded when the cache does not hold
/// a file with the checksum `sha256`. A download that does not match is deleted and refused.
pub(crate) fn fetch(cache: &Path, relative: &str, sha256: &str, download: Download) -> Result<PathBuf, String> {
    fs::create_dir_all(cache).map_err(|error| format!("cannot create {}: {error}", cache.display()))?;
    let name = relative.rsplit('/').next().unwrap_or(relative);
    let archive = cache.join(name);
    if archive.is_file() && sha256_hex(&archive)? == sha256 {
        return Ok(archive);
    }
    let partial = cache.join(format!("{name}.part"));
    let url = format!("{DOWNLOAD_ROOT}{relative}");
    let outcome = download(&url, &partial).and_then(|()| {
        let actual = sha256_hex(&partial)?;
        if actual == sha256 {
            fs::rename(&partial, &archive)
                .map_err(|error| format!("cannot move {} into the cache: {error}", partial.display()))
        } else {
            Err(format!(
                "CUDA download checksum mismatch for {url}: expected {sha256}, got {actual}"
            ))
        }
    });
    if outcome.is_err() {
        drop(fs::remove_file(&partial));
    }
    outcome.map(|()| archive)
}

/// Where the archive entry `name` goes under the destination, or `None` when the layout does not take it.
///
/// Refuses a name that is absolute or holds a `..` component, whether or not the layout would take it.
pub(crate) fn layout_path(name: &str, platform: Platform, cuda_version: &str) -> Result<Option<String>, String> {
    refuse_escape(name)?;
    let base = name.rsplit('/').next().unwrap_or(name);
    if let Some((_, header)) = name.split_once("/include/")
        && !header.is_empty()
    {
        return Ok(Some(format!("include/{header}")));
    }
    let cudart = format!("libcudart.so.{cuda_version}");
    let nvrtc = format!("libnvrtc.so.{cuda_version}");
    let builtins = format!("libnvrtc-builtins.so.{cuda_version}");
    Ok(match platform {
        Platform::Windows if name.contains("/lib/x64/") && WINDOWS_LINK_LIBRARIES.contains(&base) => {
            Some(format!("lib/{base}"))
        }
        Platform::Windows if name.contains("/bin/x64/") && WINDOWS_RUNTIME.contains(&base) => {
            Some(format!("runtime/{base}"))
        }
        Platform::Linux if name.contains("/lib/stubs/") && base == "libcuda.so" => Some("lib/libcuda.so".to_string()),
        Platform::Linux if name.contains("/lib/") && base == cudart => Some("lib/libcudart.so".to_string()),
        Platform::Linux if name.contains("/lib/") && base == nvrtc => Some("lib/libnvrtc.so".to_string()),
        Platform::Linux if name.contains("/lib/") && base == builtins => Some(format!("runtime/{base}")),
        _ => None,
    })
}

/// Refuses an archive entry name that could write outside the destination.
fn refuse_escape(name: &str) -> Result<(), String> {
    let drive = name.as_bytes().get(1) == Some(&b':');
    if name.starts_with(['/', '\\']) || drive {
        return Err(format!("archive entry {name:?} is an absolute path"));
    }
    if name.split(['/', '\\']).any(|part| part == "..") {
        return Err(format!(
            "archive entry {name:?} escapes the destination (it holds `..`)"
        ));
    }
    Ok(())
}

/// Unpacks the archive at `archive` into `destination` (a `.zip` for Windows, a `.tar.xz` for Linux).
pub(crate) fn unpack(archive: &Path, destination: &Path, platform: Platform, cuda_version: &str) -> Result<(), String> {
    let file = File::open(archive).map_err(|error| format!("cannot read {}: {error}", archive.display()))?;
    let failed = |error: &dyn Display| format!("cannot unpack {}: {error}", archive.display());
    match platform {
        Platform::Windows => unpack_zip(BufReader::new(file), destination, platform, cuda_version),
        Platform::Linux => {
            let mut tar = Vec::new();
            lzma_rs::xz_decompress(&mut BufReader::new(file), &mut tar).map_err(|error| failed(&error))?;
            unpack_tar(Cursor::new(tar), destination, platform, cuda_version)
        }
    }
    .map_err(|error| format!("{}: {error}", archive.display()))
}

/// Unpacks a tar stream (already decompressed) into `destination`.
pub(crate) fn unpack_tar(
    tar: impl Read,
    destination: &Path,
    platform: Platform,
    cuda_version: &str,
) -> Result<(), String> {
    let mut archive = Archive::new(tar);
    let entries = archive
        .entries()
        .map_err(|error| format!("cannot read the tar stream: {error}"))?;
    for entry in entries {
        let mut entry = entry.map_err(|error| format!("cannot read a tar entry: {error}"))?;
        let name = String::from_utf8_lossy(&entry.path_bytes()).into_owned();
        let target = layout_path(&name, platform, cuda_version)?;
        let Some(path) = target else {
            continue;
        };
        match entry.header().entry_type() {
            EntryType::Regular => install(destination, &path, &mut entry)?,
            EntryType::Directory => {}
            other => return Err(refuse_non_file(&name, &format!("{other:?}"))),
        }
    }
    Ok(())
}

/// Unpacks a zip archive into `destination`.
pub(crate) fn unpack_zip(
    zip: impl Read + Seek,
    destination: &Path,
    platform: Platform,
    cuda_version: &str,
) -> Result<(), String> {
    let mut archive = ZipArchive::new(zip).map_err(|error| format!("cannot read the zip archive: {error}"))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("cannot read zip entry {index}: {error}"))?;
        let name = entry.name().to_string();
        let target = layout_path(&name, platform, cuda_version)?;
        let Some(path) = target else {
            continue;
        };
        if entry.is_dir() {
            continue;
        }
        match entry.unix_mode().map(|mode| mode & UNIX_FILE_TYPE) {
            None | Some(UNIX_REGULAR | 0) => install(destination, &path, &mut entry)?,
            Some(other) => return Err(refuse_non_file(&name, &format!("unix file type {other:#o}"))),
        }
    }
    Ok(())
}

/// The refusal of an archive entry that is not a regular file but sits at a name the layout uses: skipping it would
/// leave the input incomplete, and writing it as a file would store a link's target text as a library.
fn refuse_non_file(name: &str, kind: &str) -> String {
    format!("archive entry {name:?} is not a regular file ({kind}) but the layout uses its name")
}

/// Writes `source` to `relative` under `destination`, which `layout_path` has already vetted.
fn install(destination: &Path, relative: &str, source: &mut impl Read) -> Result<(), String> {
    let output = destination.join(relative);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut file = File::create(&output).map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    io::copy(source, &mut file)
        .map(drop)
        .map_err(|error| format!("cannot write {}: {error}", output.display()))
}

/// The runtime libraries a product loads, by the names it loads them under; the unpacked layout holds each in
/// `runtime/` (on Linux after [`link_linux_runtime`]).
pub(crate) fn runtime_files(platform: Platform, cuda_version: &str) -> Vec<String> {
    match platform {
        Platform::Windows => WINDOWS_RUNTIME.iter().map(|name| (*name).to_string()).collect(),
        Platform::Linux => {
            let mut parts = cuda_version.split('.');
            let major = parts.next().unwrap_or_default();
            let minor = parts.next().unwrap_or_default();
            vec![
                format!("libcudart.so.{major}"),
                format!("libnvrtc.so.{major}"),
                format!("libnvrtc-builtins.so.{major}.{minor}"),
            ]
        }
    }
}

/// Adds the Linux runtime names the layout does not take from an archive: the SONAME copies of the two libraries and
/// the builtins under its minor-version name.
pub(crate) fn link_linux_runtime(destination: &Path, cuda_version: &str) -> Result<(), String> {
    let names = runtime_files(Platform::Linux, cuda_version);
    let runtime = destination.join("runtime");
    fs::create_dir_all(&runtime).map_err(|error| format!("cannot create {}: {error}", runtime.display()))?;
    let copies = [
        (PathBuf::from("lib/libcudart.so"), names[0].clone()),
        (PathBuf::from("lib/libnvrtc.so"), names[1].clone()),
        (
            PathBuf::from(format!("runtime/libnvrtc-builtins.so.{cuda_version}")),
            names[2].clone(),
        ),
    ];
    for (from, to) in copies {
        let (from, to) = (destination.join(from), runtime.join(to));
        fs::copy(&from, &to).map_err(|error| format!("cannot copy {} to {}: {error}", from.display(), to.display()))?;
    }
    Ok(())
}
