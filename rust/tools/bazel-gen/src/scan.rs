//! What the sources say that the manifests do not: files a macro includes from outside the package, and the sibling test
//! files a test names with `mod`.

use std::fs;
use std::path::{Path, PathBuf};

/// A source file's `include_str!` / `include_bytes!` literals, as written.
pub fn includes(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for mac in ["include_str!", "include_bytes!"] {
        let mut rest = text;
        while let Some(at) = rest.find(mac) {
            rest = &rest[at + mac.len()..];
            let Some(args) = rest.trim_start().strip_prefix('(') else {
                continue;
            };
            let Some(args) = args.trim_start().strip_prefix('"') else {
                continue;
            };
            if let Some(end) = args.find('"') {
                found.push(args[..end].to_owned());
            }
        }
    }
    found
}

/// The names `mod name;` declares (declarations without a body).
pub fn modules(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix("pub ").unwrap_or(line);
            let name = line.strip_prefix("mod ")?.strip_suffix(';')?;
            Some(name.trim().to_owned())
        })
        .collect()
}

/// `path` with `.` and `..` folded away, without touching the file system.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part.as_os_str().to_str() {
            Some(".") => {}
            Some("..") => {
                out.pop();
            }
            _ => out.push(part),
        }
    }
    out
}

/// Every `.rs` file under `dir` (recursively), skipping Cargo's output.
pub fn rust_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        let entries = match fs::read_dir(&next) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}
