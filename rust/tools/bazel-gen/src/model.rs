//! The workspace as `cargo metadata --no-deps` describes it, reduced to what the BUILD files need.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::scan::{includes, modules, normalize, rust_files};

/// Where a dependency applies: everywhere, or only on one family of operating systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Applies {
    Always,
    Unix,
    Windows,
}

/// One dependency edge of a package.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dep {
    /// The Bazel label: `//rust/crates/x:y` for a member, `@euhedral_rustee//...` for a RustEE crate, `@euhedral_crates//:name` for anything else.
    pub label: String,
    pub applies: Applies,
    pub dev: bool,
}

#[derive(Clone, Debug)]
pub struct Target {
    /// The cargo target name (a bin keeps its hyphens).
    pub name: String,
    /// The source file, relative to the package.
    pub root: String,
}

#[derive(Debug)]
pub struct Package {
    pub name: String,
    /// The package directory relative to the repository root, with `/` separators.
    pub dir: String,
    pub version: String,
    pub edition: String,
    /// The features the package is built with in the one world (sorted).
    pub features: Vec<String>,
    pub lib: Option<Target>,
    pub bins: Vec<Target>,
    pub tests: Vec<Target>,
    pub deps: Vec<Dep>,
    /// Files outside the package that its sources include (`include_str!`), by path from the repository root.
    pub outside: Vec<String>,
    /// For each test, the sibling test files it declares with `mod` (`tests/common.rs` for `mod common;`).
    pub test_modules: BTreeMap<String, Vec<String>>,
}

#[derive(Debug)]
pub struct Workspace {
    pub packages: Vec<Package>,
}

impl Package {
    /// The Bazel name of the library: the crate name.
    pub fn lib_name(&self) -> String {
        self.name.replace('-', "_")
    }
}

/// The RustEE crates this workspace takes from the module `euhedral_rustee`.
const RUSTEE_CRATES: &[&str] = &[
    "euhedral-core",
    "euhedral-data-structures",
    "euhedral-gpu",
    "euhedral-graph",
    "euhedral-hardware-utils",
    "euhedral-hashing",
    "euhedral-io",
    "euhedral-memory",
    "euhedral-net",
];

/// The target of a RustEE crate. One feature world, the one `cargo test --workspace` builds: RustEE names it `cuda_testing`
/// (cuda, fake and test-hooks on gpu), and the targets of that world agree on which copy of each crate they use.
fn rustee_label(crate_name: &str) -> Option<String> {
    RUSTEE_CRATES.contains(&crate_name).then(|| {
        format!(
            "@euhedral_rustee//{crate_name}:{}_cuda_testing",
            crate_name.replace('-', "_")
        )
    })
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("metadata: no string `{key}`"))
}

fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).and_then(Value::as_array).map_or(&[], Vec::as_slice)
}

fn strings(value: &Value, key: &str) -> Vec<String> {
    list(value, key)
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn slashes(path: &str) -> String {
    path.replace('\\', "/")
}

/// Reads the metadata, resolves the features and refuses what it does not model.
pub fn load(metadata: &str, root: &Path) -> Result<Workspace, String> {
    let metadata: Value = serde_json::from_str(metadata).map_err(|error| format!("metadata: {error}"))?;
    let members: BTreeSet<String> = strings(&metadata, "workspace_members").into_iter().collect();
    let root_text = slashes(&root.to_string_lossy());
    let members: Vec<&Value> = list(&metadata, "packages")
        .iter()
        .filter(|package| text(package, "id").is_ok_and(|id| members.contains(id)))
        .collect();

    let names: BTreeSet<String> = members
        .iter()
        .filter_map(|package| text(package, "name").ok().map(str::to_owned))
        .collect();
    let mut features = resolve_features(&members, &names)?;

    let mut packages = Vec::new();
    for package in &members {
        let name = text(package, "name")?.to_owned();
        let manifest = slashes(text(package, "manifest_path")?);
        let dir = manifest
            .strip_suffix("/Cargo.toml")
            .and_then(|dir| dir.strip_prefix(&format!("{root_text}/")))
            .ok_or_else(|| format!("{name}: manifest {manifest} is not under {root_text}"))?
            .to_owned();
        let mut lib = None;
        let mut bins = Vec::new();
        let mut tests = Vec::new();
        for target in list(package, "targets") {
            let kinds = strings(target, "kind");
            let target_name = text(target, "name")?.to_owned();
            let src = slashes(text(target, "src_path")?);
            let root = src
                .strip_prefix(&format!("{}/", dir_abs(&manifest)))
                .ok_or_else(|| format!("{name}: {src} is outside the package"))?
                .to_owned();
            if !list(target, "required-features").is_empty() {
                return Err(format!(
                    "{name}: target {target_name} has required-features, which are not modelled"
                ));
            }
            let entry = Target {
                name: target_name.clone(),
                root,
            };
            match kinds.as_slice() {
                [kind] if kind == "lib" => lib = Some(entry),
                [kind] if kind == "bin" => bins.push(entry),
                [kind] if kind == "test" => tests.push(entry),
                other => {
                    return Err(format!(
                        "{name}: target {target_name} of kind {other:?} is not modelled"
                    ));
                }
            }
        }
        let mut deps = Vec::new();
        for dep in list(package, "dependencies") {
            let dep_name = text(dep, "name")?;
            match dep.get("kind").and_then(Value::as_str) {
                None => {}
                Some("dev") => {}
                Some(kind) => return Err(format!("{name}: a {kind} dependency ({dep_name}) is not modelled")),
            }
            if dep.get("rename").is_some_and(|rename| !rename.is_null()) {
                return Err(format!("{name}: the renamed dependency {dep_name} is not modelled"));
            }
            if dep.get("optional").and_then(Value::as_bool) == Some(true) {
                return Err(format!("{name}: the optional dependency {dep_name} is not modelled"));
            }
            let applies = match dep.get("target").and_then(Value::as_str) {
                None => Applies::Always,
                Some("cfg(unix)") => Applies::Unix,
                Some("cfg(windows)") => Applies::Windows,
                Some(other) => return Err(format!("{name}: the target `{other}` of {dep_name} is not modelled")),
            };
            let dev = dep.get("kind").and_then(Value::as_str) == Some("dev");
            let label = if names.contains(dep_name) {
                if dep_name == name {
                    continue;
                }
                let owner = members
                    .iter()
                    .find(|other| text(other, "name").is_ok_and(|n| n == dep_name))
                    .ok_or_else(|| format!("{dep_name} vanished"))?;
                if !list(owner, "targets")
                    .iter()
                    .any(|target| strings(target, "kind") == ["lib"])
                {
                    return Err(format!("{name}: {dep_name} has no library"));
                }
                let owner_manifest = slashes(text(owner, "manifest_path")?);
                let owner_dir = owner_manifest
                    .strip_suffix("/Cargo.toml")
                    .and_then(|dir| dir.strip_prefix(&format!("{root_text}/")))
                    .ok_or_else(|| format!("{dep_name}: manifest outside the workspace"))?;
                format!("//{owner_dir}:{}", dep_name.replace('-', "_"))
            } else if let Some(label) = rustee_label(dep_name) {
                label
            } else if dep_name.starts_with("euhedral-") {
                return Err(format!(
                    "{name}: {dep_name} is a RustEE crate with no known Bazel target (see rustee_label)"
                ));
            } else {
                format!("@euhedral_crates//:{dep_name}")
            };
            deps.push(Dep { label, applies, dev });
        }
        deps.sort();
        deps.dedup();
        tests.sort_by(|a, b| a.name.cmp(&b.name));
        bins.sort_by(|a, b| a.name.cmp(&b.name));
        let (outside, test_modules) = scan_package(root, &dir, &tests)?;
        packages.push(Package {
            outside,
            test_modules,
            version: text(package, "version")?.to_owned(),
            edition: text(package, "edition")?.to_owned(),
            features: features.remove(&name).unwrap_or_default().into_iter().collect(),
            dir: dir.clone(),
            name,
            lib,
            bins,
            tests,
            deps,
        });
    }
    packages.sort_by(|a, b| a.dir.cmp(&b.dir));
    Ok(Workspace { packages })
}

/// The includes that leave a package, and the sibling files each test names with `mod`.
type Scanned = (Vec<String>, BTreeMap<String, Vec<String>>);

/// Reads the package's sources for the includes that leave it and the `mod` lines of its tests.
fn scan_package(root: &Path, dir: &str, tests: &[Target]) -> Result<Scanned, String> {
    let package_dir = root.join(dir);
    let mut outside = BTreeSet::new();
    for file in rust_files(&package_dir)? {
        let text = fs::read_to_string(&file).map_err(|error| format!("read {}: {error}", file.display()))?;
        for literal in includes(&text) {
            let target = normalize(&file.parent().unwrap_or(&package_dir).join(&literal));
            if !target.starts_with(&package_dir) {
                let relative = target
                    .strip_prefix(root)
                    .map_err(|_| format!("{} leaves the repository", file.display()))?;
                outside.insert(slashes(&relative.to_string_lossy()));
            }
        }
    }
    let mut modules_of = BTreeMap::new();
    for test in tests {
        let file = package_dir.join(&test.root);
        let text = fs::read_to_string(&file).map_err(|error| format!("read {}: {error}", file.display()))?;
        let siblings: Vec<String> = modules(&text)
            .into_iter()
            .map(|name| format!("tests/{name}.rs"))
            .filter(|path| package_dir.join(path).is_file() && *path != test.root)
            .collect();
        if !siblings.is_empty() {
            modules_of.insert(test.name.clone(), siblings);
        }
    }
    Ok((outside.into_iter().collect(), modules_of))
}

fn dir_abs(manifest: &str) -> &str {
    manifest.strip_suffix("/Cargo.toml").unwrap_or(manifest)
}

/// The features of every member in the one world: each member's defaults, plus what any member's dependency edge
/// (normal, dev, or on itself) asks for, closed under what a feature enables.
fn resolve_features(
    members: &[&Value],
    names: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let declared: BTreeMap<String, &Value> = members
        .iter()
        .filter_map(|package| Some((text(package, "name").ok()?.to_owned(), package.get("features")?)))
        .collect();
    let mut on: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut pending: Vec<(String, String)> = Vec::new();
    for package in members {
        let name = text(package, "name")?;
        for dep in list(package, "dependencies") {
            let dep_name = text(dep, "name")?;
            if !names.contains(dep_name) {
                continue;
            }
            if dep.get("uses_default_features").and_then(Value::as_bool) != Some(false)
                && declared[dep_name].get("default").is_some()
            {
                pending.push((dep_name.to_owned(), "default".to_owned()));
            }
            for feature in strings(dep, "features") {
                pending.push((dep_name.to_owned(), feature));
            }
        }
        if declared[name].get("default").is_some() && !pending.contains(&(name.to_owned(), "default".to_owned())) {
            pending.push((name.to_owned(), "default".to_owned()));
        }
    }
    while let Some((package, feature)) = pending.pop() {
        if !on.entry(package.clone()).or_default().insert(feature.clone()) {
            continue;
        }
        let enables = declared[package.as_str()]
            .get(&feature)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{package} has no feature `{feature}`"))?;
        for entry in enables.iter().filter_map(Value::as_str) {
            match entry.split_once('/') {
                None if entry.starts_with("dep:") => return Err(format!("{package}: `{entry}` is not modelled")),
                None => pending.push((package.clone(), entry.to_owned())),
                Some((dep, _)) if dep.ends_with('?') => {
                    return Err(format!("{package}: the weak feature `{entry}` is not modelled"));
                }
                Some((dep, enabled)) if names.contains(dep) => pending.push((dep.to_owned(), enabled.to_owned())),
                Some(_) => {
                    return Err(format!(
                        "{package}: `{entry}` enables a feature of an outside crate, which is not modelled"
                    ));
                }
            }
        }
    }
    Ok(on)
}
