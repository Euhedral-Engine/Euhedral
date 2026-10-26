//! The engine's settings: one YAML file, overridden key by key from the environment, validated in one place.
//!
//! A variable named `EUHEDRAL_` plus the key path upper-cased with `.` and `-` written as `_` replaces the file's value
//! for that key, or supplies it when the file lacks it (`EUHEDRAL_MODEL_ARTIFACT` for `model.artifact`); `PORT` is
//! accepted for `server.port`. Sizes take binary units (`4GiB`, `700MiB`; a plain number is bytes) and durations take
//! units (`10s`, `30m`, `1000ms`; `0` is zero).
//!
//! An unknown or duplicate key in the file is an error, and so is a variable named in the retired list, an empty
//! variable and a value out of range; each error names the key and the variable. Every problem found is reported
//! together. Other variables are ignored.
//!
//! An optional key written `auto` loads as [`Auto::Auto`]: this crate only represents it, and the engine computes the
//! value at run time. This crate is the one place the engine reads its configuration; it reads the process environment
//! only through [`process_env`], which a binary's `main` passes to [`load`].
//!
//! The key layout is that of `config/euhedral.example.yaml`.

mod loader;
mod model;
mod retired;
mod sizes;

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

pub use loader::{SettingsError, SettingsErrors, known_keys, variable_name};
pub use model::{
    Auto, BatchingSettings, ContextSettings, CpuList, HostLayerCompute, HostLayersSettings, HostSettings,
    KernelSettings, KvFormat, MIN_TENSOR_CORE_ROWS, MemorySettings, ModelSettings, PrefixCacheSettings,
    RequestSettings, SamplingSettings, ServerSettings, Settings, TrellisKernelSettings, WeightStreamingSettings,
};
pub use retired::{key_message as retired_key_message, message as retired_message};
pub use sizes::{ByteSize, parse_duration};

/// The configuration file named by `--config <path>` or `--config=<path>` in `args`, else by `EUHEDRAL_CONFIG`; `None`
/// when neither is given, which means the settings come from variables alone. A `--config=` argument that is not valid
/// Unicode is not recognised; give such a path as `--config <path>`.
pub fn locate(args: &[OsString], env: &dyn Fn(&str) -> Option<String>) -> Result<Option<PathBuf>, SettingsErrors> {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let value = if arg == "--config" {
            args.next().map(OsString::as_os_str).unwrap_or_default()
        } else if let Some(value) = arg.to_str().and_then(|arg| arg.strip_prefix("--config=")) {
            OsStr::new(value)
        } else {
            continue;
        };
        if value.to_string_lossy().trim().is_empty() {
            return Err(SettingsError::new("--config needs a file path").into());
        }
        return Ok(Some(PathBuf::from(value)));
    }
    Ok(env("EUHEDRAL_CONFIG")
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from))
}

/// Loads the settings: the file named by `--config` or `EUHEDRAL_CONFIG` if any, then the environment, then validation.
///
/// Reports every problem found.
pub fn load(args: &[OsString], env: &dyn Fn(&str) -> Option<String>) -> Result<Settings, SettingsErrors> {
    let file = locate(args, env)?;
    loader::load_optional(file.as_deref(), env)
}

/// Loads the settings from `path`, then the environment, then validation.
///
/// Reports every problem found.
pub fn load_file(path: &Path, env: &dyn Fn(&str) -> Option<String>) -> Result<Settings, SettingsErrors> {
    loader::load_optional(Some(path), env)
}

/// The process environment, for [`load`]. A variable that is not valid Unicode reads as absent.
// The environment is read here and nowhere else in the engine.
#[allow(clippy::disallowed_methods)]
pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
