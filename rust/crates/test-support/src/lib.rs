//! Machine settings for tests.
//!
//! Tests that need a model artifact, a tokenizer directory or another machine-specific path never hard-code it. They
//! ask for a `euhedral.*` key, which is looked up in three places, in this order: the environment, the Rust-owned
//! settings file, and Gradle's `gradle.properties`, the file the Java tests already read.
//!
//! The settings file is the path in `EUHEDRAL_TEST_CONFIG` if that is set, else
//! `$XDG_CONFIG_HOME/euhedral/test.properties`, else `~/.config/euhedral/test.properties`. It uses the `key=value`
//! format of `gradle.properties`.
//!
//! For the key `euhedral.qwen.artifact` the environment variable is `EUHEDRAL_TEST_QWEN_ARTIFACT`: the `euhedral.`
//! prefix is dropped, the rest is upper-cased, and `.` and `-` become `_`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const KEY_PREFIX: &str = "euhedral.";
const ENV_PREFIX: &str = "EUHEDRAL_TEST_";
const CONFIG_VARIABLE: &str = "EUHEDRAL_TEST_CONFIG";

/// The value of a `euhedral.*` key: the environment variable `EUHEDRAL_TEST_<KEY>` first, then the settings file (see
/// the crate docs), then the Gradle user home's `gradle.properties` (`GRADLE_USER_HOME`, else `~/.gradle`).
#[allow(
    clippy::disallowed_methods,
    reason = "single reader of the environment: this test helper resolves machine settings for tests"
)]
pub fn gradle_property(key: &str) -> Option<String> {
    let environment = |name: &str| env::var(name).ok();
    let settings = settings_file(environment, home_dir());
    lookup(key, environment, settings.as_deref(), gradle_user_home().as_deref())
}

/// Like [`gradle_property`], but prints a skip line and returns `None` when the key is absent or names a path that does not exist, so a test can
/// `let Some(path) = require_property(key) else { return };`. libtest captures the line from a passing test; run
/// with `--nocapture` to see it.
#[allow(clippy::print_stdout, reason = "the skip line is this helper's output")]
pub fn require_property(key: &str) -> Option<String> {
    let Some(value) = gradle_property(key) else {
        println!(
            "skipped: set {} or {key} in ~/.config/euhedral/test.properties",
            env_name(key)
        );
        return None;
    };
    if !Path::new(&value).exists() {
        println!("skipped: {key} names a path that does not exist here");
        return None;
    }
    Some(value)
}

/// The environment variable that overrides `key`.
fn env_name(key: &str) -> String {
    let rest = key.strip_prefix(KEY_PREFIX).unwrap_or(key);
    let mapped: String = rest
        .chars()
        .map(|c| {
            if c == '.' || c == '-' {
                '_'
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect();
    format!("{ENV_PREFIX}{mapped}")
}

/// The lookup with the environment, the settings file and the Gradle home injected, so tests touch no real one.
fn lookup(
    key: &str,
    environment: impl Fn(&str) -> Option<String>,
    settings: Option<&Path>,
    gradle_home: Option<&Path>,
) -> Option<String> {
    if let Some(value) = environment(&env_name(key)).filter(|value| !value.is_empty()) {
        return Some(value);
    }
    if let Some(value) = settings.and_then(|file| properties_file_value(file, key)) {
        return Some(value);
    }
    properties_file_value(&gradle_home?.join("gradle.properties"), key)
}

/// The Rust-owned settings file: `EUHEDRAL_TEST_CONFIG` if it is set, else `$XDG_CONFIG_HOME/euhedral/test.properties`,
/// else `~/.config/euhedral/test.properties` under `home`.
fn settings_file(environment: impl Fn(&str) -> Option<String>, home: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = environment(CONFIG_VARIABLE).filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(path));
    }
    if let Some(config) = environment("XDG_CONFIG_HOME").filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(config).join("euhedral").join("test.properties"));
    }
    Some(home?.join(".config").join("euhedral").join("test.properties"))
}

/// The value of `key` in the Java properties file at `path`; a missing or unreadable file has no keys.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test helper reads a properties file"
)]
fn properties_file_value(path: &Path, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    properties_value(&text, key)
}

#[allow(
    clippy::disallowed_methods,
    reason = "single reader of the environment: this test helper resolves machine settings for tests"
)]
fn home_dir() -> Option<PathBuf> {
    let user = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(user))
}

#[allow(
    clippy::disallowed_methods,
    reason = "single reader of the environment: this test helper resolves machine settings for tests"
)]
fn gradle_user_home() -> Option<PathBuf> {
    if let Some(home) = env::var_os("GRADLE_USER_HOME").filter(|home| !home.is_empty()) {
        return Some(PathBuf::from(home));
    }
    Some(home_dir()?.join(".gradle"))
}

/// The value of `key` in Java `.properties` text, unescaped the way Java reads it (`C:\\models` is `C:\models`).
fn properties_value(text: &str, key: &str) -> Option<String> {
    java_properties::read(text.as_bytes()).ok()?.remove(key)
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: the tests write a fixture file"
)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn gradle_property_reads_env_then_file() {
        let home = tempfile::tempdir().unwrap();
        fs::write(
            home.path().join("gradle.properties"),
            "# machine settings\neuhedral.qwen.artifact = /models/q3.edrl\neuhedral.qwen.tokenizer-dir:/models/tok\n",
        )
        .unwrap();
        let gradle_home = Some(home.path());

        // The file answers when the environment does not.
        assert_eq!(
            lookup("euhedral.qwen.artifact", no_env, None, gradle_home).as_deref(),
            Some("/models/q3.edrl")
        );
        assert_eq!(
            lookup("euhedral.qwen.tokenizer-dir", no_env, None, gradle_home).as_deref(),
            Some("/models/tok")
        );

        // The environment wins over the file, under the name EUHEDRAL_TEST_<KEY>.
        let env = |name: &str| (name == "EUHEDRAL_TEST_QWEN_ARTIFACT").then(|| "/from/env".to_owned());
        assert_eq!(
            lookup("euhedral.qwen.artifact", env, None, gradle_home).as_deref(),
            Some("/from/env")
        );
        assert_eq!(
            lookup("euhedral.qwen.tokenizer-dir", env, None, gradle_home).as_deref(),
            Some("/models/tok")
        );

        // An absent key, a missing file and no Gradle home are all `None`.
        assert_eq!(lookup("euhedral.qwen.missing", no_env, None, gradle_home), None);
        assert_eq!(
            lookup(
                "euhedral.qwen.artifact",
                no_env,
                None,
                Some(&home.path().join("absent"))
            ),
            None
        );
        assert_eq!(lookup("euhedral.qwen.artifact", no_env, None, None), None);
    }

    #[test]
    fn settings_file_wins_over_gradle_properties() {
        let dir = tempfile::tempdir().unwrap();
        let gradle = dir.path().join("gradle");
        fs::create_dir(&gradle).unwrap();
        fs::write(
            gradle.join("gradle.properties"),
            "euhedral.qwen.artifact=/gradle/q3.edrl\n",
        )
        .unwrap();
        let settings = dir.path().join("test.properties");
        fs::write(&settings, "euhedral.qwen.artifact=/rust/q3.edrl\n").unwrap();

        assert_eq!(
            lookup("euhedral.qwen.artifact", no_env, Some(&settings), Some(&gradle)).as_deref(),
            Some("/rust/q3.edrl")
        );
    }

    #[test]
    fn environment_wins_over_settings_file() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join("test.properties");
        fs::write(&settings, "euhedral.qwen.artifact=/rust/q3.edrl\n").unwrap();
        let env = |name: &str| (name == "EUHEDRAL_TEST_QWEN_ARTIFACT").then(|| "/from/env".to_owned());

        assert_eq!(
            lookup("euhedral.qwen.artifact", env, Some(&settings), None).as_deref(),
            Some("/from/env")
        );
    }

    #[test]
    fn key_only_in_gradle_properties_is_still_found() {
        let dir = tempfile::tempdir().unwrap();
        let gradle = dir.path().join("gradle");
        fs::create_dir(&gradle).unwrap();
        fs::write(
            gradle.join("gradle.properties"),
            "euhedral.qwen.tokenizer-dir=/gradle/tok\n",
        )
        .unwrap();
        let settings = dir.path().join("test.properties");
        fs::write(&settings, "euhedral.qwen.artifact=/rust/q3.edrl\n").unwrap();

        assert_eq!(
            lookup("euhedral.qwen.tokenizer-dir", no_env, Some(&settings), Some(&gradle)).as_deref(),
            Some("/gradle/tok")
        );
    }

    #[test]
    fn settings_file_path_is_chosen_by_test_config_then_xdg_then_home() {
        let home = PathBuf::from("/fixture-home");

        // EUHEDRAL_TEST_CONFIG names the file outright, even when XDG_CONFIG_HOME is set.
        let named = |name: &str| match name {
            "EUHEDRAL_TEST_CONFIG" => Some("/custom/test.properties".to_owned()),
            "XDG_CONFIG_HOME" => Some("/xdg".to_owned()),
            _ => None,
        };
        assert_eq!(
            settings_file(named, Some(home.clone())),
            Some(PathBuf::from("/custom/test.properties"))
        );

        // Otherwise XDG_CONFIG_HOME, then the home directory.
        let xdg = |name: &str| (name == "XDG_CONFIG_HOME").then(|| "/xdg".to_owned());
        assert_eq!(
            settings_file(xdg, Some(home.clone())),
            Some(PathBuf::from("/xdg/euhedral/test.properties"))
        );
        assert_eq!(
            settings_file(no_env, Some(home)),
            Some(PathBuf::from("/fixture-home/.config/euhedral/test.properties"))
        );
    }

    #[test]
    fn env_name_drops_the_prefix_and_maps_separators() {
        assert_eq!(
            env_name("euhedral.qwen.tokenizer-dir"),
            "EUHEDRAL_TEST_QWEN_TOKENIZER_DIR"
        );
    }
}
