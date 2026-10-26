//! Machine settings for tests.
//!
//! Tests that need a model artifact, a tokenizer directory or another machine-specific path never hard-code it. They
//! ask for a `euhedral.*` key, which is looked up in the environment first and then in Gradle's
//! `gradle.properties`, the file the Java tests already read.
//!
//! For the key `euhedral.qwen.artifact` the environment variable is `EUHEDRAL_TEST_QWEN_ARTIFACT`: the `euhedral.`
//! prefix is dropped, the rest is upper-cased, and `.` and `-` become `_`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const KEY_PREFIX: &str = "euhedral.";
const ENV_PREFIX: &str = "EUHEDRAL_TEST_";

/// The value of a `euhedral.*` key: the environment variable `EUHEDRAL_TEST_<KEY>` first, then the Gradle user home's
/// `gradle.properties` (`GRADLE_USER_HOME`, else `~/.gradle`).
#[allow(
    clippy::disallowed_methods,
    reason = "single reader of the environment: this test helper resolves machine settings for tests"
)]
pub fn gradle_property(key: &str) -> Option<String> {
    lookup(key, |name| env::var(name).ok(), gradle_user_home().as_deref())
}

/// Like [`gradle_property`], but prints a skip line and returns `None` when the key is absent, so a test can
/// `let Some(path) = require_property(key) else { return };`. libtest captures the line from a passing test; run
/// with `--nocapture` to see it.
#[allow(clippy::print_stdout, reason = "the skip line is this helper's output")]
pub fn require_property(key: &str) -> Option<String> {
    let value = gradle_property(key);
    if value.is_none() {
        println!("skipped: set {} or {key} in gradle.properties", env_name(key));
    }
    value
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

/// The lookup with the environment and the Gradle home injected, so tests do not touch the real ones.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test helper reads the properties file"
)]
fn lookup(key: &str, environment: impl Fn(&str) -> Option<String>, gradle_home: Option<&Path>) -> Option<String> {
    if let Some(value) = environment(&env_name(key)).filter(|value| !value.is_empty()) {
        return Some(value);
    }
    let text = fs::read_to_string(gradle_home?.join("gradle.properties")).ok()?;
    properties_value(&text, key)
}

#[allow(
    clippy::disallowed_methods,
    reason = "single reader of the environment: this test helper resolves machine settings for tests"
)]
fn gradle_user_home() -> Option<PathBuf> {
    if let Some(home) = env::var_os("GRADLE_USER_HOME").filter(|home| !home.is_empty()) {
        return Some(PathBuf::from(home));
    }
    let user = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(user).join(".gradle"))
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

    #[test]
    fn gradle_property_reads_env_then_file() {
        let home = tempfile::tempdir().unwrap();
        fs::write(
            home.path().join("gradle.properties"),
            "# machine settings\neuhedral.qwen.artifact = /models/q3.edrl\neuhedral.qwen.tokenizer-dir:/models/tok\n",
        )
        .unwrap();
        let no_env = |_: &str| None;
        let gradle_home = Some(home.path());

        // The file answers when the environment does not.
        assert_eq!(
            lookup("euhedral.qwen.artifact", no_env, gradle_home).as_deref(),
            Some("/models/q3.edrl")
        );
        assert_eq!(
            lookup("euhedral.qwen.tokenizer-dir", no_env, gradle_home).as_deref(),
            Some("/models/tok")
        );

        // The environment wins over the file, under the name EUHEDRAL_TEST_<KEY>.
        let env = |name: &str| (name == "EUHEDRAL_TEST_QWEN_ARTIFACT").then(|| "/from/env".to_owned());
        assert_eq!(
            lookup("euhedral.qwen.artifact", env, gradle_home).as_deref(),
            Some("/from/env")
        );
        assert_eq!(
            lookup("euhedral.qwen.tokenizer-dir", env, gradle_home).as_deref(),
            Some("/models/tok")
        );

        // An absent key, a missing file and no Gradle home are all `None`.
        assert_eq!(lookup("euhedral.qwen.missing", no_env, gradle_home), None);
        assert_eq!(
            lookup("euhedral.qwen.artifact", no_env, Some(&home.path().join("absent"))),
            None
        );
        assert_eq!(lookup("euhedral.qwen.artifact", no_env, None), None);
    }

    #[test]
    fn properties_values_are_unescaped_like_java() {
        let text = "euhedral.a=C:\\\\models\\\\q3\neuhedral.b=C\\:/models/q3\n";
        assert_eq!(properties_value(text, "euhedral.a").as_deref(), Some("C:\\models\\q3"));
        assert_eq!(properties_value(text, "euhedral.b").as_deref(), Some("C:/models/q3"));
    }

    #[test]
    fn env_name_drops_the_prefix_and_maps_separators() {
        assert_eq!(
            env_name("euhedral.qwen.tokenizer-dir"),
            "EUHEDRAL_TEST_QWEN_TOKENIZER_DIR"
        );
    }

    #[test]
    fn require_property_returns_none_for_an_absent_key() {
        assert_eq!(require_property("euhedral.no-such-key.for-tests"), None);
    }
}
