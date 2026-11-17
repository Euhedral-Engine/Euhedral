//! The served-model check every API surface makes first.

use crate::error::ApiError;

/// The model a request names, which must be the served one; a missing or blank name is an invalid request.
/// `missing` is the message when the name is absent or blank.
pub(crate) fn require_served_model(model: Option<&str>, served: &str, missing: &str) -> Result<String, ApiError> {
    match model {
        Some(model) if !is_blank(model) => {
            if model == served {
                Ok(model.to_owned())
            } else {
                Err(ApiError::model_not_found(model))
            }
        }
        _ => Err(ApiError::invalid_request(missing, Some("model"))),
    }
}

/// Blank text: empty or only whitespace as the wire format counts it, which is the space separators other than the
/// no-break spaces, the line and paragraph separators and the controls U+0009 to U+000D and U+001C to U+001F.
pub(crate) fn is_blank(text: &str) -> bool {
    text.chars().all(|character| {
        matches!(character, '\u{9}'..='\u{d}' | '\u{1c}'..='\u{1f}')
            || (character.is_whitespace() && !matches!(character, '\u{a0}' | '\u{2007}' | '\u{202f}' | '\u{85}'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSING: &str = "model: Field required";

    #[test]
    fn blank_is_empty_or_only_the_wire_formats_whitespace() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\n\u{b}\u{c}\r\u{1c}\u{2003}\u{2028}\u{3000}"));
        assert!(!is_blank("\u{a0}"));
        assert!(!is_blank("\u{2007}"));
        assert!(!is_blank("\u{85}"));
        assert!(!is_blank(" m "));
    }

    #[test]
    fn the_served_model_is_required() {
        assert_eq!(require_served_model(Some("m"), "m", MISSING), Ok("m".to_owned()));
        assert_eq!(
            require_served_model(Some("\u{a0}"), "m", MISSING)
                .expect_err("unknown")
                .status,
            404
        );
        assert_eq!(
            require_served_model(Some("  "), "m", MISSING)
                .expect_err("blank")
                .param
                .as_deref(),
            Some("model")
        );
        let missing = require_served_model(None, "m", MISSING).expect_err("missing");
        assert_eq!(missing.status, 400);
        assert_eq!(missing.message, MISSING);
    }
}
