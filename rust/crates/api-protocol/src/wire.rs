//! The two directions of the API's JSON: replies are built as [`JsonValue`]s and written in compact form; requests
//! are parsed to a [`JsonValue`] and read field by field.

use crate::error::ApiError;
use crate::json::{JsonInteger, JsonValue};

/// A JSON object's members in order.
pub(crate) type Members = Vec<(String, JsonValue)>;

/// A reply (or part of one) that is written as JSON.
pub trait ToJson {
    /// The value, its members in the order they are written.
    fn to_json(&self) -> JsonValue;

    /// The compact JSON text of the value.
    fn to_json_string(&self) -> String {
        self.to_json().to_compact_string()
    }
}

/// An object with the given members, in order.
pub(crate) fn object<const N: usize>(members: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(
        members
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

/// A string value.
pub(crate) fn text(value: &str) -> JsonValue {
    JsonValue::String(value.to_owned())
}

/// The arguments of a tool call: the compact JSON text the call was written as, in which an unpaired surrogate is
/// an escape. As a string value it keeps the escape as the escape rather than as escaped text.
pub(crate) fn arguments_text(value: &str) -> JsonValue {
    JsonValue::from_compact_text(value)
}

/// An error message as a string value; `cut` is the high surrogate its excerpt was cut after, if any.
pub(crate) fn message_text(value: &str, cut: Option<u16>) -> JsonValue {
    JsonValue::from_message_text(value, cut)
}

/// A string value, or null.
pub(crate) fn optional_text(value: Option<&str>) -> JsonValue {
    value.map_or(JsonValue::Null, text)
}

/// An integer value.
pub(crate) fn int(value: impl Into<i128>) -> JsonValue {
    JsonValue::Integer(JsonInteger::from(value.into()))
}

/// An array of the items' values.
pub(crate) fn list<T: ToJson>(items: &[T]) -> JsonValue {
    JsonValue::Array(items.iter().map(ToJson::to_json).collect())
}

/// An array of integers.
pub(crate) fn int_list(items: &[i32]) -> JsonValue {
    JsonValue::Array(items.iter().map(|item| int(*item)).collect())
}

/// An object from members built so far.
pub(crate) fn members(members: Members) -> JsonValue {
    JsonValue::Object(members)
}

/// Appends a member.
pub(crate) fn push(members: &mut Members, key: &str, value: JsonValue) {
    members.push((key.to_owned(), value));
}

/// Appends a member when it has a value.
pub(crate) fn push_some(members: &mut Members, key: &str, value: Option<JsonValue>) {
    if let Some(value) = value {
        push(members, key, value);
    }
}

/// The members of a request object, taken one known key at a time; what is left is the unknown fields in the order
/// they arrived.
pub(crate) struct Reader {
    members: Members,
}

impl Reader {
    /// The members of a request body: one JSON object.
    pub(crate) fn parse(body: &str) -> Result<Reader, ApiError> {
        let value = JsonValue::parse(body).ok_or_else(ApiError::unparsable_body)?;
        Reader::of(value)
    }

    /// The members of an object value; any other value is an unparsable body.
    pub(crate) fn of(value: JsonValue) -> Result<Reader, ApiError> {
        match value {
            JsonValue::Object(members) => Ok(Reader { members }),
            _ => Err(ApiError::unparsable_body()),
        }
    }

    /// Removes and returns the value of `key`; `None` when the key is absent or null.
    pub(crate) fn take(&mut self, key: &str) -> Option<JsonValue> {
        let position = self.members.iter().position(|(name, _)| name == key)?;
        match self.members.remove(position).1 {
            JsonValue::Null => None,
            value => Some(value),
        }
    }

    /// The members not taken, nulls included.
    pub(crate) fn rest(self) -> Members {
        self.members
    }
}
