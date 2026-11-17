//! The tool results owed to one assistant turn's calls, from any API surface.

use crate::error::ApiError;
use crate::template::Message;

/// The tool results owed to one assistant turn's calls. The template renders results without their ids, so they are
/// emitted in call order, which is how the model pairs them with its calls; every call needs exactly one result
/// before the conversation goes on. `id_field` names the request field that carries the id and `calls_path` the
/// calls in the request, for errors.
#[derive(Clone, Debug)]
pub struct ToolResults {
    ids: Vec<String>,
    results: Vec<Option<Message>>,
    calls_path: String,
    id_field: String,
}

impl ToolResults {
    pub fn new(ids: Vec<String>, calls_path: impl Into<String>, id_field: impl Into<String>) -> ToolResults {
        ToolResults {
            results: vec![None; ids.len()],
            ids,
            calls_path: calls_path.into(),
            id_field: id_field.into(),
        }
    }

    /// Records the result `content` for the call `id` (`None` or empty when the result carries none); `path` names
    /// the id's request field.
    pub fn answer(&mut self, id: Option<&str>, content: String, path: &str) -> Result<(), ApiError> {
        let name = &self.id_field;
        let Some(call_id) = id.filter(|id| !id.is_empty()) else {
            return Err(ApiError::invalid_request(
                format!("Tool results require '{name}'."),
                Some(path),
            ));
        };
        let Some(position) = self.ids.iter().position(|known| known == call_id) else {
            return Err(ApiError::invalid_request(
                format!("'{name}' {call_id} does not match a tool call of the preceding assistant message."),
                Some(path),
            ));
        };
        if self.results[position].is_some() {
            return Err(ApiError::invalid_request(
                format!("Tool call {call_id} already has a response."),
                Some(path),
            ));
        }
        self.results[position] = Some(Message::tool(content));
        Ok(())
    }

    /// Appends the results in call order; refuses a conversation that goes on before every call was answered.
    pub fn close_into(&self, messages: &mut Vec<Message>) -> Result<(), ApiError> {
        let missing: Vec<&str> = self
            .ids
            .iter()
            .zip(&self.results)
            .filter(|(_, result)| result.is_none())
            .map(|(id, _)| id.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(ApiError::invalid_request(
                format!(
                    "An assistant message with tool calls must be followed by a result for each call, matched by \
                     '{}'. Missing responses: {}.",
                    self.id_field,
                    missing.join(", ")
                ),
                Some(&self.calls_path),
            ));
        }
        messages.extend(self.results.iter().flatten().cloned());
        Ok(())
    }
}
