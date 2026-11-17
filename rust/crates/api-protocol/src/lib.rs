//! The HTTP API's protocol: the request and reply types, the errors and the text processing that turn a request into
//! engine input and an engine result into a reply. The crate does no I/O; the server crate carries the bytes.

pub mod anthropic;
pub mod chat_mapper;
pub mod coerce;
pub mod error;
pub mod generation;
pub mod json;
pub mod openai;
pub mod pipeline;
pub mod reasoning;
pub mod response_format;
pub mod responses;
pub mod template;
pub mod tool_calling;
mod tool_definition;
pub mod tool_results;
mod wire;

pub use chat_mapper::map_chat;
pub use error::{ApiError, Surface};
pub use tool_definition::ToolDefinition;
pub use wire::ToJson;
