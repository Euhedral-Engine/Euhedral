//! The HTTP API's protocol: the request and reply types, the errors and the text processing that turn a request into
//! engine input and an engine result into a reply. The crate does no I/O; the server crate carries the bytes.

pub mod anthropic;
pub mod coerce;
pub mod error;
pub mod json;
pub mod openai;
pub mod pipeline;
pub mod responses;
pub mod template;
mod tool_definition;
mod wire;

pub use error::{ApiError, Surface};
pub use tool_definition::ToolDefinition;
pub use wire::ToJson;
