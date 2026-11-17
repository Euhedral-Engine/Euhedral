//! The HTTP API's protocol: the request and reply types, the errors and the text processing that turn a request into
//! engine input and an engine result into a reply. The crate does no I/O; the server crate carries the bytes.

pub mod template;
