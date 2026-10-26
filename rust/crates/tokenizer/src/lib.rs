//! The Qwen tokenizer an artifact packages.
//!
//! [`Tokenizer::from_section`] builds it from the [`TokenizerSection`](euhedral_inference_edrl::TokenizerSection) the
//! `edrl` crate reads, so the Rust engine tokenizes prompts exactly as the Java engine does. Encoding runs on the
//! `tokenizers` library (byte-level BPE, NFC, one isolated split); decoding is [`TokenStream`], which holds back an
//! incomplete UTF-8 suffix between tokens.

mod byte_level;
mod error;
mod files;
mod stream;
mod tokenizer;

pub use error::TokenizerError;
pub use stream::TokenStream;
pub use tokenizer::Tokenizer;
