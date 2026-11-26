//! The EXL3 checkpoint the trellis quantizer writes (exllamav3's layout), and the choices that precede it.
//!
//! - [`allocation`]: the bits per weight of each linear for a target average.
//! - [`linears`]: the text linears of a Qwen 3.8 model, where their rows sit in the fused objects, and their
//!   allocation inputs.
//! - [`layout`]: what the written files are, as plain values: the tensors of a quantized linear, the ledger of what
//!   has been written, the JSON documents, the half-precision conversion.
//! - [`writer`]: [`Exl3Writer`], which writes the files through `AsyncIo`.
//!
//! Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).

pub mod allocation;
pub mod layout;
pub mod linears;
pub mod writer;

pub use layout::{MTP_BITS, MTP_FILE, MUL1, Quantized, auxiliary_files, closing_files, layer_file};
pub use writer::{Done, Exl3Settings, Exl3Writer, Listing};
