//! Kernel launches as data: the catalog of every kernel the Rust engine launches, and the families that plan them.

pub mod attention;
pub mod catalog;
pub mod elementwise;
pub mod embedding;
pub mod gdn;
#[cfg(target_os = "linux")]
pub mod load;
pub mod nvfp4;
#[cfg(target_os = "linux")]
pub mod queue;
pub mod table;
