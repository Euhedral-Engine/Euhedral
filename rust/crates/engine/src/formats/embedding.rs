//! Token embedding lookups of a Q3 or a plain NVFP4 table (`GroupedLinearFormat.embed`, `Nvfp4Format.embed`).
//!
//! The table is `[vocabulary, hidden]`. In Q3 (64-value groups) either layout is read: the row-split layout looks rows
//! up directly, and the P2E2 layout decodes the entropy-coded rows of the ids it is asked for. In NVFP4 the plain
//! layout is read by `euhedral_nvfp4_embedding`, given where the table's scale plane and global scale start; an SD4
//! table has no lookup. An id outside the vocabulary leaves its output row as it was.

use euhedral_inference_cuda::{CudaError, DevicePtr, KernelOps};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::formats::shapes::{KernelArguments, LinearKernel};
use crate::model::WeightRef;
use crate::residency::planner::nvfp4;

/// The token embedding lookup of a Q3 or a plain NVFP4 table.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenEmbedding;

impl TokenEmbedding {
    /// Whether the lookup can read `table`, decided from its description alone: a two-dimensional Q3 table, or a plain
    /// (not SD4) two-dimensional NVFP4 table whose width is a multiple of 128 and whose size is its layout's.
    pub fn check(table: &WeightRef) -> Result<(), String> {
        match (table.format, table.shape.as_slice()) {
            (WeightFormat::Q3G64Fp16, [_, _]) => Ok(()),
            (WeightFormat::Nvfp4, [vocabulary, hidden]) => {
                if table.layout == WeightLayout::RowSplitK128Sd4V1 {
                    return Err(format!(
                        "{} is an SD4 NVFP4 table, which has no embedding lookup",
                        table.name
                    ));
                }
                if table.layout != WeightLayout::RowSplitK128V1 {
                    return Err(format!(
                        "{} is an NVFP4 table in a layout the lookup cannot read",
                        table.name
                    ));
                }
                if *hidden == 0 || nvfp4::padded_k(*hidden) != *hidden {
                    return Err(format!(
                        "{} is an NVFP4 embedding of width {hidden}; it needs a width that is a multiple of 128",
                        table.name
                    ));
                }
                if nvfp4::byte_size(*vocabulary, *hidden, table.layout) != Some(table.bytes) {
                    return Err(format!(
                        "{} holds {} bytes, not those of an NVFP4 table of {vocabulary} x {hidden}",
                        table.name, table.bytes
                    ));
                }
                Ok(())
            }
            _ => Err(format!(
                "{} is not a two-dimensional Q3 or plain NVFP4 table",
                table.name
            )),
        }
    }

    /// Queues the lookup of the `rows` 32-bit token ids at `token_ids` in `table`, writing `rows x hidden` BF16 to
    /// `output`. Nothing waits.
    pub fn submit(
        &self,
        kernels: &mut dyn KernelOps,
        table: &WeightRef,
        token_ids: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<(), CudaError> {
        Self::check(table).map_err(CudaError::Argument)?;
        let (vocabulary, hidden) = (table.shape[0], table.shape[1]);
        if rows == 0 || vocabulary == 0 || hidden == 0 || table.bytes == 0 {
            return Err(CudaError::Argument(
                "an embedding lookup needs positive rows, vocabulary and width".to_string(),
            ));
        }
        let narrow = |value: u64| {
            u32::try_from(value).map_err(|_| CudaError::Argument(format!("{value} does not fit a 32-bit extent")))
        };
        if table.format == WeightFormat::Nvfp4 {
            let (rows, vocabulary, hidden) = (narrow(rows as u64)?, narrow(vocabulary)?, narrow(hidden)?);
            let (vocabulary_rows, width) = (u64::from(vocabulary), u64::from(hidden));
            let arguments = KernelArguments::new()
                .pointer(token_ids)
                .pointer(table.ptr)
                .pointer(output)
                .int32(rows)
                .int32(vocabulary)
                .int32(hidden)
                .int64(nvfp4::scale_offset(vocabulary_rows, width))
                .int64(nvfp4::trailer_offset(vocabulary_rows, width, table.layout).unwrap_or_default());
            return LinearKernel::Nvfp4Embedding
                .launch([rows, 1, 1], [128, 1, 1], 0, arguments)
                .queue(kernels);
        }
        if !hidden.is_multiple_of(64) {
            return Err(CudaError::Argument(
                "a Q3 embedding needs a width divisible by 64".to_string(),
            ));
        }
        let (rows, vocabulary, hidden) = (narrow(rows as u64)?, narrow(vocabulary)?, narrow(hidden)?);
        match table.layout {
            WeightLayout::RowSplitP2e2V1 => {
                kernels.embed_q3_p2e2(token_ids, table.ptr, output, rows, vocabulary, hidden, table.bytes)
            }
            _ => kernels.embed_q3(token_ids, table.ptr, output, rows, vocabulary, hidden, table.bytes),
        }
    }
}
