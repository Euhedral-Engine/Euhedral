//! Token embedding lookups of a Q3 table (`GroupedLinearFormat.embed`).
//!
//! The table is `[vocabulary, hidden]` in Q3 (64-value groups) in either layout: the row-split layout looks rows up
//! directly, and the P2E2 layout decodes the entropy-coded rows of the ids it is asked for. An id outside the
//! vocabulary leaves its output row as it was.

use euhedral_inference_cuda::{CudaError, DevicePtr, Submission, ops};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::model::WeightRef;

/// The Q3 token embedding lookup.
#[derive(Clone, Copy, Debug, Default)]
pub struct Q3Embedding;

impl Q3Embedding {
    /// Queues the lookup of the `rows` 32-bit token ids at `token_ids` in `table`, writing `rows x hidden` BF16 to
    /// `output`. Nothing waits.
    pub fn submit(
        &self,
        submission: &Submission<'_>,
        table: &WeightRef,
        token_ids: DevicePtr,
        output: DevicePtr,
        rows: usize,
    ) -> Result<(), CudaError> {
        let (vocabulary, hidden) = match (table.format, table.shape.as_slice()) {
            (WeightFormat::Q3G64Fp16, [vocabulary, hidden]) => (*vocabulary, *hidden),
            _ => {
                return Err(CudaError::Argument(format!(
                    "{} is not a two-dimensional Q3 table",
                    table.name
                )));
            }
        };
        if rows == 0 || vocabulary == 0 || hidden == 0 || table.bytes == 0 {
            return Err(CudaError::Argument(
                "an embedding lookup needs positive rows, vocabulary and width".to_string(),
            ));
        }
        if !hidden.is_multiple_of(64) {
            return Err(CudaError::Argument(
                "a Q3 embedding needs a width divisible by 64".to_string(),
            ));
        }
        let narrow = |value: u64| {
            u32::try_from(value).map_err(|_| CudaError::Argument(format!("{value} does not fit a 32-bit extent")))
        };
        let (rows, vocabulary, hidden) = (narrow(rows as u64)?, narrow(vocabulary)?, narrow(hidden)?);
        match table.layout {
            WeightLayout::RowSplitP2e2V1 => ops::embed_q3_p2e2(
                submission,
                token_ids,
                table.ptr,
                output,
                rows,
                vocabulary,
                hidden,
                table.bytes,
            ),
            _ => ops::embed_q3(
                submission,
                token_ids,
                table.ptr,
                output,
                rows,
                vocabulary,
                hidden,
                table.bytes,
            ),
        }
    }
}
