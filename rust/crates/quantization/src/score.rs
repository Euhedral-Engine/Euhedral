//! Scoring a candidate against a reference: the logits of an artifact, forced through the engine over the chunks of a
//! `.kld` file, reduced on the device to a KL divergence and a negative log likelihood per position, and summarised.
//!
//! - [`summary`]: the text of a score, and the statistics under it.
//! - [`device`]: the score kernel on a lane.
//! - [`Scorer`]: groups rows from every run on the lane.
//! - [`score`]: the run over a reference's chunks.
//!
//! The engine hands a row to a sink as host bytes that are valid only for the call, so a row is copied, uploaded
//! with its reference record on the scorer's own lane and reduced there.

mod chunk;
pub mod device;
mod run;
mod scorer;
pub mod summary;

pub use run::{Scored, Source, score};
pub use scorer::Scorer;

/// The chunks a run forces together: `requested`, or as many as the engine holds sequences and the reference has
/// chunks, at most [`DEFAULT_BATCH`]. A request over what the engine holds is refused.
pub fn batch_for(requested: Option<usize>, sequences: usize, chunks: usize) -> Result<usize, String> {
    match requested {
        Some(0) => Err("--batch must be at least 1".to_owned()),
        Some(batch) if batch > sequences => Err(format!(
            "--batch {batch} is more than the {sequences} sequences the engine holds"
        )),
        Some(batch) => Ok(batch.min(chunks.max(1))),
        None => Ok(DEFAULT_BATCH.min(sequences).min(chunks).max(1)),
    }
}

/// The chunks forced together when `--batch` is not given.
pub const DEFAULT_BATCH: usize = 4;

#[cfg(test)]
mod tests {
    use super::{DEFAULT_BATCH, batch_for};

    #[test]
    fn the_default_batch_is_held_by_the_chunks_and_the_sequences() {
        assert_eq!(batch_for(None, 32, 100), Ok(DEFAULT_BATCH));
        assert_eq!(batch_for(None, 2, 100), Ok(2));
        assert_eq!(batch_for(None, 32, 3), Ok(3));
        assert_eq!(batch_for(None, 1, 1), Ok(1));
        assert_eq!(batch_for(None, 32, 0), Ok(1));
    }

    #[test]
    fn a_requested_batch_is_clamped_to_the_chunks_and_refused_over_the_sequences() {
        assert_eq!(batch_for(Some(8), 32, 100), Ok(8));
        assert_eq!(batch_for(Some(8), 32, 5), Ok(5));
        assert!(batch_for(Some(33), 32, 100).is_err());
        assert!(batch_for(Some(0), 32, 100).is_err());
    }
}
