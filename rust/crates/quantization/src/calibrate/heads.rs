//! What a run of the whole model writes after its layers, in Java's order, over the final hidden states:
//! 1. the Hessian of the final hidden states under the final norm (the output head's);
//! 2. the MTP block, unless `--skip-mtp`: its input projection, then the states it makes (which replace the hidden
//!    states), then the MTP layer, then the Hessian of the MTP output (the draft head's);
//! 3. the output head;
//! 4. the draft head, under the MTP output's Hessian, or the output head's when the MTP block is skipped.
//!
//! The MTP layer reads the input projection's output, so the projection comes first, as in Java's code. Each part's
//! files are listed in the manifest as soon as the part is sealed, and the resume file is dropped only after the draft
//! head, the last part, is listed: a part that fails leaves the resume file, whose states are the layers' output, so
//! the next run starts at these steps again. Java drops its resume file before them and loses that point.
//!
//! The calls on the card are [`Heads`]; this sequence is generic over them and never waits, as [`super::driver`].

use std::sync::Arc;

use super::CalibrateError;
use super::driver::Then;
use super::manifest::Part;

/// Which Hessian the draft head is rounded under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftHessian {
    /// The MTP output's, which the MTP block's rounding made.
    Mtp,
    /// The output head's: the MTP block was skipped.
    OutputHead,
}

/// The calls of the heads. Each takes the continuation to run once its work, which may run on other threads, is over;
/// none waits. A part's call gives the names of the files it saved, and lists nothing.
pub trait Heads: Send + Sync + 'static {
    /// The Hessian of the final hidden states under the final norm, kept for the output head.
    fn observe(&self, then: Then<()>);

    /// The MTP block rounded: the hidden states become the MTP output and the draft head's Hessian is made.
    fn mtp(&self, then: Then<Vec<String>>);

    /// The output head rounded under the Hessian [`observe`](Self::observe) made.
    fn output_head(&self, then: Then<Vec<String>>);

    /// The draft head rounded under `hessian`; the heads' device state is freed once it is.
    fn draft_head(&self, hessian: DraftHessian, then: Then<Vec<String>>);

    /// Lists `files` as saved and marks `part` written, in the manifest on disk.
    fn seal(&self, part: Part, files: Vec<String>, then: Then<()>);

    /// Unlists the resume file in the manifest on disk, then removes the file.
    fn drop_checkpoint(&self, then: Then<()>);
}

/// Continues with `next` on success, else ends the sequence with the failure.
fn onward<T>(result: Result<T, CalibrateError>, done: Then<()>, next: impl FnOnce(T, Then<()>)) {
    match result {
        Ok(value) => next(value, done),
        Err(error) => done(Err(error)),
    }
}

/// Rounds the heads in order and drops the resume file; `done` gets the first failure, or Ok once the resume file is
/// gone.
pub fn run<H: Heads>(heads: Arc<H>, skip_mtp: bool, done: Then<()>) {
    let observing = Arc::clone(&heads);
    observing.observe(Box::new(move |observed| {
        onward(observed, done, move |(), done| {
            if skip_mtp {
                output_head(heads, DraftHessian::OutputHead, done);
            } else {
                mtp(heads, done);
            }
        });
    }));
}

fn mtp<H: Heads>(heads: Arc<H>, done: Then<()>) {
    let rounding = Arc::clone(&heads);
    rounding.mtp(Box::new(move |files| {
        onward(files, done, move |files, done| {
            let sealing = Arc::clone(&heads);
            sealing.seal(
                Part::Mtp,
                files,
                Box::new(move |sealed| {
                    onward(sealed, done, move |(), done| {
                        output_head(heads, DraftHessian::Mtp, done)
                    });
                }),
            );
        });
    }));
}

fn output_head<H: Heads>(heads: Arc<H>, hessian: DraftHessian, done: Then<()>) {
    let rounding = Arc::clone(&heads);
    rounding.output_head(Box::new(move |files| {
        onward(files, done, move |files, done| {
            let sealing = Arc::clone(&heads);
            sealing.seal(
                Part::OutputHead,
                files,
                Box::new(move |sealed| {
                    onward(sealed, done, move |(), done| draft_head(heads, hessian, done));
                }),
            );
        });
    }));
}

fn draft_head<H: Heads>(heads: Arc<H>, hessian: DraftHessian, done: Then<()>) {
    let rounding = Arc::clone(&heads);
    rounding.draft_head(
        hessian,
        Box::new(move |files| {
            onward(files, done, move |files, done| {
                let sealing = Arc::clone(&heads);
                sealing.seal(
                    Part::DraftHead,
                    files,
                    Box::new(move |sealed| {
                        onward(sealed, done, move |(), done| heads.drop_checkpoint(done));
                    }),
                );
            });
        }),
    );
}
