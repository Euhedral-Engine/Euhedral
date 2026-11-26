//! What a run of the whole model writes after its layers, in Java's order, over the final hidden states:
//! 1. the Hessian of the final hidden states under the final norm (the output head's);
//! 2. the MTP block: its input projection, then the states it makes (which replace the hidden states), then the MTP
//!    layer, then the Hessian of the MTP output (the draft head's). `--skip-mtp` leaves it out, and the manifest
//!    records that, so that the run is complete without it;
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
                skipped(heads, done);
            } else {
                mtp(heads, done);
            }
        });
    }));
}

/// Records that the MTP block is left out, then goes on to the output head.
fn skipped<H: Heads>(heads: Arc<H>, done: Then<()>) {
    let sealing = Arc::clone(&heads);
    sealing.seal(
        Part::MtpSkipped,
        Vec::new(),
        Box::new(move |sealed| {
            onward(sealed, done, move |(), done| {
                output_head(heads, DraftHessian::OutputHead, done)
            });
        }),
    );
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

/// What a trellis run adds to the heads: its MTP block is the trellis writer's (the [`Heads::mtp`] call rounds and
/// writes it, and moves no hidden states), and the checkpoint's closing files are written once the heads are rounded.
pub trait TrellisHeads: Heads {
    /// Writes the closing files of the checkpoint (the embedding, the final norm and head, the vision tower, the
    /// copied files, the index and the configs); gives their names, which are saved once it ends.
    fn close_checkpoint(&self, then: Then<Vec<String>>);
}

/// Seals `part` with `files`, then goes on to `next`.
fn sealed_then<H: Heads>(
    heads: Arc<H>,
    part: Part,
    files: Vec<String>,
    done: Then<()>,
    next: impl FnOnce(Arc<H>, Then<()>) + Send + 'static,
) {
    let sealing = Arc::clone(&heads);
    sealing.seal(
        part,
        files,
        Box::new(move |sealed| onward(sealed, done, move |(), done| next(heads, done))),
    );
}

/// The heads of a trellis run, in Java's order: the MTP block (written by the trellis writer, whatever `--skip-mtp`
/// says: the checkpoint carries it), the output head's Hessian, the output head and the draft head, both under that
/// Hessian, then the checkpoint's closing files, and last the resume file's dropping.
///
/// The draft head is sealed after the closing files, so that the manifest never counts the heads written before the
/// checkpoint is whole: a run that fails in between resumes at the heads, and rounds them again.
pub fn run_trellis<H: TrellisHeads>(heads: Arc<H>, done: Then<()>) {
    let first = Arc::clone(&heads);
    first.mtp(Box::new(move |files| {
        onward(files, done, move |files, done| {
            sealed_then(heads, Part::Mtp, files, done, trellis_observe);
        });
    }));
}

fn trellis_observe<H: TrellisHeads>(heads: Arc<H>, done: Then<()>) {
    let observing = Arc::clone(&heads);
    observing.observe(Box::new(move |observed| {
        onward(observed, done, move |(), done| {
            let rounding = Arc::clone(&heads);
            rounding.output_head(Box::new(move |files| {
                onward(files, done, move |files, done| {
                    sealed_then(heads, Part::OutputHead, files, done, trellis_draft);
                });
            }));
        });
    }));
}

fn trellis_draft<H: TrellisHeads>(heads: Arc<H>, done: Then<()>) {
    let rounding = Arc::clone(&heads);
    rounding.draft_head(
        DraftHessian::OutputHead,
        Box::new(move |drafted| {
            onward(drafted, done, move |mut files, done| {
                let closing = Arc::clone(&heads);
                closing.close_checkpoint(Box::new(move |closed| {
                    onward(closed, done, move |closed, done| {
                        files.extend(closed);
                        sealed_then(heads, Part::DraftHead, files, done, |heads, done| {
                            heads.drop_checkpoint(done)
                        });
                    });
                }));
            });
        }),
    );
}
