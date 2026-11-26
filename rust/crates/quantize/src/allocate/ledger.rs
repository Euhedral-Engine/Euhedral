//! The search's files: the resume state, the out directory and the log on standard error.
//!
//! The state is a [`SearchState`] on the host runtime, which hands itself to a continuation after each call; the
//! command's own thread waits for each answer, so the search sees plain calls. A call that fails loses the state (it
//! belongs to the call), and every later call is refused with that said: what was recorded is in the file.

use std::collections::HashSet;
use std::fs::{File, create_dir_all, read_dir, remove_file, rename};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use euhedral_core::frames::DummyFrame;
use euhedral_graph::Lake;
use euhedral_inference_quantization::mixed::search::{Ledger, SearchFailure};
use euhedral_inference_quantization::mixed::{Allocation, Measured, Progress, SearchError, SearchState, SearchThen};
use euhedral_io::AsyncIo;

const FRONT_DIRECTORY: &str = "allocations";

/// Starts a state call with a continuation that sends its answer here, and waits for it.
pub(super) fn blocking<T: Send + 'static>(
    start: impl FnOnce(SearchThen<T>) -> Result<(), SearchError>,
) -> Result<T, SearchFailure> {
    let (sender, receiver) = mpsc::channel();
    start(Box::new(move |answer| {
        // The receiver is gone only when the run was abandoned.
        let _ = sender.send(answer);
        Box::new(DummyFrame::INSTANCE)
    }))?;
    Ok(receiver
        .recv()
        .map_err(|_| SearchFailure::Output("the state's file calls ended without an answer".to_owned()))??)
}

/// The measured candidates of another search's state file.
pub(super) fn read_measured(io: &Arc<AsyncIo>, host: &Arc<Lake>, path: &Path) -> Result<Vec<Measured>, SearchFailure> {
    blocking(|then| SearchState::read(io, host, path, then))
}

/// The resume file of one search and its out directory.
pub(super) struct FileLedger {
    state: Option<SearchState>,
    /// What the state holds, kept here so that a lost state still lets the search read what was recorded.
    measured: Vec<Measured>,
    known: HashSet<Allocation>,
    progress: Progress,
    seed: i64,
    out: PathBuf,
}

impl FileLedger {
    /// Opens (or starts) the state in `path` for the search `fingerprint`, and creates `out`. A state of another
    /// search is refused.
    pub(super) fn open(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        fingerprint: &str,
        seed: i64,
        free_bytes: u64,
        out: &Path,
    ) -> Result<FileLedger, SearchFailure> {
        create_dir_all(out.join(FRONT_DIRECTORY))
            .map_err(|error| SearchFailure::Output(format!("{}: {error}", out.display())))?;
        let state = blocking(|then| SearchState::open(io, host, path, fingerprint, seed, free_bytes, then))?;
        Ok(FileLedger {
            measured: state.measured().to_vec(),
            known: state.measured().iter().map(|each| each.allocation.clone()).collect(),
            progress: state.progress(),
            seed: state.seed(),
            state: Some(state),
            out: out.to_path_buf(),
        })
    }

    /// The free bytes the state was started with, so a resumed search plans against what it started with.
    pub(super) fn free_bytes(&self) -> Option<u64> {
        self.state.as_ref().map(SearchState::free_bytes)
    }

    fn take(&mut self) -> Result<SearchState, SearchFailure> {
        self.state.take().ok_or_else(|| {
            SearchFailure::Output(
                "the search state was lost when an earlier call to its file failed; run again to resume from the file"
                    .to_owned(),
            )
        })
    }
}

impl Ledger for FileLedger {
    fn seed(&self) -> i64 {
        self.seed
    }

    fn measured(&self) -> &[Measured] {
        &self.measured
    }

    fn contains(&self, allocation: &Allocation) -> bool {
        self.known.contains(allocation)
    }

    fn progress(&self) -> Progress {
        self.progress
    }

    fn add(&mut self, candidate: Measured) -> Result<bool, SearchFailure> {
        let state = self.take()?;
        let kept = candidate.clone();
        let (state, added) = blocking(|then| state.add(candidate, then))?;
        self.state = Some(state);
        if added {
            self.known.insert(kept.allocation.clone());
            self.measured.push(kept);
        }
        Ok(added)
    }

    fn set_progress(&mut self, progress: Progress) -> Result<(), SearchFailure> {
        let state = self.take()?;
        self.state = Some(blocking(|then| state.set_progress(progress, then))?);
        self.progress = progress;
        Ok(())
    }

    fn publish(&mut self, files: &[(String, String)]) -> Result<(), SearchFailure> {
        publish(&self.out, files).map_err(SearchFailure::Output)
    }

    #[allow(clippy::print_stderr, reason = "the command's progress, one line a step")]
    fn note(&mut self, line: &str) {
        eprintln!("{line}");
    }
}

/// Writes each file below `out` through a `.part` file renamed into place, then removes the front's allocation files
/// the new front does not name, so the directory holds the front and nothing older.
fn publish(out: &Path, files: &[(String, String)]) -> Result<(), String> {
    for (name, text) in files {
        let target = out.join(name);
        let mut part = target.clone().into_os_string();
        part.push(".part");
        let part = PathBuf::from(part);
        let written = File::create(&part)
            .and_then(|mut file| {
                file.write_all(text.as_bytes())?;
                file.sync_all()
            })
            .and_then(|()| rename(&part, &target));
        written.map_err(|error| {
            // The failure being reported is the write's; a part that cannot be removed is left for the next run.
            let _ = remove_file(&part);
            format!("{}: {error}", target.display())
        })?;
    }
    let kept: HashSet<&str> = files.iter().map(|(name, _)| name.as_str()).collect();
    let directory = out.join(FRONT_DIRECTORY);
    let entries = read_dir(&directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries.flatten() {
        let name = format!("{FRONT_DIRECTORY}/{}", entry.file_name().to_string_lossy());
        if name.ends_with(".json") && !kept.contains(name.as_str()) {
            remove_file(entry.path()).map_err(|error| format!("{}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{read_to_string, write};

    use euhedral_core::generics::LatticeSource;
    use euhedral_inference_quantization::mixed::QuantOption;
    use euhedral_inference_settings::load_for_tool;

    use super::*;
    use crate::rig::Rig;

    fn rig() -> Rig {
        Rig::start(&load_for_tool(&[], &|_| None).unwrap()).unwrap()
    }

    fn measured(name: &str, kl: f64) -> Measured {
        Measured {
            allocation: Allocation::new(BTreeMap::from([(name.to_owned(), QuantOption::Nvfp4)])),
            kl,
            kl_se: 0.01,
            bytes: 7,
            prefill: 1.5,
            decode: 2.5,
            fits: true,
            round: 0,
            origin: "seed:pure".to_owned(),
        }
    }

    fn open(rig: &Rig, directory: &Path, fingerprint: &str) -> Result<FileLedger, SearchFailure> {
        FileLedger::open(
            &rig.io,
            &rig.host,
            &directory.join("search.resume"),
            fingerprint,
            9,
            1 << 30,
            &directory.join("out"),
        )
    }

    #[test]
    fn what_a_ledger_adds_and_sets_is_there_when_the_file_is_opened_again() {
        let (rig, directory) = (rig(), tempfile::tempdir().unwrap());
        let mut ledger = open(&rig, directory.path(), "one").unwrap();
        assert_eq!((ledger.seed(), ledger.free_bytes()), (9, Some(1 << 30)));
        assert!(ledger.add(measured("a", 0.5)).unwrap());
        assert!(
            !ledger.add(measured("a", 0.9)).unwrap(),
            "an allocation is recorded once"
        );
        assert!(ledger.add(measured("b", 0.25)).unwrap());
        let progress = Progress {
            round: 3,
            explore: 0.4,
            stale: 1,
            held_out: 0.06,
        };
        ledger.set_progress(progress).unwrap();
        assert!(ledger.contains(&measured("a", 0.0).allocation));
        assert_eq!(ledger.progress(), progress);
        drop(ledger);

        let again = open(&rig, directory.path(), "one").unwrap();
        let kls: Vec<f64> = again.measured().iter().map(|each| each.kl).collect();
        assert_eq!(kls, [0.5, 0.25], "the first record of an allocation stands");
        assert_eq!(again.progress(), progress);
        assert!(again.contains(&measured("b", 0.0).allocation));
        assert!(
            open(&rig, directory.path(), "another").is_err(),
            "a state of another search is refused"
        );
        rig.stop();
    }

    #[test]
    fn a_ledger_that_lost_its_state_refuses_writes_and_still_reads() {
        let (rig, directory) = (rig(), tempfile::tempdir().unwrap());
        let mut ledger = open(&rig, directory.path(), "one").unwrap();
        ledger.add(measured("a", 0.5)).unwrap();
        // With the asynchronous I/O ended, the next write to the file fails.
        rig.io.complete();
        let failed = ledger.add(measured("b", 0.1)).unwrap_err().to_string();
        assert!(
            !failed.contains("lost"),
            "the failed write is reported as itself: {failed}"
        );
        for refusal in [
            ledger.add(measured("b", 0.1)).map(drop),
            ledger.set_progress(Progress::default()),
        ] {
            let message = refusal.unwrap_err().to_string();
            assert!(message.contains("lost"), "{message}");
        }
        assert_eq!(ledger.measured().len(), 1, "what was recorded can still be read");
        assert!(!ledger.contains(&measured("b", 0.0).allocation));
        rig.stop();
    }

    fn files(names: &[&str]) -> Vec<(String, String)> {
        names
            .iter()
            .map(|name| ((*name).to_owned(), format!("text of {name}")))
            .collect()
    }

    #[test]
    fn a_published_front_replaces_the_earlier_one_whole() {
        let directory = tempfile::tempdir().unwrap();
        create_dir_all(directory.path().join(FRONT_DIRECTORY)).unwrap();
        publish(
            directory.path(),
            &files(&["front.json", "allocations/front-00.json", "allocations/front-01.json"]),
        )
        .unwrap();
        publish(directory.path(), &files(&["front.json", "allocations/front-00.json"])).unwrap();
        let held: Vec<_> = read_dir(directory.path().join(FRONT_DIRECTORY)).unwrap().collect();
        assert_eq!(held.len(), 1, "the point that left the front has no file");
        assert_eq!(
            read_to_string(directory.path().join("allocations/front-00.json")).unwrap(),
            "text of allocations/front-00.json"
        );
        assert!(!directory.path().join("front.json.part").exists());
    }

    #[test]
    fn a_publish_that_cannot_write_names_the_file_and_leaves_no_part() {
        let directory = tempfile::tempdir().unwrap();
        // The target is a directory: the rename onto it fails after the part was written.
        create_dir_all(directory.path().join("front.json")).unwrap();
        write(directory.path().join("front.json").join("keep"), b"earlier").unwrap();
        let refusal = publish(directory.path(), &files(&["front.json"])).unwrap_err();
        assert!(refusal.contains("front.json"), "{refusal}");
        assert!(!directory.path().join("front.json.part").exists());
    }
}
