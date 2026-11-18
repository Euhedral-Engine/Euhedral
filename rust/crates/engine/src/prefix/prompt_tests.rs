//! The request-side protocol over the cache, driven the way the sequence's owner drives it: what a request restores,
//! what it captures, and when its next chunk may start.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::cache_tests::{CHUNK, Device, FORMAT, Rig, answer, same_state};
use super::fixtures::tokens;
use super::{Ended, Hit, NodeId, PromptCheckpoints, Resolve, RestoreError, Restored};
use crate::state::SequenceId;

const ROWS: u64 = 8192;

/// What one simulated request saw.
struct Seen {
    /// The restored position.
    cached: u32,
    /// The ends of the checkpoints captured.
    captured: Vec<u32>,
}

/// Starts a request: asks for the lookup and reports the hit, if any.
fn begin(rig: &Rig, prompt: &[u32]) -> (PromptCheckpoints, Option<Hit>) {
    let slot: Arc<Mutex<Option<Option<Hit>>>> = Arc::new(Mutex::new(None));
    let told = Arc::clone(&slot);
    let mut checkpoints = PromptCheckpoints::begin(rig.cache.clone(), prompt.to_vec().into(), move |hit| {
        *told.lock().unwrap() = Some(hit);
    });
    assert!(checkpoints.holds(), "the first chunk waits for the lookup");
    assert_eq!(checkpoints.pending(), 1);
    let hit = answer(&slot);
    checkpoints.looked();
    assert_eq!(checkpoints.pending(), 0);
    (checkpoints, hit)
}

/// Runs a whole request the way the owner does, returning what it restored and captured. `device` holds the
/// sequence's state; each checkpoint's chunk leaves it filled (the same bytes each time, so the pages a later checkpoint
/// shares with an earlier one agree).
fn serve(rig: &Rig, device: &Arc<Device>, prompt: &[u32]) -> Seen {
    let (mut checkpoints, hit) = begin(rig, prompt);
    let len = u32::try_from(prompt.len()).unwrap();
    let mut offset = 0;
    if let Some(hit) = hit {
        let slot = Arc::new(Mutex::new(None));
        let told = Arc::clone(&slot);
        checkpoints.restore(
            hit,
            SequenceId(1),
            Arc::clone(device) as Arc<dyn Resolve>,
            |_| rig.submit(),
            move |result| *told.lock().unwrap() = Some(result),
        );
        assert!(checkpoints.holds(), "the first chunk waits for the restore's job");
        let job = rig.next_job();
        checkpoints.restore_submitted();
        assert!(!checkpoints.holds(), "and starts once the job is admitted, behind it");
        assert_eq!(
            checkpoints.pending(),
            1,
            "the sequence is not given back before the answer"
        );
        rig.run(job, device);
        match checkpoints.restored(answer(&slot)) {
            Restored::Resumed(position) => offset = position,
            other => panic!("{other:?}"),
        }
        assert_eq!(checkpoints.pending(), 0);
    }
    let mut captured = Vec::new();
    while offset < len {
        assert!(!checkpoints.holds(), "a chunk is admitted");
        let end = (offset + CHUNK).min(len);
        if checkpoints.wants(end as usize) {
            device.fill(&rig.layout, end, 5);
            checkpoints.expect(end as usize, Arc::clone(device) as Arc<dyn Resolve>);
            assert!(
                checkpoints.holds(),
                "nothing starts behind a chunk whose state is to be captured"
            );
            checkpoints.concluded();
            assert!(checkpoints.holds(), "nor before the capture is asked");
            let node = Arc::new(Mutex::new(None));
            let told = Arc::clone(&node);
            assert!(checkpoints.ask(
                SequenceId(1),
                |_| rig.submit(),
                move |n| *told.lock().unwrap() = Some(n)
            ));
            assert!(checkpoints.holds(), "nor before the capture's job is admitted");
            let job = rig.next_job();
            checkpoints.capture_submitted();
            assert!(!checkpoints.holds(), "the next chunk starts behind the job");
            rig.run(job, device);
            checkpoints.captured(answer(&node));
            captured.push(end);
        }
        offset = end;
    }
    assert_eq!(checkpoints.pending(), 0);
    Seen {
        cached: checkpoints.cached(),
        captured,
    }
}

fn rig() -> (Rig, Arc<Device>) {
    (Rig::new(64), Device::new(FORMAT, ROWS))
}

#[test]
fn a_prompt_with_a_stored_prefix_resumes_after_it() {
    let (rig, device) = rig();
    let prompt = tokens(0, 2600);
    let first = serve(&rig, &device, &prompt);
    assert_eq!((first.cached, first.captured), (0, vec![1024, 2048, 2560]));
    let other = Device::new(FORMAT, ROWS);
    let second = serve(&rig, &other, &prompt);
    assert_eq!(second.cached, 2560, "the longest checkpoint");
    assert!(second.captured.is_empty(), "nothing new to store");
    assert!(
        same_state(&rig.layout, 2560, &device, &other),
        "the state is the checkpoint's, byte for byte"
    );
}

#[test]
fn a_prompt_that_diverges_reuses_the_shared_checkpoint() {
    let (rig, device) = rig();
    serve(&rig, &device, &tokens(0, 2600));
    let mut other = tokens(0, 2100);
    other.extend(tokens(10_000, 500));
    let second = serve(&rig, &Device::new(FORMAT, ROWS), &other);
    assert_eq!(second.cached, 2048);
    assert_eq!(
        second.captured,
        vec![2560],
        "the new tail's own checkpoint hangs under the restored one"
    );
}

#[test]
fn a_prompt_that_shares_nothing_prefills_cold() {
    let (rig, device) = rig();
    serve(&rig, &device, &tokens(0, 2600));
    let second = serve(&rig, &Device::new(FORMAT, ROWS), &tokens(50_000, 2600));
    assert_eq!(second.cached, 0);
}

#[test]
fn an_identical_prompt_still_prefills_its_last_tokens() {
    let (rig, device) = rig();
    let prompt = tokens(0, 1024);
    let first = serve(&rig, &device, &prompt);
    assert_eq!(
        first.captured,
        vec![512],
        "a checkpoint at the prompt's end would never be restored"
    );
    let second = serve(&rig, &Device::new(FORMAT, ROWS), &prompt);
    assert_eq!(
        second.cached, 512,
        "the checkpoint at the prompt's end would leave nothing to prefill"
    );
}

#[test]
fn a_short_prompt_stores_its_last_chunk_boundary() {
    let (rig, device) = rig();
    let first = serve(&rig, &device, &tokens(0, 900));
    assert_eq!(first.captured, vec![512]);
    let second = serve(&rig, &Device::new(FORMAT, ROWS), &tokens(0, 900));
    assert_eq!(second.cached, 512);
}

#[test]
fn a_chunk_that_never_concluded_is_never_captured() {
    let (rig, device) = rig();
    let (mut checkpoints, _) = begin(&rig, &tokens(0, 2600));
    checkpoints.expect(1024, Arc::clone(&device) as Arc<dyn Resolve>);
    assert!(checkpoints.holds());
    assert!(
        !checkpoints.ask(SequenceId(1), |_| rig.submit(), |_| {}),
        "the chunk is still in flight"
    );
    checkpoints.stop();
    assert!(
        !checkpoints.ask(SequenceId(1), |_| rig.submit(), |_| {}),
        "a stopped request asks for nothing"
    );
    assert_eq!(rig.queued(), 0);
    assert_eq!(checkpoints.pending(), 0);
    assert!(!checkpoints.holds(), "nothing is waited for: the request ends");
}

#[test]
fn the_next_capture_waits_for_the_answer_of_the_one_before() {
    let (rig, device) = rig();
    let (mut checkpoints, _) = begin(&rig, &tokens(0, 2600));
    let node: Arc<Mutex<Option<NodeId>>> = Arc::new(Mutex::new(None));
    let told = Arc::clone(&node);
    checkpoints.expect(1024, Arc::clone(&device) as Arc<dyn Resolve>);
    checkpoints.concluded();
    assert!(checkpoints.ask(
        SequenceId(1),
        |_| rig.submit(),
        move |n| *told.lock().unwrap() = Some(n)
    ));
    checkpoints.capture_submitted();
    checkpoints.expect(2048, Arc::clone(&device) as Arc<dyn Resolve>);
    checkpoints.concluded();
    assert!(
        !checkpoints.ask(SequenceId(1), |_| rig.submit(), |_| {}),
        "its parent is the node the first answers with"
    );
    assert_eq!(checkpoints.pending(), 1);
    let job = rig.next_job();
    rig.run(job, &device);
    checkpoints.captured(answer(&node));
    assert!(checkpoints.ask(SequenceId(1), |_| rig.submit(), |_| {}));
}

#[test]
fn a_capture_that_fails_publishes_nothing_and_is_answered() {
    for how in 0..2 {
        let (rig, device) = rig();
        let prompt = tokens(0, 1100);
        let (mut checkpoints, _) = begin(&rig, &prompt);
        checkpoints.expect(1024, Arc::clone(&device) as Arc<dyn Resolve>);
        checkpoints.concluded();
        let node = Arc::new(Mutex::new(None));
        let told = Arc::clone(&node);
        checkpoints.ask(
            SequenceId(1),
            |_| rig.submit(),
            move |n| *told.lock().unwrap() = Some(n),
        );
        checkpoints.capture_submitted();
        let mut job = rig.next_job();
        if how == 0 {
            job.finish(Ended::Failed("the lane failed".to_string()));
        } else {
            drop(job);
        }
        assert_eq!(checkpoints.pending(), 1);
        checkpoints.captured(answer(&node));
        assert_eq!(checkpoints.pending(), 0);
        assert!(rig.lookup(&prompt, None).is_none(), "case {how}: nothing was published");
    }
}

#[test]
fn a_refused_restore_goes_cold_and_asks_nothing_of_the_device() {
    let (rig, device) = rig();
    serve(&rig, &device, &tokens(0, 1100));
    let short = Device::new(FORMAT, 100);
    let (mut checkpoints, hit) = begin(&rig, &tokens(0, 1100));
    let slot = Arc::new(Mutex::new(None));
    let told = Arc::clone(&slot);
    checkpoints.restore(
        hit.expect("a hit"),
        SequenceId(1),
        short as Arc<dyn Resolve>,
        |_| rig.submit(),
        move |result| *told.lock().unwrap() = Some(result),
    );
    assert_eq!(rig.queued(), 0, "a refusal submits no job");
    assert!(matches!(
        checkpoints.restored(answer(&slot)),
        Restored::Refused(RestoreError::Unreserved { .. })
    ));
    assert_eq!(checkpoints.pending(), 0);
    assert_eq!(checkpoints.cached(), 0);
    assert!(!checkpoints.holds(), "the first chunk starts from the prompt's start");
}

#[test]
fn a_restore_whose_copy_fails_is_reported_failed_and_kills_the_chunk_behind_it() {
    let (rig, device) = rig();
    serve(&rig, &device, &tokens(0, 1100));
    let (mut checkpoints, hit) = begin(&rig, &tokens(0, 1100));
    let slot = Arc::new(Mutex::new(None));
    let told = Arc::clone(&slot);
    let kill = checkpoints.restore(
        hit.expect("a hit"),
        SequenceId(1),
        Device::new(FORMAT, ROWS) as Arc<dyn Resolve>,
        |_| rig.submit(),
        move |result| *told.lock().unwrap() = Some(result),
    );
    let mut job = rig.next_job();
    checkpoints.restore_submitted();
    job.finish(Ended::Failed("the lane failed".to_string()));
    assert!(kill.load(Ordering::Acquire), "the first chunk shares the flag");
    assert!(matches!(
        checkpoints.restored(answer(&slot)),
        Restored::Failed(RestoreError::Copy(_))
    ));
}

#[test]
fn a_cancel_while_the_restore_runs_is_answered_cancelled_and_the_request_waits_for_it() {
    let (rig, device) = rig();
    serve(&rig, &device, &tokens(0, 1100));
    let (mut checkpoints, hit) = begin(&rig, &tokens(0, 1100));
    let slot = Arc::new(Mutex::new(None));
    let told = Arc::clone(&slot);
    let kill = checkpoints.restore(
        hit.expect("a hit"),
        SequenceId(1),
        Device::new(FORMAT, ROWS) as Arc<dyn Resolve>,
        |_| rig.submit(),
        move |result| *told.lock().unwrap() = Some(result),
    );
    let mut job = rig.next_job();
    checkpoints.restore_submitted();
    checkpoints.stop();
    assert!(kill.load(Ordering::Acquire));
    assert_eq!(
        checkpoints.pending(),
        1,
        "the sequence is not given back while the copy may run"
    );
    job.finish(Ended::Cancelled);
    assert!(matches!(checkpoints.restored(answer(&slot)), Restored::Cancelled));
    assert_eq!(checkpoints.pending(), 0);
}

#[test]
fn a_job_lets_go_of_the_sequences_references_before_the_cache_answers() {
    for how in 0..3 {
        let (rig, device) = rig();
        serve(&rig, &device, &tokens(0, 1100));
        let target = Device::new(FORMAT, ROWS);
        let (mut checkpoints, hit) = begin(&rig, &tokens(0, 1100));
        let held = Arc::new(Mutex::new(None));
        let told = Arc::clone(&held);
        let watched = Arc::clone(&target);
        checkpoints.restore(
            hit.expect("a hit"),
            SequenceId(1),
            Arc::clone(&target) as Arc<dyn Resolve>,
            |_| rig.submit(),
            move |_| *told.lock().unwrap() = Some(Arc::strong_count(&watched)),
        );
        let mut job = rig.next_job();
        match how {
            0 => job.finish(Ended::Done),
            1 => job.finish(Ended::Failed("the lane failed".to_string())),
            _ => drop(job),
        }
        // The test's own handle and the one the answer's closure watches: the job's is gone.
        assert_eq!(
            answer(&held),
            2,
            "case {how}: the job still holds the sequence's references"
        );
        assert_eq!(Arc::strong_count(&target), 1);
    }
}
