//! Objects shared by engines are loaded once and freed with the last engine that uses them; an engine starts only
//! while what it adds fits.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "peer threads and gates: a test holds a load open while another thread asks, and waits for the answer under a hang deadline that names what never happened"
)]

use std::collections::{HashMap, HashSet};
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use super::*;

const MIB: u64 = 1 << 20;
/// A hang deadline: a run that reaches it has failed, and says what never happened.
const DEADLINE: Duration = Duration::from_secs(20);

fn key(file: &str, name: &str) -> Key {
    Key {
        file: PathBuf::from(file),
        name: name.to_string(),
        offset: 0,
    }
}

/// A door a test opens: the thread that waits at it fails, naming `what`, if it stays shut past the deadline.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    fn pass(&self, what: &str) {
        let (open, timeout) = self
            .changed
            .wait_timeout_while(self.open.lock().unwrap(), DEADLINE, |open| !*open)
            .unwrap();
        assert!(*open && !timeout.timed_out(), "{what} never happened");
    }
}

/// An object in the fake's memory: counts itself while alive and tells the hook when it ends.
struct Object {
    id: u64,
    live: Arc<AtomicUsize>,
    on_end: Arc<OnceLock<Box<dyn Fn() + Send + Sync>>>,
}

impl Drop for Object {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
        if let Some(hook) = self.on_end.get() {
            hook();
        }
    }
}

/// A backend that reads nothing: it counts what it is asked to load and which objects are alive.
#[derive(Default)]
struct Fake {
    next: AtomicU64,
    reads: Mutex<Vec<String>>,
    live: Arc<AtomicUsize>,
    fail_on: Option<&'static str>,
    /// An object that waits here, once the test has been told it began.
    hold: Option<(&'static str, Arc<Gate>, Arc<Gate>)>,
    on_end: Arc<OnceLock<Box<dyn Fn() + Send + Sync>>>,
}

impl Backend for Arc<Fake> {
    type Handle = Object;

    fn load(&self, key: &Key, _bytes: u64) -> Result<Object, LoadError> {
        if let Some((name, began, release)) = &self.hold
            && key.name == *name
        {
            began.open();
            release.pass("the release of the held load");
        }
        if self.fail_on == Some(key.name.as_str()) {
            return Err(LoadError(format!("unreadable {}", key.name)));
        }
        self.reads
            .lock()
            .unwrap()
            .push(format!("{}:{}", key.file.display(), key.name));
        self.live.fetch_add(1, Ordering::AcqRel);
        Ok(Object {
            id: self.next.fetch_add(1, Ordering::AcqRel),
            live: Arc::clone(&self.live),
            on_end: Arc::clone(&self.on_end),
        })
    }
}

impl Fake {
    fn reads(&self) -> usize {
        self.reads.lock().unwrap().len()
    }
}

/// Frames that run a task where it is submitted.
struct Inline;

impl Frames for Inline {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        task();
    }
}

/// Frames that hold the tasks until the test runs them: the interleaving is the test's.
#[derive(Default)]
struct Deferred {
    tasks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
}

impl Frames for Deferred {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        self.tasks.lock().unwrap().push(task);
    }
}

impl Deferred {
    fn run_all(&self) {
        let tasks = mem::take(&mut *self.tasks.lock().unwrap());
        for task in tasks {
            task();
        }
    }
}

/// Frames of a lake that finished: the task is dropped unrun.
struct Refusing;

impl Frames for Refusing {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        drop(task);
    }
}

type Cache = WeightCache<Arc<Fake>>;

fn rig(fake: Fake) -> (Cache, Arc<Fake>) {
    rig_with(fake, Arc::new(Inline))
}

fn rig_with(fake: Fake, frees: Arc<dyn Frames>) -> (Cache, Arc<Fake>) {
    let fake = Arc::new(fake);
    (WeightCache::new(Arc::clone(&fake), frees), fake)
}

type Answer = Arc<Mutex<Option<Outcome<Arc<Fake>>>>>;

fn ask(cache: &Cache, frames: &dyn Frames, key: Key, bytes: u64) -> Answer {
    let answer: Answer = Arc::default();
    let put = Arc::clone(&answer);
    cache.acquire(key, bytes, frames, move |outcome| {
        *put.lock().unwrap() = Some(outcome);
    });
    answer
}

fn lease(answer: &Answer) -> Lease<Arc<Fake>> {
    answer
        .lock()
        .unwrap()
        .take()
        .expect("the callback was called")
        .expect("the object loaded")
}

fn answered(answer: &Answer) -> bool {
    answer.lock().unwrap().is_some()
}

/// Runs `body` on a thread of its own and fails, naming `what`, if it does not end by the deadline: a cache that held a
/// lock where it must not shows as a body that never ends, and the test thread survives to say so.
fn within_deadline(what: &str, body: impl FnOnce() + Send + 'static) {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        body();
        sender.send(()).ok();
    });
    match receiver.recv_timeout(DEADLINE) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => panic!("{what} never ended"),
        Err(RecvTimeoutError::Disconnected) => panic!("the body of {what} panicked"),
    }
}

#[test]
fn shared_objects_are_loaded_once_and_freed_with_the_last_engine_that_uses_them() {
    let (cache, fake) = rig(Fake::default());
    // Engine A reads norm, up and down from the base; B reads norm and up from the base and down from the other file.
    let norm_a = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    let up_a = lease(&ask(&cache, &Inline, key("base", "up"), 128));
    let down_a = lease(&ask(&cache, &Inline, key("base", "down"), 256));
    let norm_b = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    let up_b = lease(&ask(&cache, &Inline, key("base", "up"), 128));
    let down_b = lease(&ask(&cache, &Inline, key("other", "down"), 256));

    assert_eq!(cache.loads(), 4, "norm and up once, down once per source");
    assert_eq!(fake.reads(), 4);
    assert!(ptr::eq(norm_a.handle(), norm_b.handle()));
    assert!(ptr::eq(up_a.handle(), up_b.handle()));
    assert_ne!(down_a.handle().id, down_b.handle().id);
    assert_eq!(cache.resident_bytes(), 64 + 128 + 256 + 256);

    // A ends: only its own down goes.
    drop((norm_a, up_a, down_a));
    assert_eq!(fake.live.load(Ordering::Acquire), 3);
    assert!(!cache.contains(&key("base", "down")));
    assert!(cache.contains(&key("other", "down")));
    assert_eq!(cache.resident_bytes(), 64 + 128 + 256);

    // B ends: nothing is left.
    drop((norm_b, up_b, down_b));
    assert_eq!(fake.live.load(Ordering::Acquire), 0);
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.resident_bytes(), 0);
}

#[test]
fn a_second_engine_shares_the_copy_and_reads_nothing() {
    let (cache, fake) = rig(Fake::default());
    let first = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    assert_eq!(fake.reads(), 1);
    let second = ask(&cache, &Refusing, key("base", "norm"), 64);
    // Answered at once, with the same object, though no frame could have loaded another.
    assert!(answered(&second));
    assert!(ptr::eq(first.handle(), lease(&second).handle()));
    assert_eq!(fake.reads(), 1);
    assert_eq!(cache.resident_bytes(), 64);
}

#[test]
fn an_engine_that_asks_while_the_load_is_in_flight_shares_it() {
    let (cache, fake) = rig(Fake::default());
    let frames = Deferred::default();
    let first = ask(&cache, &frames, key("base", "norm"), 64);
    let second = ask(&cache, &frames, key("base", "norm"), 64);
    assert!(!answered(&first) && !answered(&second));
    assert_eq!(frames.tasks.lock().unwrap().len(), 1, "one load for both");
    assert_eq!(cache.resident_bytes(), 64, "the loading object is counted");
    frames.run_all();
    assert!(ptr::eq(lease(&first).handle(), lease(&second).handle()));
    assert_eq!(fake.reads(), 1);
}

#[test]
fn the_object_is_freed_when_the_last_engine_drops_it_and_loaded_again_after() {
    let (cache, fake) = rig(Fake::default());
    let first = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    let second = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    drop(first);
    assert_eq!(fake.live.load(Ordering::Acquire), 1, "one engine still uses it");
    assert!(cache.contains(&key("base", "norm")));
    drop(second);
    assert_eq!(fake.live.load(Ordering::Acquire), 0);
    assert!(!cache.contains(&key("base", "norm")));
    assert_eq!(cache.resident_bytes(), 0);

    let again = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    assert_eq!(cache.loads(), 2);
    assert_eq!(fake.live.load(Ordering::Acquire), 1);
    drop(again);
}

#[test]
fn a_failed_load_fails_every_holder_and_is_not_kept() {
    let (cache, fake) = rig(Fake {
        fail_on: Some("norm"),
        ..Fake::default()
    });
    let frames = Deferred::default();
    let first = ask(&cache, &frames, key("base", "norm"), 64);
    let second = ask(&cache, &frames, key("base", "norm"), 64);
    let up = ask(&cache, &frames, key("base", "up"), 128);
    frames.run_all();
    for answer in [&first, &second] {
        let error = answer.lock().unwrap().take().unwrap().unwrap_err();
        assert_eq!(error.0, "unreadable norm");
    }
    let up = lease(&up);
    assert_eq!(cache.size(), 1, "the failed object is not held");
    assert_eq!(cache.resident_bytes(), 128);
    // The next engine to ask tries again.
    let retry = ask(&cache, &Inline, key("base", "norm"), 64);
    assert!(retry.lock().unwrap().take().unwrap().is_err());
    drop(up);
    assert_eq!(fake.live.load(Ordering::Acquire), 0);
    assert_eq!(cache.resident_bytes(), 0);
}

#[test]
fn a_load_no_frame_runs_fails_the_engines_waiting_for_it() {
    let (cache, _) = rig(Fake::default());
    let first = ask(&cache, &Refusing, key("base", "norm"), 64);
    assert!(first.lock().unwrap().take().unwrap().is_err());
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.resident_bytes(), 0);
}

#[test]
fn no_lock_is_held_while_an_object_is_read() {
    let (began, release) = (Arc::new(Gate::default()), Arc::new(Gate::default()));
    let (cache, fake) = rig(Fake {
        hold: Some(("slow", Arc::clone(&began), Arc::clone(&release))),
        ..Fake::default()
    });
    let cache = Arc::new(cache);
    let slow: Answer = Arc::default();
    let (reader, put) = (Arc::clone(&cache), Arc::clone(&slow));
    let (finished, ended) = mpsc::channel();
    thread::spawn(move || {
        reader.acquire(key("base", "slow"), 512, &Inline, move |outcome| {
            *put.lock().unwrap() = Some(outcome);
        });
        finished.send(()).ok();
    });
    began.pass("the start of the held read");

    // While that object is being read: the table answers, another object loads, and an engine joins the held one.
    let (other, joined): (Answer, Answer) = (Arc::default(), Arc::default());
    let (probe, put_other, put_joined) = (Arc::clone(&cache), Arc::clone(&other), Arc::clone(&joined));
    within_deadline("the cache's answers while an object is read", move || {
        assert!(probe.contains(&key("base", "slow")));
        assert_eq!(probe.resident_bytes(), 512);
        probe.acquire(key("base", "fast"), 64, &Inline, move |outcome| {
            *put_other.lock().unwrap() = Some(outcome);
        });
        probe.acquire(key("base", "slow"), 512, &Inline, move |outcome| {
            *put_joined.lock().unwrap() = Some(outcome);
        });
        assert_eq!(probe.size(), 2);
    });
    assert!(answered(&other), "another object loads beside the held read");
    assert!(!answered(&joined) && !answered(&slow));

    release.open();
    ended.recv_timeout(DEADLINE).expect("the held read ended");
    let (slow, joined) = (lease(&slow), lease(&joined));
    assert!(ptr::eq(slow.handle(), joined.handle()));
    assert_eq!(fake.reads(), 2);
}

#[test]
fn no_lock_is_held_while_an_object_is_freed() {
    let (cache, fake) = rig(Fake::default());
    let cache = Arc::new(cache);
    // When the object ends, the engine that held it asks for it again: the cache must answer, and the new object
    // must outlive the end of the old one.
    let again: Answer = Arc::default();
    let (cache_again, put) = (Arc::clone(&cache), Arc::clone(&again));
    let asked = AtomicBool::new(false);
    let hook: Box<dyn Fn() + Send + Sync> = Box::new(move || {
        // Only the first object's end asks again.
        if asked.swap(true, Ordering::AcqRel) {
            return;
        }
        let put = Arc::clone(&put);
        cache_again.acquire(key("base", "norm"), 64, &Inline, move |outcome| {
            *put.lock().unwrap() = Some(outcome);
        });
    });
    let first = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    assert!(fake.on_end.set(hook).is_ok());
    within_deadline("the end of an object that is asked for again", move || drop(first));
    assert_eq!(cache.loads(), 2);
    assert!(cache.contains(&key("base", "norm")));
    assert_eq!(cache.size(), 1);
    assert_eq!(cache.resident_bytes(), 64);
    assert_eq!(
        fake.live.load(Ordering::Acquire),
        1,
        "the old object is gone and the new one is live"
    );
    assert!(answered(&again));
}

fn need(other: u64, objects: &[(&str, u64)]) -> Need {
    Need {
        objects: objects.iter().map(|(name, mib)| (key("f", name), mib * MIB)).collect(),
        other_bytes: other * MIB,
    }
}

#[test]
fn an_engine_adds_its_other_bytes_and_the_objects_not_yet_held() {
    let need = need(300, &[("a", 100), ("b", 200), ("c", 400)]);
    assert_eq!(need.additional(|_| false), 1000 * MIB);
    let held: HashSet<Key> = [key("f", "a"), key("f", "c")].into();
    assert_eq!(need.additional(|key| held.contains(key)), (300 + 200) * MIB);
    assert_eq!(need.additional(|_| true), 300 * MIB);
    assert_eq!(need.weight_bytes(), 700 * MIB);
}

#[test]
fn what_is_available_is_the_lower_of_the_devices_report_and_the_books_less_the_margin() {
    let mib = MIB as i64;
    // The device reports 6000 free, but the cache holds 5000 and a running engine has 1500 of other bytes not yet
    // allocated: of the 10000 free at open, 3500 are left.
    assert_eq!(
        Admission::available(6000 * mib, 10000 * mib, 5000 * mib, 1500 * mib, 512 * mib),
        (3500 - 512) * mib
    );
    // The device reports less than the books.
    assert_eq!(
        Admission::available(2000 * mib, 10000 * mib, 5000 * mib, 1500 * mib, 512 * mib),
        (2000 - 512) * mib
    );
}

#[test]
fn engines_start_in_order_while_they_fit_and_the_rest_wait() {
    let (free_at_open, margin) = (10000 * MIB as i64, 512 * MIB as i64);
    // Four engines share a, b (3000 MiB of weights) and differ in one 1000 MiB object each; each has 800 MiB of
    // other bytes. The device's report is taken as the books say.
    let waiting: Vec<Need> = ["x", "y", "z", "w"]
        .iter()
        .map(|own| need(800, &[("a", 1500), ("b", 1500), (own, 1000)]))
        .collect();
    let mut cache: HashMap<Key, u64> = HashMap::new();
    let (mut cache_bytes, mut other_bytes, mut running) = (0_i64, 0_i64, 0);
    for need in &waiting {
        let additional = need.additional(|key| cache.contains_key(key));
        let available = Admission::available(
            free_at_open - cache_bytes - other_bytes,
            free_at_open,
            cache_bytes,
            other_bytes,
            margin,
        );
        if !Admission::admits(running, 0, additional, available) {
            break;
        }
        for (key, bytes) in &need.objects {
            if cache.insert(key.clone(), *bytes).is_none() {
                cache_bytes += *bytes as i64;
            }
        }
        other_bytes += need.other_bytes as i64;
        running += 1;
    }
    // The first adds 4000 + 800, the second and third 1000 + 800 each (a and b shared); the fourth's 1800 does not
    // fit in what is left: 10000 - 512 - (3000 + 3000) - 2400 = 88.
    assert_eq!(running, 3);
}

#[test]
fn an_engine_too_big_for_the_device_still_runs_alone_and_the_limit_holds() {
    assert!(Admission::admits(0, 0, u64::MAX, -1));
    assert!(!Admission::admits(1, 0, 10, 9));
    assert!(Admission::admits(1, 0, 9, 9));
    // At the limit nothing more starts, however much room there is.
    assert!(!Admission::admits(2, 2, 1, 1000));
    assert!(Admission::admits(1, 2, 1, 1000));
}

#[test]
fn a_refusal_says_by_how_much_the_engine_does_not_fit() {
    let refusal = Admission::check(1, 0, 900 * MIB, 600 * MIB as i64).unwrap_err();
    assert_eq!(
        refusal,
        Refusal::Short {
            additional: 900 * MIB,
            available: 600 * MIB as i64
        }
    );
    assert_eq!(refusal.short_by(), 300 * MIB);
    assert!(refusal.to_string().ends_with(&format!("short by {}", 300 * MIB)));
    // With no room at all it is short by everything it adds.
    assert_eq!(Admission::check(1, 0, 5, -100).unwrap_err().short_by(), 105);
    // A refusal by the limit is not about bytes.
    let limit = Admission::check(2, 2, 1, 1000).unwrap_err();
    assert_eq!(limit, Refusal::Limit { running: 2, limit: 2 });
    assert_eq!(limit.short_by(), 0);
}

/// Frames whose worker survives a panicking task, as the lattice's do.
struct Catching;

impl Frames for Catching {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        let _ = panic::catch_unwind(AssertUnwindSafe(task));
    }
}

struct Panics;

impl Backend for Panics {
    type Handle = u64;
    fn load(&self, _key: &Key, _bytes: u64) -> Result<u64, LoadError> {
        panic!("the load panics");
    }
}

#[test]
fn a_panicking_load_still_answers_the_waiter() {
    let cache = WeightCache::new(Panics, Arc::new(Inline));
    let answer: Arc<Mutex<Option<bool>>> = Arc::default();
    let put = Arc::clone(&answer);
    cache.acquire(key("base", "norm"), 64, &Catching, move |outcome| {
        *put.lock().unwrap() = Some(outcome.is_ok());
    });
    assert!(
        answer.lock().unwrap().is_some(),
        "the waiter of a panicking load was never called"
    );
}

#[test]
fn a_panicking_callback_does_not_starve_the_other_waiters() {
    let (cache, _) = rig(Fake::default());
    let frames = Deferred::default();
    cache.acquire(key("base", "norm"), 64, &frames, |_| {
        panic!("the first callback panics")
    });
    let second = ask(&cache, &frames, key("base", "norm"), 64);
    let _ = panic::catch_unwind(AssertUnwindSafe(|| frames.run_all()));
    assert!(answered(&second), "the second waiter was never called");
}

/// Frames shared with the cache, so the test can run the frees it holds back.
struct Held(Arc<Deferred>);

impl Frames for Held {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        self.0.submit(task);
    }
}

#[test]
fn an_object_is_freed_in_a_frame_and_counted_until_it_is() {
    let frees = Arc::new(Deferred::default());
    let told = Arc::new(AtomicUsize::new(0));
    let fake = Arc::new(Fake::default());
    let observer = Arc::clone(&told);
    let cache = WeightCache::with_freed(Arc::clone(&fake), Arc::new(Held(Arc::clone(&frees))), move || {
        observer.fetch_add(1, Ordering::AcqRel);
    });
    let held = lease(&ask(&cache, &Inline, key("base", "norm"), 64));
    drop(held);
    // The last lease is gone, the memory is not back: the object is still counted, and nobody was told.
    assert!(!cache.contains(&key("base", "norm")));
    assert_eq!(cache.frees_in_flight(), 1);
    assert_eq!(cache.resident_bytes(), 64);
    assert_eq!(fake.live.load(Ordering::Acquire), 1);
    assert_eq!(told.load(Ordering::Acquire), 0);
    frees.run_all();
    assert_eq!(cache.frees_in_flight(), 0);
    assert_eq!(cache.resident_bytes(), 0);
    assert_eq!(fake.live.load(Ordering::Acquire), 0);
    assert_eq!(told.load(Ordering::Acquire), 1);
}

#[test]
fn a_free_no_frame_runs_still_frees_the_object() {
    let (cache, fake) = rig_with(Fake::default(), Arc::new(Refusing));
    drop(lease(&ask(&cache, &Inline, key("base", "norm"), 64)));
    assert_eq!(fake.live.load(Ordering::Acquire), 0);
    assert_eq!(cache.resident_bytes(), 0);
    assert_eq!(cache.frees_in_flight(), 0);
}
