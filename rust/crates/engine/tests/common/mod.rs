//! The tensor table of the RCO artifact and the model around it, for the tests that need neither the file nor a GPU.

#![allow(dead_code, reason = "each test file uses some of these helpers")]
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the tests of one binary take turns on the card: a lock of the test harness, no engine state"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test helper: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the log counts the engine loads")]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Once, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::residency::budget;
use euhedral_inference_engine::residency::offload::{EMBEDDING, mapped};
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_with};
use euhedral_inference_settings::{Auto, ByteSize};

use euhedral_inference_edrl::{
    FOOTER_BYTES, HEADER_BYTES, Header, LayerType, Metadata, ModelMetadata, ReadPlan, Region, TensorTable,
    TokenizerFooter, content_end,
};
use euhedral_inference_engine::error::EngineError;
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_engine::model::dflash2::Dflash2Config;
use euhedral_inference_engine::{EngineRuntime, RuntimeOwner};
use euhedral_inference_settings::{ModelSettings, Settings};

pub fn fixture() -> &'static str {
    include_str!("../fixtures/rco_table.tsv")
}

pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// The RCO artifact's table, rebuilt in the container's binary form with synthetic offsets, and parsed.
pub fn rco_table_without(skip: impl Fn(&str) -> bool) -> TensorTable {
    table_of(fixture(), skip)
}

/// The table of a Q3 artifact (`compressed`: its P2E2 form), rebuilt the same way.
pub fn q3_table(compressed: bool) -> TensorTable {
    table_of(
        if compressed {
            include_str!("../fixtures/q3_compressed_table.tsv")
        } else {
            include_str!("../fixtures/q3_table.tsv")
        },
        |_| false,
    )
}

/// A tensor table listed in a fixture's rows (`name`, `shape`, ordinals of dtype, format and layout, bytes), rebuilt in
/// the container's binary form with synthetic offsets, and parsed.
pub fn table_of(fixture: &str, skip: impl Fn(&str) -> bool) -> TensorTable {
    let rows: Vec<Vec<&str>> = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|columns| !skip(columns[0]))
        .collect();
    let mut table = Vec::new();
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for columns in &rows {
        let shape: Vec<u64> = columns[1].split('x').map(|d| d.parse().unwrap()).collect();
        let mut entry = Vec::new();
        put_u32(&mut entry, columns[0].len() as u32);
        entry.extend_from_slice(columns[0].as_bytes());
        put_u32(&mut entry, shape.len() as u32);
        for dimension in &shape {
            put_u64(&mut entry, *dimension);
        }
        for ordinal in &columns[2..5] {
            put_u32(&mut entry, ordinal.parse().unwrap());
        }
        entries.push((entry, columns[5].parse().unwrap()));
    }
    let table_bytes: usize = entries.iter().map(|(entry, _)| entry.len() + 16).sum();
    let table_offset = HEADER_BYTES as u64;
    let data_offset = table_offset + table_bytes as u64;
    let mut offset = data_offset.div_ceil(4096) * 4096;
    for (entry, bytes) in &entries {
        table.extend_from_slice(entry);
        put_u64(&mut table, offset);
        put_u64(&mut table, *bytes);
        offset = (offset + bytes).div_ceil(4096) * 4096;
    }
    let mut header = Vec::new();
    put_u32(&mut header, 0x5157_454E);
    put_u32(&mut header, 2);
    put_u64(&mut header, HEADER_BYTES as u64);
    put_u64(&mut header, 0);
    put_u64(&mut header, table_offset);
    put_u32(&mut header, entries.len() as u32);
    put_u32(&mut header, 0);
    put_u64(&mut header, data_offset);
    let header = Header::parse(header.as_slice().try_into().unwrap()).unwrap();
    TensorTable::parse(&table, &header, offset).unwrap()
}

/// The drafter configuration words of the DFlash2 artifact.
pub fn drafter_words() -> Vec<i32> {
    vec![
        1,
        5,
        5120,
        17_408,
        32,
        8,
        128,
        8,
        248_070,
        2,
        16,
        16,
        256,
        2048,
        248_320,
        1e-6f32.to_bits() as i32,
        1e7f32.to_bits() as i32,
        5,
        5,
        19,
        33,
        47,
        61,
    ]
}

/// The model's configuration with the drafter's, as the engine holds them once the artifact's file is open.
pub fn dflash2_config() -> ModelConfig {
    let bytes: Vec<u8> = drafter_words().iter().flat_map(|word| word.to_le_bytes()).collect();
    ModelConfig {
        dflash2: Some(Dflash2Config::parse(&bytes).unwrap()),
        ..config()
    }
}

/// The rows of the drafter's objects (`name`, shape, dtype, format and layout ordinals, bytes), as the fixture lists
/// them; the projections are NVFP4 when `nvfp4`, else BF16.
pub fn drafter_rows(nvfp4: bool) -> String {
    let drafter = dflash2_config().dflash2.unwrap();
    let mut rows = String::new();
    let mut put = |name: &str, shape: &[usize], dtype: u32, format: u32, layout: u32| {
        let elements: usize = shape.iter().product();
        let bytes = match format {
            3 => 4 * elements,
            7 => {
                // NVFP4 in the K128 row split: codes, then one scale per 16 columns, each plane aligned, then a trailer.
                let (rows, k) = (shape[0], shape[1].div_ceil(128) * 128);
                let scales = (rows * k / 2).div_ceil(256) * 256;
                (scales + rows * (k / 16)).div_ceil(256) * 256 + 4
            }
            _ => 2 * elements,
        };
        let shape: Vec<String> = shape.iter().map(ToString::to_string).collect();
        rows.push_str(&format!(
            "{name}\t{}\t{dtype}\t{format}\t{layout}\t{bytes}\n",
            shape.join("x")
        ));
    };
    let words = drafter_words();
    put("dflash2/config", &[words.len()], 6, 3, 0);
    let (hidden, rank) = (drafter.hidden, drafter.selector_rank);
    let (projected_format, projected_layout) = if nvfp4 { (7, 1) } else { (0, 0) };
    put(
        "dflash2/fc",
        &[hidden, drafter.tap_width()],
        0,
        projected_format,
        projected_layout,
    );
    put("dflash2/hidden_norm", &[hidden], 0, 0, 0);
    put("dflash2/final_norm", &[hidden], 0, 0, 0);
    put("dflash2/selector/hidden_projection", &[rank, hidden], 0, 0, 0);
    put("dflash2/selector/predecessor", &[drafter.vocab, rank], 0, 0, 0);
    put("dflash2/selector/successor", &[drafter.vocab, rank], 0, 0, 0);
    for layer in 0..drafter.layers {
        let at = |tail: &str| format!("dflash2/layers/{layer}/{tail}");
        put(&at("input_norm"), &[hidden], 0, 0, 0);
        put(&at("post_attention_norm"), &[hidden], 0, 0, 0);
        put(
            &at("attention/query"),
            &[drafter.query_width(), hidden],
            0,
            projected_format,
            projected_layout,
        );
        put(
            &at("attention/key_value"),
            &[2 * drafter.kv_width(), hidden],
            0,
            projected_format,
            projected_layout,
        );
        put(
            &at("attention/output"),
            &[hidden, drafter.query_width()],
            0,
            projected_format,
            projected_layout,
        );
        put(&at("attention/query_norm"), &[drafter.head_dim], 0, 0, 0);
        put(&at("attention/key_norm"), &[drafter.head_dim], 0, 0, 0);
        put(
            &at("mlp/gate_up"),
            &[2 * drafter.intermediate, hidden],
            0,
            projected_format,
            projected_layout,
        );
        put(
            &at("mlp/down"),
            &[hidden, drafter.intermediate],
            0,
            projected_format,
            projected_layout,
        );
        for conv in ["attention_conv", "mlp_conv"] {
            put(&at(&format!("{conv}/base")), &[2, drafter.conv_kernel, hidden], 0, 0, 0);
            put(
                &at(&format!("{conv}/projection")),
                &[drafter.conv_projection_width(), hidden],
                0,
                projected_format,
                projected_layout,
            );
        }
    }
    rows
}

/// The RCO artifact's table with the drafter's objects added (the DFlash2 artifact keeps the MTP layer), less the
/// objects `skip` names.
pub fn dflash2_table_without(nvfp4: bool, skip: impl Fn(&str) -> bool) -> TensorTable {
    let base = fixture().to_string();
    table_of(&(base + &drafter_rows(nvfp4)), skip)
}

pub fn rco_table() -> TensorTable {
    rco_table_without(|_| false)
}

pub fn metadata() -> ModelMetadata {
    ModelMetadata {
        vocab_size: 248_320,
        hidden_size: 5120,
        num_hidden_layers: 64,
        num_attention_heads: 24,
        num_key_value_heads: 4,
        attention_head_dim: 256,
        intermediate_size: 17_408,
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1e-6,
        rope_theta: 1e7,
        partial_rotary_factor: 0.25,
        max_position_embeddings: 262_144,
        hidden_activation: "silu".to_string(),
        layer_types: (0..64)
            .map(|layer| {
                if layer % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::GatedDeltaNet
                }
            })
            .collect(),
        num_experts: 0,
        num_experts_per_token: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        tie_word_embeddings: false,
        attention_output_gate: true,
        mtp_layer_count: 1,
        draft_depth: Some(4),
    }
}

pub fn config() -> ModelConfig {
    ModelConfig::from(&metadata())
}

pub fn settings(max_tokens: u32) -> Settings {
    let mut settings = Settings::with_model(ModelSettings::default());
    settings.context.max_tokens = max_tokens;
    settings
}

/// What closes a device test's engine, on every path out of the test. A test that panics unwinds through its locals: the
/// runtime's device memory is freed only by the owner's shutdown, so without this one failure leaves the card full and
/// every later test of the binary fails to load its weights. Bind it before the runtime (`let (closer, runtime) = ...`):
/// the runtime's own handle is then dropped first, and the shutdown here finds the last one.
pub struct Closer {
    owner: Option<RuntimeOwner>,
    runtime: Option<Arc<EngineRuntime>>,
}

impl Closer {
    pub fn new(owner: RuntimeOwner, runtime: &Arc<EngineRuntime>) -> Closer {
        Closer {
            owner: Some(owner),
            runtime: Some(Arc::clone(runtime)),
        }
    }

    /// A handle to the engine.
    pub fn runtime(&self) -> Arc<EngineRuntime> {
        Arc::clone(self.runtime.as_ref().expect("the engine is not shut down yet"))
    }

    /// Shuts the engine down: the test's `runtime` is the last handle.
    pub fn shutdown(mut self, runtime: Arc<EngineRuntime>) -> Result<(), EngineError> {
        drop(self.runtime.take());
        self.owner
            .take()
            .expect("the engine is not shut down yet")
            .shutdown(runtime)
    }

    /// Shuts the engine down within `window`.
    pub fn shutdown_within(mut self, runtime: Arc<EngineRuntime>, window: Duration) -> Result<(), EngineError> {
        drop(self.runtime.take());
        self.owner
            .take()
            .expect("the engine is not shut down yet")
            .shutdown_within(runtime, window)
    }
}

impl Drop for Closer {
    fn drop(&mut self) {
        // Reached with the engine still up only when the test unwound.
        if let (Some(owner), Some(runtime)) = (self.owner.take(), self.runtime.take()) {
            let _shutting = doing("shutting down its engine after the test unwound");
            if let Err(error) = owner.shutdown(runtime) {
                mark_unfreed(&error.to_string());
            }
        }
    }
}

/// Set when an engine's device memory could not be freed (another handle to it was alive): the card is then full, and no
/// test loads a second model on top of it.
static UNFREED: Mutex<Option<String>> = Mutex::new(None);

fn mark_unfreed(why: &str) {
    let name = thread::current().name().unwrap_or("?").to_owned();
    *UNFREED.lock().unwrap_or_else(PoisonError::into_inner) = Some(format!("found by {name}, {why}"));
}

/// Fails the test that would load a model on a card an earlier engine still fills.
fn require_free_card() {
    if let Some(why) = UNFREED.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
        panic!("an engine of an earlier test still holds the card, so no model is loaded on top of it ({why})");
    }
}

/// What the card holds between tests: the engine the last test left running, under the key of the settings it started
/// with, and how many tests use the card without an engine.
struct State {
    engine: Option<(String, Closer)>,
    /// The test that held the card last, for the failure of a shutdown it left unclean.
    last: String,
    light: usize,
    exclusive: bool,
    /// The tests waiting for the card, oldest first, with the key of the engine each one wants (none: an empty card).
    waiting: Vec<(u64, Option<String>)>,
    tickets: u64,
    /// The tests inside the card now, by thread name, for the message of a hang.
    holders: Vec<String>,
}

/// How long the run may go with no test taking or leaving the card, starting or shutting an engine down, or waiting
/// for one of those, before the watchdog ends it naming what every live test is doing. The longest test is about two
/// minutes.
const HANG_GUARD: Duration = Duration::from_secs(5 * 60);

fn thread_name() -> String {
    thread::current().name().unwrap_or("?").to_owned()
}

/// Counts the steps the run has made; the watchdog compares it with what it saw before.
static PROGRESS: AtomicU64 = AtomicU64::new(0);

fn progress() {
    PROGRESS.fetch_add(1, Ordering::Relaxed);
}

/// What each test thread is waiting on or doing now, outside the card's lock, so that the watchdog can read it when
/// the lock itself is held by a start or a shutdown that does not end.
static ACTIVITY: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Records what the thread does until the guard drops.
struct Doing(String);

fn doing(what: impl Into<String>) -> Doing {
    let name = thread_name();
    ACTIVITY
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((name.clone(), what.into()));
    progress();
    Doing(name)
}

impl Drop for Doing {
    fn drop(&mut self) {
        let mut all = ACTIVITY.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(at) = all.iter().rposition(|(name, _)| *name == self.0) {
            all.remove(at);
        }
        drop(all);
        progress();
    }
}

/// Starts the thread that ends the process, naming every live test and what it does, when [`PROGRESS`] does not move
/// for [`HANG_GUARD`]: whatever the tests wait on, a hung job then fails in minutes with the names.
fn watchdog() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        thread::Builder::new()
            .name("device-test-watchdog".to_owned())
            .spawn(|| {
                let mut seen = PROGRESS.load(Ordering::Relaxed);
                let mut since = Instant::now();
                loop {
                    thread::sleep(Duration::from_secs(5));
                    let now = PROGRESS.load(Ordering::Relaxed);
                    if now != seen {
                        seen = now;
                        since = Instant::now();
                    } else if since.elapsed() >= HANG_GUARD {
                        hang_report();
                    }
                }
            })
            .expect("the watchdog thread starts");
    });
}

fn hang_report() -> ! {
    let activity = ACTIVITY
        .try_lock()
        .map_or_else(|_| "the activity list is locked".to_owned(), |all| format!("{all:#?}"));
    let card = STATE.try_lock().map_or_else(
        |_| {
            "the card's lock is held by a thread that does not release it (an engine start or shutdown under it)"
                .to_owned()
        },
        |state| {
            format!(
                "holders {:?}, exclusive {}, light {}, waiting {} tests, engine kept {:?}, last to keep it {}",
                state.holders,
                state.exclusive,
                state.light,
                state.waiting.len(),
                state
                    .engine
                    .as_ref()
                    .map(|(key, _)| key.chars().take(60).collect::<String>()),
                state.last,
            )
        },
    );
    println!(
        "HANG: no test took or left the card, started or shut an engine down for {} s; {} loads so far.\n\
         what each test thread is doing: {activity}\nthe card: {card}",
        HANG_GUARD.as_secs(),
        LOADS.load(Ordering::Relaxed),
    );
    process::abort()
}

/// Waits for a change of the card.
fn wait_for_change<'a>(state: MutexGuard<'a, State>, wanted: &str) -> MutexGuard<'a, State> {
    let _waiting = doing(format!("waiting for {wanted}"));
    CHANGED.wait(state).unwrap_or_else(PoisonError::into_inner)
}

static STATE: Mutex<State> = Mutex::new(State {
    engine: None,
    last: String::new(),
    light: 0,
    exclusive: false,
    waiting: Vec::new(),
    tickets: 0,
    holders: Vec::new(),
});
static CHANGED: Condvar = Condvar::new();
static LOADS: AtomicUsize = AtomicUsize::new(0);

/// The longest an engine start waits for its latch.
const START_WAIT: Duration = Duration::from_secs(5 * 60);

fn lock() -> MutexGuard<'static, State> {
    watchdog();
    // A test that failed holding the card leaves it usable: the next one starts from what is left.
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The host memory the process holds and the machine can still spare, for the log: engines read the second at start to
/// size their host budget, so a test process that keeps memory across engines starves the later ones.
fn host_memory() -> String {
    let read = |path: &str, key: &str| {
        fs::read_to_string(path)
            .ok()
            .and_then(|text| {
                text.lines()
                    .find_map(|line| line.strip_prefix(key).map(|rest| rest.trim().to_owned()))
            })
            .unwrap_or_default()
    };
    format!(
        "process {}, machine available {}; {}",
        read("/proc/self/status", "VmRSS:"),
        read("/proc/meminfo", "MemAvailable:"),
        cgroup_budget()
    )
}

/// The figures an engine start reads to size its host budget, from the process's cgroup and its parents, with the
/// budget they give (the engine's own rule, from its public parts): one line per group, `max`, `current`, the
/// reclaimable page cache (`-` when `memory.stat` is unreadable, which counts the cache as used) and the headroom.
pub fn cgroup_budget() -> String {
    let root = Path::new("/sys/fs/cgroup");
    let own = fs::read_to_string("/proc/self/cgroup").ok();
    let mut group = own
        .as_deref()
        .and_then(budget::cgroup_path)
        .map(|path| root.join(path.trim_start_matches('/')));
    let mut least: Option<u64> = None;
    let mut lines = Vec::new();
    while let Some(path) = group.take().filter(|path| path.starts_with(root)) {
        let read = |name: &str| fs::read_to_string(path.join(name));
        let (max, current) = (read("memory.max").ok(), read("memory.current").ok());
        let stat = read("memory.stat").ok();
        let headroom = max.as_deref().zip(current.as_deref()).and_then(|(max, current)| {
            let stat = (max.trim() != "max").then_some(stat.as_deref()).flatten();
            budget::cgroup_headroom(max, current, stat)
        });
        least = match (least, headroom) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        lines.push(format!(
            "[{} max {} current {} reclaimable {} headroom {}]",
            path.display(),
            max.as_deref().map_or("-", str::trim),
            current.as_deref().map_or("-", str::trim),
            stat.as_deref()
                .and_then(budget::reclaimable_file_bytes)
                .map_or_else(|| "-".to_owned(), |bytes| bytes.to_string()),
            headroom.map_or_else(|| "-".to_owned(), |bytes| bytes.to_string()),
        ));
        if path != root {
            group = path.parent().map(Path::to_path_buf);
        }
    }
    let meminfo = fs::read_to_string("/proc/meminfo").ok();
    let available = budget::available_bytes(meminfo.as_deref().and_then(budget::mem_available), least);
    let host = available.map(|bytes| budget::HostBudget::of_available(bytes, Auto::Auto).pinnable_bytes);
    format!("cgroups {}; spare {available:?}; host budget {host:?}", lines.join(" "))
}

/// Counts one model load in the log: a test that starts an engine itself, to see how the start ends, calls this.
pub fn note_load() {
    require_free_card();
    let loads = LOADS.fetch_add(1, Ordering::Relaxed) + 1;
    println!("ENGINE LOAD {loads} ({})", host_memory());
}

/// Starts an engine and waits for its latch, as `main` does, and counts the start: every engine of the device tests
/// is started here, so the count in the log (`ENGINE LOAD n`) is the number of model loads of the run.
pub fn start_engine(settings: &Settings, library: &Path) -> (Closer, Arc<EngineRuntime>) {
    require_free_card();
    let _starting = doing("starting an engine");
    let (owner, ready) = EngineRuntime::start(settings, library);
    let runtime = ready
        .wait(START_WAIT)
        .unwrap_or_else(|_| {
            panic!(
                "test {} waited {} s for its engine to report its start; the card is held by {:?}",
                thread_name(),
                START_WAIT.as_secs(),
                lock().holders,
            )
        })
        .expect("the engine starts");
    note_load();
    (Closer::new(owner, &runtime), runtime)
}

/// The card, held by one test at a time unless the tests need no engine. Every engine of the model fills the card, so
/// two engines never exist side by side: a test takes the card, then either runs on the engine the tests before it left
/// running ([`engine`]), or starts one of its own after `card()` and shuts it down.
pub struct Card {
    held: Option<(String, Closer)>,
    /// The test that held the engine before.
    before: String,
}

/// Waits for the card, alone. When the card is free the next test is the oldest one that wants the engine already
/// running, so that tests with the same settings follow each other whatever the order they arrived in; with none, the
/// oldest.
fn acquire_exclusive(wanted: Option<&str>) -> Card {
    let mut state = lock();
    let ticket = state.tickets;
    state.tickets += 1;
    state.waiting.push((ticket, wanted.map(str::to_owned)));
    loop {
        if !state.exclusive && state.light == 0 {
            let running = state.engine.as_ref().map(|(key, _)| key.as_str());
            let next = state
                .waiting
                .iter()
                .find(|(_, key)| running.is_some() && key.as_deref() == running)
                .or_else(|| state.waiting.first())
                .map(|(next, _)| *next);
            if next == Some(ticket) {
                break;
            }
        }
        state = wait_for_change(state, wanted.unwrap_or("an empty card"));
    }
    state.waiting.retain(|(waiting, _)| *waiting != ticket);
    state.exclusive = true;
    state.holders.push(thread_name());
    progress();
    Card {
        held: state.engine.take(),
        before: state.last.clone(),
    }
}

/// Shuts the engine of `held` down, which must end clean: the leak check of a shared engine is made here, where it ends.
fn retire((_, closer): (String, Closer), before: &str) {
    let _shutting = doing(format!("shutting down the engine the test {before} left"));
    let runtime = closer.runtime();
    closer.shutdown(runtime).unwrap_or_else(|error| {
        mark_unfreed(&format!("the engine the test {before} left: {error}"));
        panic!("the engine the test {before} left shuts down clean: {error}")
    });
}

/// Takes the card for a test that starts its own engine (a shutdown, a fault, a failed start), or that needs the card
/// empty. Waits for the tests that hold it, then shuts the cached engine down.
pub fn card() -> Card {
    let mut card = acquire_exclusive(None);
    if let Some(held) = card.held.take() {
        retire(held, &card.before);
    }
    card
}

/// Takes the card for a test that runs on an engine started with `settings`: the one the tests before it left running
/// when their settings are these, else a new one, after the old one is shut down. The test runs alone against it: a
/// request starts each sequence from nothing, and the tests read the first sequence and expect it idle, so tests do not
/// interleave on one engine. The test keeps no handle past its end, and leaves the engine as a request leaves it.
pub fn engine(settings: &Settings, library: &Path) -> (Card, Arc<EngineRuntime>) {
    let key = format!("{settings:?} {}", library.display());
    let mut card = acquire_exclusive(Some(&key));
    if card.held.as_ref().is_some_and(|(held, _)| *held != key) {
        {
            let held = card.held.take().expect("checked");
            retire(held, &card.before);
        }
    }
    let runtime = match card.held.as_ref() {
        Some((_, closer)) => closer.runtime(),
        None => {
            let (closer, runtime) = start_engine(settings, library);
            card.held = Some((key, closer));
            runtime
        }
    };
    (card, runtime)
}

impl Drop for Card {
    fn drop(&mut self) {
        let mut state = lock();
        // A test that failed may have left the engine in a state the next test would trip on: it is shut down, not kept.
        let held = self.held.take();
        if thread::panicking() {
            drop(state);
            drop(held);
            state = lock();
        } else {
            state.engine = held;
            state.last = thread_name();
        }
        state.exclusive = false;
        let name = thread_name();
        state.holders.retain(|holder| *holder != name);
        progress();
        drop(state);
        CHANGED.notify_all();
    }
}

/// A test that uses the card without an engine, beside others that do the same.
pub struct Light(());

/// Takes the card for a test of kernels alone: it waits for the test that holds an engine and shuts the cached one
/// down, then runs beside the other tests of this kind.
pub fn light() -> Light {
    let mut state = lock();
    loop {
        if state.exclusive {
            state = wait_for_change(state, "the card, for a test without an engine");
        } else if let Some(held) = state.engine.take() {
            let before = state.last.clone();
            retire(held, &before);
        } else {
            break;
        }
    }
    state.light += 1;
    state.holders.push(thread_name());
    progress();
    Light(())
}

impl Drop for Light {
    fn drop(&mut self) {
        let mut state = lock();
        state.light -= 1;
        let name = thread_name();
        if let Some(at) = state.holders.iter().position(|holder| *holder == name) {
            state.holders.remove(at);
        }
        progress();
        drop(state);
        CHANGED.notify_all();
    }
}

/// The model config and tensor table of the artifact at `path`, read from its header, metadata and table.
#[allow(
    clippy::disallowed_methods,
    reason = "a test reads the artifact's index before any engine runs"
)]
pub fn open_artifact(path: &Path) -> (ModelConfig, TensorTable) {
    let mut file = File::open(path).unwrap();
    let length = file.metadata().unwrap().len();
    let mut header_bytes = [0u8; HEADER_BYTES];
    file.read_exact(&mut header_bytes).unwrap();
    let header = Header::parse(&header_bytes).unwrap();
    let mut footer_bytes = [0u8; FOOTER_BYTES];
    file.seek(SeekFrom::Start(length - FOOTER_BYTES as u64)).unwrap();
    file.read_exact(&mut footer_bytes).unwrap();
    let footer = TokenizerFooter::parse(&footer_bytes).unwrap();
    let end = content_end(length, footer.as_ref()).unwrap();
    let plan = ReadPlan::new(&header, length, footer.as_ref()).unwrap();
    let mut read = |region: Region| {
        let range = plan.range(region).unwrap();
        let mut bytes = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        bytes
    };
    let metadata = Metadata::parse(&read(Region::Metadata)).unwrap();
    let table = TensorTable::parse(&read(Region::Table), &header, end).unwrap();
    (ModelConfig::from(&metadata), table)
}

/// The `memory.gpu-limit` that leaves `below` bytes under the device need of the resident plan for `settings`. The
/// plan is made from the header and the tensor table of the artifact `settings` names, on this device's multiprocessor
/// count with no limit on the device or the host, never on the card's free memory, which other jobs change. A card
/// with less free memory than that limit fails here, with the figures: a forced plan the card cannot give must not
/// pass as a skipped test.
pub fn limit_below_need(library: &Path, settings: &Settings, below: u64) -> u64 {
    let loaded = CudaLibrary::load(library).unwrap();
    let inputs = PlanInputs {
        device_free: 64 << 30,
        multiprocessors: Some(loaded.device().sm_count),
        exact_numerics: false,
        host_pinned_bytes: Some(u64::MAX),
        sequences: 1,
    };
    let (config, table) = open_artifact(&settings.model.artifact);
    let resident = plan_with(&config, &table, settings, inputs).unwrap();
    assert!(
        resident.offload.is_empty(),
        "the unlimited plan keeps everything on the device"
    );
    let limit = resident.device_bytes - below;
    let free = loaded.free_bytes().unwrap();
    assert!(
        free >= limit,
        "the card has {free} bytes free, the forced limit needs {limit}: run on an idle card"
    );
    limit
}

/// Sets `memory.gpu-limit` to `limit`.
pub fn limit_gpu(settings: &mut Settings, limit: u64) {
    settings.memory.gpu_limit = Auto::Value(ByteSize(limit));
}

/// Asserts the plan of a started engine is the one a test forced: the device budget is the limit and the embedding
/// alone is held on the host.
pub fn embedding_alone_is_on_the_host(runtime: &EngineRuntime, limit: u64) {
    let plan = runtime.plan();
    assert_eq!(plan.budget_bytes, limit, "the plan's device budget is the forced limit");
    assert_eq!(
        plan.offload.host,
        [EMBEDDING],
        "the plan holds the embedding alone on the host"
    );
    assert!(plan.offload.disk.is_empty(), "no object is left in the file");
}

/// The `memory.gpu-limit` that leaves `staged` bytes under the device need of the resident plan for `settings` (its
/// context size and its sequences), so the plan must stage projections through a ring. Made as [`limit_below_need`]:
/// from the header and the table, with no limit on the device or the host.
pub fn limit_to_stage(library: &Path, settings: &Settings, staged: u64) -> u64 {
    let loaded = CudaLibrary::load(library).unwrap();
    let inputs = PlanInputs {
        device_free: 64 << 30,
        multiprocessors: Some(loaded.device().sm_count),
        exact_numerics: false,
        host_pinned_bytes: Some(u64::MAX),
        sequences: settings.batching.max_sequences as usize,
    };
    let (config, table) = open_artifact(&settings.model.artifact);
    let resident = plan_with(&config, &table, settings, inputs).unwrap();
    assert!(
        resident.offload.is_empty(),
        "the unlimited plan keeps everything on the device"
    );
    let limit = resident.device_bytes - staged;
    let free = loaded.free_bytes().unwrap();
    assert!(
        free >= limit,
        "the card has {free} bytes free, the forced limit needs {limit}: run on an idle card"
    );
    limit
}

/// Whether the plan of a started engine is the staged one a test forced: the device budget is the limit, projections
/// are held on the host and read through the staging ring, and nothing is left in the file. A budget other than the
/// limit (the sequences or the context differ from the ones the limit was made for) prints why and returns false, so
/// the test skips.
#[allow(clippy::print_stdout, reason = "a test says why it skipped")]
pub fn staged_through_a_ring(runtime: &EngineRuntime, limit: u64) -> bool {
    let plan = runtime.plan();
    if plan.budget_bytes != limit {
        println!(
            "skipped: the plan's budget is {} bytes, not the forced limit {limit}",
            plan.budget_bytes
        );
        return false;
    }
    let staged = plan.offload.host.iter().filter(|name| !mapped(name)).count();
    assert!(
        staged > 0,
        "the plan holds projections on the host: {:?}",
        plan.offload.host
    );
    assert!(plan.offload.disk.is_empty(), "no object is left in the file");
    let ring = plan.offload.staging.expect("the plan stages through a ring");
    let report = runtime.report();
    assert!(report.host_backed_bytes > 0);
    assert_eq!(report.disk_backed_bytes, 0);
    assert!(report.staging_bytes > 0, "the startup report shows the ring: {ring:?}");
    true
}

/// The `memory.host-weights-limit` that, with `memory.gpu-limit` at `gpu_limit`, leaves `on_disk` bytes of the host
/// arena without room: the plan must read the objects that no longer fit from the artifact file. Made from the header
/// and the table, as [`limit_below_need`] is, with no limit on the host's pinned memory.
pub fn limit_to_disk(library: &Path, settings: &Settings, gpu_limit: u64, on_disk: u64) -> u64 {
    let loaded = CudaLibrary::load(library).unwrap();
    let mut forced = settings.clone();
    limit_gpu(&mut forced, gpu_limit);
    let inputs = PlanInputs {
        device_free: 64 << 30,
        multiprocessors: Some(loaded.device().sm_count),
        exact_numerics: false,
        host_pinned_bytes: Some(u64::MAX),
        sequences: settings.batching.max_sequences as usize,
    };
    let (config, table) = open_artifact(&settings.model.artifact);
    let staged = plan_with(&config, &table, &forced, inputs).unwrap();
    assert!(
        staged.offload.disk.is_empty(),
        "the host holds what the device does not"
    );
    let mut limit = staged
        .offload
        .host_arena_bytes
        .checked_sub(on_disk)
        .expect("the host arena is larger than the bytes to leave in the file");
    // The limit also holds the window the disk reads land in and the cache pages: raise it by what the plan says it is
    // short until the plan fits and still leaves objects in the file.
    for _ in 0..8 {
        let limited = PlanInputs {
            host_pinned_bytes: Some(limit),
            ..inputs
        };
        match plan_with(&config, &table, &forced, limited) {
            Err(PlanError::HostTooSmall { needed, budget, .. }) => limit += needed - budget,
            Ok(plan) => {
                assert!(
                    !plan.offload.disk.is_empty(),
                    "the limit {limit} leaves nothing in the file"
                );
                return limit;
            }
            Err(other) => panic!("the plan of the disk tier is refused: {other}"),
        }
    }
    panic!("no host limit leaves {on_disk} bytes in the file")
}

/// Sets `memory.host-weights-limit` to `limit`.
pub fn limit_host_weights(settings: &mut Settings, limit: u64) {
    settings.memory.host_weights_limit = Auto::Value(ByteSize(limit));
}

/// Asserts the plan of a started engine reads objects from the artifact file through the window, as a test forced:
/// the device budget is `gpu_limit`, objects are left on disk, and the report shows their bytes and the window.
#[allow(clippy::print_stdout, reason = "a test says why it skipped")]
pub fn read_through_the_window(runtime: &EngineRuntime, gpu_limit: u64) -> bool {
    let plan = runtime.plan();
    if plan.budget_bytes != gpu_limit {
        println!(
            "skipped: the plan's budget is {} bytes, not the forced limit {gpu_limit}",
            plan.budget_bytes
        );
        return false;
    }
    assert!(!plan.offload.disk.is_empty(), "the plan leaves objects in the file");
    assert!(plan.offload.window.is_some(), "the plan has a disk window");
    assert!(plan.offload.staging.is_some(), "the plan stages through a ring");
    let report = runtime.report();
    assert!(report.disk_backed_bytes > 0);
    assert!(report.window_bytes > 0);
    true
}
