use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use euhedral_inference_edrl::WeightFormat;
use euhedral_inference_engine::formats::LinearMode;

use super::cost_table::{CostTable, Key, Shape};
use super::error::MixedError;
use super::option::QuantOption;
use super::space::Sources;

/// The most rows of a launch that is a verification, not a prefill chunk: the MTP drafts verify at most this many
/// positions.
pub const VERIFY_MAX_ROWS: u32 = 8;

/// The row counts the command times when none is given: the one-row decode step, the verifications of the drafts,
/// and the prefill chunk the cost table charges a prompt token at.
pub const DEFAULT_ROWS: [u32; 7] = [1, 2, 3, 4, 5, 6, 512];

/// The timed launches of a measurement when none is given.
pub const DEFAULT_REPEATS: u32 = 50;

/// The least payload bytes the copies of an object span when none is given, in MiB: enough that the weights of the
/// next launch are not in the L2 cache.
pub const DEFAULT_ROTATE_MIB: u64 = 256;

/// The mode the engine launches a linear of `rows` rows in: a single row is a decode step, two to
/// [`VERIFY_MAX_ROWS`] rows are a verification (every row computes as a one-row launch does), and more are a prefill
/// chunk. The format picks its route from the rows and this mode, so the route a row count takes here is the route
/// the engine takes for it.
pub fn launch_mode(rows: u32) -> LinearMode {
    LinearMode {
        row_exact: rows > 1 && rows <= VERIFY_MAX_ROWS,
        prefill: rows > VERIFY_MAX_ROWS,
    }
}

/// Refuses a source whose objects are not stored in the format `option` names: the cost of an option is the cost of
/// its own format's launch, so an artifact of another format under its name would time the wrong kernels.
pub fn check_format(option: QuantOption, format: WeightFormat) -> Result<(), MixedError> {
    let wanted = if option == QuantOption::Nvfp4 {
        WeightFormat::Nvfp4
    } else {
        WeightFormat::Trellis
    };
    if format == wanted {
        return Ok(());
    }
    Err(MixedError::invalid(format!(
        "the source of {option} stores {format:?} objects, not {wanted:?}"
    )))
}

/// The device side of a measurement: the launches of one option's linear of one object. A bench holds one object at
/// a time.
pub trait Bench {
    /// Loads `copies` copies of `object` as `option` stores it, replacing the object it held, with room for launches
    /// of up to `max_rows` rows. The copies are numbered from 0.
    fn load(
        &mut self,
        option: QuantOption,
        object: &str,
        shape: Shape,
        copies: usize,
        max_rows: u32,
    ) -> Result<(), MixedError>;

    /// Launches the linear over `rows` rows once for each copy in `order` (indices into the copies, in launch order)
    /// and returns when they ended.
    fn run(&mut self, rows: u32, order: &[usize]) -> Result<(), MixedError>;

    /// As [`Bench::run`], and returns the device time from the first launch's start to the last launch's end.
    fn time(&mut self, rows: u32, order: &[usize]) -> Result<Duration, MixedError>;
}

/// What to measure: the options that have a source, the row counts, the timed launches of each measurement and the
/// least payload bytes the copies of an object span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    options: BTreeSet<QuantOption>,
    rows: Vec<u32>,
    repeats: u32,
    rotate_bytes: u64,
}

impl Plan {
    /// A plan over `options` and `rows`; `repeats` and `rotate_bytes` as [`Plan`] says. Refuses no option, no row
    /// count, a row count of 0 and no repeats.
    pub fn new(
        options: impl IntoIterator<Item = QuantOption>,
        rows: Vec<u32>,
        repeats: u32,
        rotate_bytes: u64,
    ) -> Result<Plan, MixedError> {
        let options: BTreeSet<QuantOption> = options.into_iter().collect();
        if options.is_empty() {
            return Err(MixedError::invalid("no option to measure"));
        }
        if rows.is_empty() {
            return Err(MixedError::invalid("no row counts"));
        }
        if rows.contains(&0) {
            return Err(MixedError::invalid(format!("a row count of 0 in {rows:?}")));
        }
        if repeats < 1 {
            return Err(MixedError::invalid(format!("repeats must be positive: {repeats}")));
        }
        Ok(Plan {
            options,
            rows,
            repeats,
            rotate_bytes,
        })
    }

    /// The options of the plan, in order.
    pub fn options(&self) -> impl Iterator<Item = QuantOption> + '_ {
        self.options.iter().copied()
    }

    fn max_rows(&self) -> u32 {
        self.rows.iter().copied().max().unwrap_or(1)
    }
}

/// One finished measurement, for the progress the command reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timed {
    pub key: Key,
    pub micros: f64,
}

/// The copies of an object that span `rotate_bytes` of payload, and at least one.
fn copies_for(payload: i64, rotate_bytes: u64) -> Result<usize, MixedError> {
    let payload = u64::try_from(payload)
        .ok()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| MixedError::invalid(format!("an object of {payload} bytes cannot be launched")))?;
    usize::try_from(rotate_bytes.div_ceil(payload).max(1)).map_err(|_| MixedError::invalid("too many copies"))
}

/// The objects `measure` loads, in order: the first projection of the base of each distinct shape.
pub fn measured_objects(sources: &dyn Sources) -> Result<Vec<(Shape, &str)>, MixedError> {
    let mut first_of_shape: Vec<(Shape, &str)> = Vec::new();
    for name in sources.projections() {
        let shape = sources.shape(sources.base(), name)?;
        if first_of_shape.iter().all(|(known, _)| *known != shape) {
            first_of_shape.push((shape, name));
        }
    }
    Ok(first_of_shape)
}

/// The time of the launches of one timed run, behind a gate of as many copies as it takes to hide them.
///
/// `attempt(copies)` makes one timed run behind a gate of `copies` and returns the time of the launches, or `None` when
/// the gate had ended before the last launch was queued (the time then holds a host gap and is not a measurement). A
/// run that was not hidden is repeated behind a gate twice as long, up to `max_copies`, which is where it is refused.
/// Refuses a gate of no copies. Returns the time and the length of the gate that hid the launches, which the next run may start from. An attempt's
/// own error ends the loop at once and is returned as it is.
pub fn hide_launches(
    start_copies: usize,
    max_copies: usize,
    mut attempt: impl FnMut(usize) -> Result<Option<Duration>, MixedError>,
) -> Result<(Duration, usize), MixedError> {
    if start_copies == 0 {
        return Err(MixedError::invalid("a gate of 0 copies hides nothing"));
    }
    let mut copies = start_copies;
    loop {
        if let Some(elapsed) = attempt(copies)? {
            return Ok((elapsed, copies));
        }
        if copies >= max_copies {
            return Err(MixedError::invalid(format!(
                "the launches outran a gate of {copies} copies, so their time holds a host gap: use fewer --repeats"
            )));
        }
        copies = copies.saturating_mul(2).min(max_copies);
    }
}

/// Times one launch of each option's linear for every distinct projection shape of `sources`, at each row count of
/// `plan`, on `bench`: the cost table (with no fixed terms) the allocation search reads.
///
/// A shape is measured on the first projection of the base that has it, once for each option, one object at a time.
/// An object is loaded as enough copies to span the plan's rotation bytes. A measurement launches every copy once
/// (the warmup), then times `repeats` launches round-robin over the copies; its cost is the device time over the
/// repeats, the mean of one launch. `report` is told each measurement as it ends. A failing launch ends the run, and
/// no table is made.
pub fn measure(
    sources: &dyn Sources,
    plan: &Plan,
    bench: &mut dyn Bench,
    report: &mut dyn FnMut(&Timed),
) -> Result<CostTable, MixedError> {
    let mut micros = BTreeMap::new();
    for (shape, name) in measured_objects(sources)? {
        for option in QuantOption::ALL
            .into_iter()
            .filter(|option| plan.options.contains(option))
        {
            let copies = copies_for(sources.bytes(option, name)?, plan.rotate_bytes)?;
            bench.load(option, name, shape, copies, plan.max_rows())?;
            let warmup: Vec<usize> = (0..copies).collect();
            let timed: Vec<usize> = (0..plan.repeats as usize).map(|repeat| repeat % copies).collect();
            for &rows in &plan.rows {
                bench.run(rows, &warmup)?;
                let elapsed = bench.time(rows, &timed)?;
                let measured = Timed {
                    key: Key { shape, option, rows },
                    micros: elapsed.as_secs_f64() * 1e6 / f64::from(plan.repeats),
                };
                micros.insert(measured.key, measured.micros);
                report(&measured);
            }
        }
    }
    Ok(CostTable::new(micros, 0.0, BTreeMap::new()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    /// Projections of two shapes in base order: 64 x 64, then 64 x 128 twice, then 64 x 64 again.
    struct Fake {
        projections: Vec<String>,
    }

    impl Fake {
        fn new() -> Fake {
            Fake {
                projections: [
                    "text/layers/0/a",
                    "text/layers/0/b",
                    "text/layers/1/b",
                    "text/layers/1/a",
                ]
                .map(str::to_owned)
                .to_vec(),
            }
        }
    }

    impl Sources for Fake {
        fn base(&self) -> QuantOption {
            QuantOption::Trellis3_0
        }

        fn projections(&self) -> &[String] {
            &self.projections
        }

        fn shape(&self, _option: QuantOption, object: &str) -> Result<Shape, MixedError> {
            let output = if object.ends_with('a') { 64 } else { 128 };
            Ok(Shape { input: 64, output })
        }

        fn bytes(&self, option: QuantOption, _object: &str) -> Result<i64, MixedError> {
            Ok(if option == QuantOption::Nvfp4 { 100 } else { 300 })
        }
    }

    #[derive(Debug, PartialEq)]
    enum Call {
        Load(QuantOption, String, usize, u32),
        Run(u32, Vec<usize>),
        Time(u32, Vec<usize>),
    }

    /// Records the calls; a timed run takes `step` microseconds for each launch plus `fixed` for the whole run.
    struct Recorder {
        calls: Vec<Call>,
        fixed: Duration,
        step: Duration,
        fail_at_time: Option<usize>,
        timed: usize,
    }

    impl Recorder {
        fn new() -> Recorder {
            Recorder {
                calls: Vec::new(),
                fixed: Duration::ZERO,
                step: Duration::from_micros(10),
                fail_at_time: None,
                timed: 0,
            }
        }
    }

    impl Bench for Recorder {
        fn load(
            &mut self,
            option: QuantOption,
            object: &str,
            _shape: Shape,
            copies: usize,
            max_rows: u32,
        ) -> Result<(), MixedError> {
            self.calls.push(Call::Load(option, object.to_owned(), copies, max_rows));
            Ok(())
        }

        fn run(&mut self, rows: u32, order: &[usize]) -> Result<(), MixedError> {
            self.calls.push(Call::Run(rows, order.to_vec()));
            Ok(())
        }

        fn time(&mut self, rows: u32, order: &[usize]) -> Result<Duration, MixedError> {
            self.calls.push(Call::Time(rows, order.to_vec()));
            self.timed += 1;
            if self.fail_at_time == Some(self.timed) {
                return Err(MixedError::invalid("the launch failed"));
            }
            Ok(self.fixed + self.step * u32::try_from(order.len()).unwrap())
        }
    }

    fn plan(options: &[QuantOption], rows: &[u32], repeats: u32, rotate: u64) -> Plan {
        Plan::new(options.iter().copied(), rows.to_vec(), repeats, rotate).unwrap()
    }

    fn measured(bench: &mut Recorder, plan: &Plan) -> Result<CostTable, MixedError> {
        measure(&Fake::new(), plan, bench, &mut |_| {})
    }

    #[test]
    fn a_run_not_hidden_is_repeated_behind_a_gate_twice_as_long_until_it_is() {
        let mut seen = Vec::new();
        let result = hide_launches(4, 8192, |copies| {
            seen.push(copies);
            Ok((seen.len() == 3).then_some(Duration::from_micros(7)))
        });
        assert_eq!(result.unwrap(), (Duration::from_micros(7), 16));
        assert_eq!(seen, [4, 8, 16]);
    }

    #[test]
    fn launches_that_are_never_hidden_are_refused_at_the_bound_naming_the_cause() {
        let mut seen = Vec::new();
        let error = hide_launches(4, 32, |copies| {
            seen.push(copies);
            Ok(None)
        })
        .unwrap_err()
        .to_string();
        assert_eq!(seen, [4, 8, 16, 32]);
        assert!(error.contains("outran a gate of 32 copies"), "{error}");
        assert!(error.contains("fewer --repeats"), "{error}");
    }

    #[test]
    fn a_gate_of_no_copies_is_refused_before_any_attempt() {
        let error = hide_launches(0, 32, |_| panic!("attempted")).unwrap_err().to_string();
        assert!(error.contains("0 copies"), "{error}");
    }

    #[test]
    fn an_attempts_error_passes_through_and_ends_the_loop() {
        let mut attempts = 0;
        let error = hide_launches(4, 8192, |_| {
            attempts += 1;
            if attempts == 2 {
                return Err(MixedError::invalid("the span was lost"));
            }
            Ok(None)
        })
        .unwrap_err();
        assert_eq!(error.to_string(), MixedError::invalid("the span was lost").to_string());
        assert_eq!(attempts, 2);
    }

    #[test]
    fn rows_up_to_eight_are_launched_as_the_engine_launches_them() {
        let mode = |rows| {
            let mode = launch_mode(rows);
            (mode.row_exact, mode.prefill)
        };
        assert_eq!(mode(1), (false, false));
        for rows in 2..=VERIFY_MAX_ROWS {
            assert_eq!(mode(rows), (true, false), "{rows} rows");
        }
        assert_eq!(mode(9), (false, true));
        assert_eq!(mode(512), (false, true));
    }

    #[test]
    fn each_distinct_shape_is_measured_once_on_its_first_object_for_each_option_in_order() {
        let mut bench = Recorder::new();
        let both = plan(&[QuantOption::Trellis3_0, QuantOption::Nvfp4], &[1, 4], 2, 0);
        measured(&mut bench, &both).unwrap();
        let loads: Vec<(QuantOption, &str)> = bench
            .calls
            .iter()
            .filter_map(|call| match call {
                Call::Load(option, object, _, _) => Some((*option, object.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            loads,
            [
                (QuantOption::Nvfp4, "text/layers/0/a"),
                (QuantOption::Trellis3_0, "text/layers/0/a"),
                (QuantOption::Nvfp4, "text/layers/0/b"),
                (QuantOption::Trellis3_0, "text/layers/0/b"),
            ]
        );
    }

    #[test]
    fn the_table_has_an_entry_for_every_shape_option_and_row_count_and_no_other() {
        let mut bench = Recorder::new();
        let table = measured(
            &mut bench,
            &plan(&[QuantOption::Nvfp4, QuantOption::Trellis4_0], &[1, 4, 512], 3, 0),
        )
        .unwrap();
        let text = table.render();
        let read = CostTable::parse(&text).unwrap();
        assert_eq!(read, table);
        let mut keys = 0;
        for output in [64, 128] {
            for option in [QuantOption::Nvfp4, QuantOption::Trellis4_0] {
                for rows in [1, 4, 512] {
                    let key = Key {
                        shape: Shape { input: 64, output },
                        option,
                        rows,
                    };
                    assert_eq!(read.micros(key).unwrap(), 10.0, "{key:?}");
                    keys += 1;
                }
            }
        }
        assert_eq!(text.matches("\"micros\"").count(), keys);
    }

    #[test]
    fn the_cost_of_a_launch_is_the_device_time_over_the_repeats() {
        let mut bench = Recorder::new();
        bench.step = Duration::ZERO;
        bench.fixed = Duration::from_micros(1000);
        let table = measured(&mut bench, &plan(&[QuantOption::Nvfp4], &[1], 8, 0)).unwrap();
        let key = Key {
            shape: Shape { input: 64, output: 64 },
            option: QuantOption::Nvfp4,
            rows: 1,
        };
        assert_eq!(table.micros(key).unwrap(), 125.0);
    }

    #[test]
    fn every_copy_is_warmed_once_and_the_timed_launches_rotate_over_them() {
        let mut bench = Recorder::new();
        // 100 bytes of NVFP4 payload and 250 to span: 3 copies.
        measured(&mut bench, &plan(&[QuantOption::Nvfp4], &[2, 512], 7, 250)).unwrap();
        let first = &bench.calls[..5];
        assert_eq!(
            first[0],
            Call::Load(QuantOption::Nvfp4, "text/layers/0/a".to_owned(), 3, 512)
        );
        assert_eq!(first[1], Call::Run(2, vec![0, 1, 2]));
        assert_eq!(first[2], Call::Time(2, vec![0, 1, 2, 0, 1, 2, 0]));
        assert_eq!(first[3], Call::Run(512, vec![0, 1, 2]));
        assert_eq!(first[4], Call::Time(512, vec![0, 1, 2, 0, 1, 2, 0]));
    }

    #[test]
    fn an_object_bigger_than_the_rotation_is_one_copy() {
        assert_eq!(copies_for(100, 0).unwrap(), 1);
        assert_eq!(copies_for(100, 100).unwrap(), 1);
        assert_eq!(copies_for(100, 101).unwrap(), 2);
        assert!(copies_for(0, 100).is_err());
        assert!(copies_for(-5, 100).is_err());
    }

    #[test]
    fn every_measurement_is_reported_as_it_ends() {
        let mut bench = Recorder::new();
        let mut seen = Vec::new();
        measure(
            &Fake::new(),
            &plan(&[QuantOption::Nvfp4], &[1, 4], 1, 0),
            &mut bench,
            &mut |measured| seen.push((measured.key.shape.output, measured.key.rows, measured.micros)),
        )
        .unwrap();
        assert_eq!(seen, [(64, 1, 10.0), (64, 4, 10.0), (128, 1, 10.0), (128, 4, 10.0)]);
    }

    #[test]
    fn a_failing_launch_ends_the_run_with_no_table() {
        let mut bench = Recorder::new();
        bench.fail_at_time = Some(3);
        let mut seen = 0;
        let failure = measure(
            &Fake::new(),
            &plan(&[QuantOption::Nvfp4], &[1, 4], 2, 0),
            &mut bench,
            &mut |_| seen += 1,
        )
        .unwrap_err();
        assert_eq!(failure, MixedError::invalid("the launch failed"));
        assert_eq!((seen, bench.timed), (2, 3));
    }

    #[test]
    fn a_source_of_another_format_than_its_option_is_refused() {
        assert!(check_format(QuantOption::Nvfp4, WeightFormat::Nvfp4).is_ok());
        for option in [
            QuantOption::Trellis2_5,
            QuantOption::Trellis3_0,
            QuantOption::Trellis3_5,
            QuantOption::Trellis4_0,
        ] {
            assert!(check_format(option, WeightFormat::Trellis).is_ok(), "{option}");
            assert!(check_format(option, WeightFormat::Nvfp4).is_err(), "{option}");
        }
        let failure = check_format(QuantOption::Nvfp4, WeightFormat::Trellis).unwrap_err();
        assert!(failure.to_string().contains("nvfp4"), "{failure}");
    }

    #[test]
    fn a_plan_with_nothing_to_measure_is_refused() {
        let ok = [QuantOption::Nvfp4];
        assert!(Plan::new([], vec![1], 1, 0).is_err());
        assert!(Plan::new(ok, vec![], 1, 0).is_err());
        assert!(Plan::new(ok, vec![1, 0], 1, 0).is_err());
        assert!(Plan::new(ok, vec![1], 0, 0).is_err());
        assert!(Plan::new(ok, vec![1], 1, 0).is_ok());
    }
}
