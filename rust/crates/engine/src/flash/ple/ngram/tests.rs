#[allow(
    clippy::disallowed_types,
    reason = "the test thread, not a worker, holds the pending reads"
)]
use std::sync::Mutex;

use super::*;

fn ids(eos: i32) -> NgramIds {
    NgramIds::new(3, 1, &[3, 5, 7], &[100, 1000], &[0, 100], eos).unwrap()
}

#[test]
fn a_fresh_context_reads_end_of_sequence_before_the_first_token() {
    let ids = ids(9);
    let context = ids.fresh_context();
    assert_eq!(&*context, &[9, 9]);
    let (rows, next) = ids.compute(&context, &[1, 2, 3]).unwrap();
    assert_eq!(rows, [46, 117, 3, 160, 3, 104]);
    assert_eq!(&*next, &[2, 3]);
}

#[test]
fn the_context_carries_the_last_tokens_into_the_next_call() {
    let ids = ids(9);
    let (_, context) = ids.compute(&ids.fresh_context(), &[1, 2, 3]).unwrap();
    let (rows, next) = ids.compute(&context, &[4]).unwrap();
    assert_eq!(rows, [3, 113]);
    assert_eq!(&*next, &[3, 4]);
    // Two calls give the ids of one call over the same tokens.
    let (whole, _) = ids.compute(&ids.fresh_context(), &[1, 2, 3, 4]).unwrap();
    let (first, _) = ids.compute(&ids.fresh_context(), &[1, 2, 3]).unwrap();
    assert_eq!([first, rows].concat(), whole);
}

#[test]
fn an_end_of_sequence_token_stops_the_tokens_before_it_from_reaching_past_it() {
    let ids = ids(9);
    let (rows, _) = ids.compute(&[1, 2], &[5, 9, 6]).unwrap();
    // The last token sees the end-of-sequence token before it as both of its predecessors; the end-of-sequence token
    // itself still reads the token before it.
    assert_eq!(rows, [5, 102, 2, 112, 63, 100]);
}

#[test]
fn the_products_wrap_at_64_bits_and_the_remainder_is_never_negative() {
    let ids = NgramIds::new(2, 1, &[i64::MAX, 3], &[1000], &[0], 7).unwrap();
    let (rows, _) = ids.compute(&[7], &[2]).unwrap();
    assert_eq!(rows, [979]);
}

#[test]
fn a_context_of_the_wrong_length_is_refused_with_its_length() {
    let error = ids(9).compute(&[1], &[1]).unwrap_err().to_string();
    assert!(error.contains("1 tokens"), "{error}");
}

#[test]
fn ids_the_table_cannot_address_are_refused_at_construction() {
    assert!(NgramIds::new(1, 1, &[3], &[], &[], 0).is_err());
    assert!(NgramIds::new(3, 1, &[3, 5], &[100, 1000], &[0, 0], 0).is_err());
    assert!(NgramIds::new(3, 1, &[3, 5, 7], &[100, 0], &[0, 0], 0).is_err());
    assert!(NgramIds::new(3, 1, &[3, 5, 7], &[100, 1000], &[0, -1], 0).is_err());
}

#[test]
fn the_parts_of_every_gather_cover_its_rows_once_in_order() {
    for count in [0, 1, 63, 64, 127, 128, 511, 512, 513, 10_000] {
        let joined: Vec<usize> = (0..PARTS + 2).flat_map(|part| part_range(count, part)).collect();
        assert_eq!(joined, (0..count).collect::<Vec<_>>(), "count {count}");
    }
    let counts: Vec<_> = [0, 63, 64, 128, 512, 10_000].map(part_count).to_vec();
    assert_eq!(counts, [1, 1, 1, 2, 8, 8]);
    assert!(part_range(100, 1).is_empty(), "a part past the count is empty");
}

fn shard_bytes(rows: u64, tag: u8) -> Vec<u8> {
    let layout = NgramLayout::new(rows, 32).unwrap();
    let mut bytes = vec![0u8; layout.byte_size() as usize];
    for row in 0..rows as usize {
        for byte in 0..18 {
            bytes[row * 18 + byte] = tag + (row * 10 + byte) as u8 % 100;
        }
    }
    bytes
}

fn shard(rows: u64, tag: u8) -> Arc<dyn ShardBytes> {
    Arc::new(MemoryShard(shard_bytes(rows, tag)))
}

fn table() -> Arc<NgramRows> {
    Arc::new(NgramRows::new(vec![shard(5, 100), shard(3, 0)], 5, 8, 32, vec![2.0, 0.5]).unwrap())
}

fn record(tag: u8, row: usize, scale: f32) -> Vec<u8> {
    let mut record: Vec<u8> = (0..18).map(|byte| tag + (row * 10 + byte) as u8 % 100).collect();
    record.extend([0, 0]);
    record.extend(scale.to_le_bytes());
    record
}

#[test]
fn a_record_is_the_packed_row_zero_padding_and_the_shards_scale() {
    let table = table();
    assert_eq!(table.record_bytes(), 24);
    let out = table.gather_in_memory(&[7, 0, 4, 5], 1..4).unwrap();
    assert_eq!(out[..24], record(100, 0, 2.0)[..]);
    assert_eq!(out[24..48], record(100, 4, 2.0)[..]);
    assert_eq!(out[48..], record(0, 0, 0.5)[..]);
}

#[test]
fn a_range_gathers_the_same_records_as_the_whole() {
    let table = table();
    let ids = [7, 0, 4, 5, 2, 6];
    let whole = table.gather_in_memory(&ids, 0..6).unwrap();
    let pieces: Vec<u8> = [0..2, 2..3, 3..6]
        .into_iter()
        .flat_map(|chunk| table.gather_in_memory(&ids, chunk).unwrap())
        .collect();
    assert_eq!(whole, pieces);
}

#[test]
fn a_gather_the_table_cannot_serve_is_refused_with_the_row_and_starts_nothing() {
    let table = table();
    let error = table.gather_in_memory(&[8], 0..1).unwrap_err().to_string();
    assert!(error.contains("row 8 of 8"), "{error}");
    assert!(table.gather_in_memory(&[1], 0..2).is_err());
    assert_eq!(
        table.gather_in_memory(&[1], 1..1).unwrap(),
        Vec::<u8>::new(),
        "no rows end at once"
    );
}

#[test]
fn a_table_whose_shards_do_not_hold_its_rows_is_refused() {
    assert!(
        NgramRows::new(vec![shard(5, 0)], 5, 8, 32, vec![1.0]).is_err(),
        "one shard of two"
    );
    let short: Arc<dyn ShardBytes> = Arc::new(MemoryShard(vec![0u8; 100]));
    assert!(NgramRows::new(vec![short, shard(3, 0)], 5, 8, 32, vec![1.0; 2]).is_err());
    assert!(NgramRows::new(vec![shard(5, 0)], 5, 5, 24, vec![1.0]).is_err());
    let scales = NgramRows::new(vec![shard(5, 0)], 5, 5, 32, vec![])
        .unwrap_err()
        .to_string();
    assert!(scales.contains("0 scales for 1 shards"), "{scales}");
}

#[test]
fn the_extent_of_a_shard_is_its_size_and_where_its_scale_is() {
    let layout = NgramLayout::new(3, 32).unwrap();
    let (size, scale_at) = NgramRows::shard_extent(1, 5, 8, 32).unwrap();
    assert_eq!((size, scale_at), (layout.byte_size(), layout.trailer_offset()));
    assert!(NgramRows::shard_extent(2, 5, 8, 32).is_err(), "past the table");
}

/// A shard whose reads end when the test says so, in the order it says.
#[derive(Default)]
#[allow(
    clippy::disallowed_types,
    reason = "the test thread, not a worker, holds the pending reads"
)]
struct Deferred {
    bytes: Vec<u8>,
    refuse: bool,
    pending: Mutex<Vec<(u64, usize, RowRead)>>,
}

impl fmt::Debug for Deferred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deferred").finish_non_exhaustive()
    }
}

impl ShardBytes for Deferred {
    fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn start(&self, offset: u64, len: usize, then: RowRead) -> Result<(), String> {
        if self.refuse {
            return Err("the read was refused".to_owned());
        }
        self.pending.lock().unwrap().push((offset, len, then));
        Ok(())
    }
}

impl Deferred {
    fn holding(bytes: Vec<u8>) -> Arc<Deferred> {
        Arc::new(Deferred {
            bytes,
            ..Deferred::default()
        })
    }

    fn pending(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Ends the `index`-th pending read (counted as they were started) with the shard's bytes, or with `failure`.
    fn end(&self, index: usize, failure: Option<&str>) {
        let (offset, len, then) = self.pending.lock().unwrap().remove(index);
        let result = match failure {
            Some(why) => Err(why.to_owned()),
            None => Ok(self.bytes[offset as usize..offset as usize + len].to_vec()),
        };
        then(result).execute().unwrap();
    }
}

/// What a gather told, once.
#[derive(Default)]
#[allow(
    clippy::disallowed_types,
    reason = "the test thread reads what the end of the gather wrote"
)]
struct Told(Mutex<Vec<Result<Vec<u8>, FlashError>>>);

impl Told {
    fn done(self: &Arc<Self>) -> GatherDone {
        let told = Arc::clone(self);
        Box::new(move |result| told.0.lock().unwrap().push(result))
    }

    fn count(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

#[test]
fn a_gather_ends_once_when_the_last_row_ends_whatever_the_order_and_keeps_the_rows_in_order() {
    for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0]] {
        let held = Deferred::holding(shard_bytes(5, 100));
        let table =
            Arc::new(NgramRows::new(vec![Arc::clone(&held) as Arc<dyn ShardBytes>], 5, 5, 32, vec![2.0]).unwrap());
        let told = Arc::new(Told::default());
        table.gather(&[4, 0, 2], 0..3, told.done()).unwrap();
        assert_eq!(held.pending(), 3, "every row is started before any ends");
        assert_eq!(told.count(), 0);
        // `order` names the started reads by their place in the queue as it shrinks.
        let mut left = vec![0, 1, 2];
        for (step, place) in order.into_iter().enumerate() {
            let at = left.iter().position(|row| *row == place).unwrap();
            left.remove(at);
            assert_eq!(told.count(), 0, "{order:?} step {step}");
            held.end(at, None);
        }
        assert_eq!(told.count(), 1, "{order:?}");
        let out = told.0.lock().unwrap().remove(0).unwrap();
        let expected: Vec<u8> = [4, 0, 2].into_iter().flat_map(|row| record(100, row, 2.0)).collect();
        assert_eq!(out, expected, "{order:?}");
    }
}

#[test]
fn a_row_that_fails_fails_the_gather_after_the_others_have_ended_and_names_the_row_and_the_cause() {
    let held = Deferred::holding(shard_bytes(5, 100));
    let table = Arc::new(NgramRows::new(vec![Arc::clone(&held) as Arc<dyn ShardBytes>], 5, 5, 32, vec![2.0]).unwrap());
    let told = Arc::new(Told::default());
    table.gather(&[1, 2, 3], 0..3, told.done()).unwrap();
    held.end(1, Some("the disk went away"));
    held.end(0, None);
    assert_eq!(told.count(), 0, "a row is still out");
    held.end(0, None);
    assert_eq!(told.count(), 1);
    let error = told.0.lock().unwrap().remove(0).unwrap_err().to_string();
    assert!(
        error.contains("row 1") && error.contains("the disk went away"),
        "{error}"
    );
}

#[test]
fn a_row_that_cannot_start_fails_the_gather_once_and_ends_it_in_the_call() {
    let refusing = Arc::new(Deferred {
        bytes: shard_bytes(5, 0),
        refuse: true,
        ..Deferred::default()
    });
    let table = Arc::new(NgramRows::new(vec![refusing as Arc<dyn ShardBytes>], 5, 5, 32, vec![1.0]).unwrap());
    let told = Arc::new(Told::default());
    table.gather(&[0, 1], 0..2, told.done()).unwrap();
    assert_eq!(told.count(), 1, "nothing was started, so nothing is out");
    let error = told.0.lock().unwrap().remove(0).unwrap_err().to_string();
    assert!(error.contains("the read was refused"), "{error}");
}

#[test]
fn parts_arrive_in_any_order_and_join_in_row_order() {
    let permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for order in permutations {
        let slots = PartSlots::new(PARTS);
        for part in order {
            slots.publish(part, part * 1000).unwrap();
        }
        let joined = slots.join(192).unwrap();
        let values: Vec<_> = joined.iter().map(|(range, value)| (range.clone(), **value)).collect();
        assert_eq!(values, [(0..64, 0), (64..128, 1000), (128..192, 2000)], "{order:?}");
    }
}

#[test]
fn a_part_that_never_arrived_fails_the_join_by_name() {
    let slots = PartSlots::new(PARTS);
    for part in [0, 1, 2, 4, 5, 6, 7] {
        slots.publish(part, ()).unwrap();
    }
    assert_eq!(slots.join(512).unwrap_err(), 3);
    assert_eq!(
        slots.join(0).map(|joined| joined.len()),
        Ok(0),
        "no rows, no parts to wait for"
    );
    let empty: PartSlots<()> = PartSlots::new(PARTS);
    assert_eq!(empty.join(1).unwrap_err(), 0);
}

#[test]
fn a_part_published_twice_keeps_the_first_and_hands_back_the_second() {
    let slots = PartSlots::new(2);
    slots.publish(1, "first").unwrap();
    let (error, value) = slots.publish(1, "second").unwrap_err();
    assert_eq!((error, value), (PublishError::Twice { part: 1 }, "second"));
    assert_eq!(slots.get(1), Some(&"first"));
    let (error, _) = slots.publish(2, "late").unwrap_err();
    assert_eq!(error, PublishError::OutOfRange { part: 2, parts: 2 });
}
