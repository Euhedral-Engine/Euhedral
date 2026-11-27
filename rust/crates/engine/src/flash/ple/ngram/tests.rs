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

fn shard(rows: u64, tag: u8, scale: f32) -> Arc<dyn ShardBytes> {
    let layout = NgramLayout::new(rows, 32).unwrap();
    let mut bytes = vec![0u8; layout.byte_size() as usize];
    for row in 0..rows as usize {
        for byte in 0..18 {
            bytes[row * 18 + byte] = tag + (row * 10 + byte) as u8 % 100;
        }
    }
    let at = layout.trailer_offset() as usize;
    bytes[at..at + 4].copy_from_slice(&scale.to_le_bytes());
    Arc::new(bytes)
}

fn table() -> NgramRows {
    NgramRows::new(vec![shard(5, 100, 2.0), shard(3, 0, 0.5)], 5, 8, 32).unwrap()
}

#[test]
fn a_record_is_the_packed_row_zero_padding_and_the_shards_scale() {
    let table = table();
    assert_eq!(table.record_bytes(), 24);
    let mut out = vec![0xAA; 3 * 24];
    table.gather(&[7, 0, 4, 5], 1..4, &mut out).unwrap();
    let record = |tag: u8, row: usize, scale: f32| {
        let mut record: Vec<u8> = (0..18).map(|byte| tag + (row * 10 + byte) as u8 % 100).collect();
        record.extend([0, 0]);
        record.extend(scale.to_le_bytes());
        record
    };
    assert_eq!(out[..24], record(100, 0, 2.0)[..]);
    assert_eq!(out[24..48], record(100, 4, 2.0)[..]);
    assert_eq!(out[48..], record(0, 0, 0.5)[..]);
}

#[test]
fn a_range_gathers_the_same_records_as_the_whole() {
    let table = table();
    let ids = [7, 0, 4, 5, 2, 6];
    let mut whole = vec![0; 6 * 24];
    table.gather(&ids, 0..6, &mut whole).unwrap();
    let mut pieces = vec![0; 6 * 24];
    for (part, chunk) in [0..2, 2..3, 3..6].into_iter().enumerate() {
        let _ = part;
        let (from, to) = (chunk.start * 24, chunk.end * 24);
        table.gather(&ids, chunk, &mut pieces[from..to]).unwrap();
    }
    assert_eq!(whole, pieces);
}

#[test]
fn a_gather_the_table_cannot_serve_is_refused_with_the_row() {
    let table = table();
    let mut out = vec![0; 24];
    let error = table.gather(&[8], 0..1, &mut out).unwrap_err().to_string();
    assert!(error.contains("row 8 of 8"), "{error}");
    assert!(table.gather(&[1], 0..1, &mut vec![0; 23]).is_err());
    assert!(table.gather(&[1], 0..2, &mut vec![0; 48]).is_err());
}

#[test]
fn a_table_whose_shards_do_not_hold_its_rows_is_refused() {
    assert!(NgramRows::new(vec![shard(5, 0, 1.0)], 5, 8, 32).is_err());
    let short: Arc<dyn ShardBytes> = Arc::new(vec![0u8; 100]);
    assert!(NgramRows::new(vec![short, shard(3, 0, 1.0)], 5, 8, 32).is_err());
    assert!(NgramRows::new(vec![shard(5, 0, 1.0)], 5, 5, 24).is_err());
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

#[test]
fn two_publishers_of_one_part_leave_exactly_one_winner() {
    let slots = PartSlots::new(1);
    let results = std::thread::scope(|scope| {
        let first = scope.spawn(|| slots.publish(0, 1).is_ok());
        let second = scope.spawn(|| slots.publish(0, 2).is_ok());
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert_eq!(results.iter().filter(|won| **won).count(), 1);
    let kept = *slots.get(0).unwrap();
    assert_eq!(results[kept - 1], true, "the kept value is the winner's");
}
