use super::*;

/// A deterministic stream of values (xorshift).
struct Stream(u64);

impl Stream {
    fn next(&mut self, below: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % u64::from(below)) as u32
    }
}

/// A router output for `rows` rows: `top_k` distinct experts per row (some padding) and arbitrary weights.
fn router_output(stream: &mut Stream, rows: u32, experts: u32, top_k: u32, padding: bool) -> (Vec<i32>, Vec<u16>) {
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for _ in 0..rows {
        let mut row: Vec<u32> = Vec::new();
        while row.len() < top_k as usize {
            let pick = if padding && stream.next(4) == 0 {
                experts
            } else {
                stream.next(experts)
            };
            if pick == experts || !row.contains(&pick) {
                row.push(pick);
            }
        }
        ids.extend(row.iter().map(|&id| id as i32));
        weights.extend((0..top_k).map(|_| stream.next(65536) as u16));
    }
    (ids, weights)
}

/// What the description holds, decoded from its words.
struct Decoded {
    items: Vec<[u32; 4]>,
    pairs: Vec<(u32, u32)>,
    row_offsets: Vec<u32>,
    row_pairs: Vec<u32>,
}

fn written(routing: &mut Routing) -> Vec<u32> {
    let mut words = vec![0xdead_beef_u32; routing.layout().bytes() / 4];
    routing.fill(&mut |index, value| words[index] = value);
    words
}

fn decode(routing: &Routing, words: &[u32], items: usize, pairs: usize) -> Decoded {
    let layout = routing.layout();
    let at = |offset: usize| offset / 4;
    Decoded {
        items: (0..items)
            .map(|i| std::array::from_fn(|f| words[at(layout.items_offset() + 16 * i) + f]))
            .collect(),
        pairs: (0..pairs)
            .map(|p| {
                let base = at(layout.pairs_offset() + 8 * p);
                (words[base], words[base + 1])
            })
            .collect(),
        row_offsets: (0..=layout.max_rows() as usize)
            .map(|r| words[at(layout.row_offsets_offset() + 4 * r)])
            .collect(),
        row_pairs: (0..pairs)
            .map(|p| words[at(layout.row_pairs_offset() + 4 * p)])
            .collect(),
    }
}

/// The grouping by sorting: pairs by expert, stably (so ascending row, then position), the items of 16, and each row's
/// list of pair indices.
fn reference(
    rows: u32,
    experts: u32,
    top_k: u32,
    ids: &[i32],
    weights: &[u16],
) -> (Vec<u32>, Vec<[u32; 4]>, Vec<(u32, u32)>, Vec<Vec<u32>>) {
    let mut sorted: Vec<(u32, u32, u32)> = (0..(rows * top_k) as usize)
        .filter(|&i| ids[i] as u32 != experts)
        .map(|i| (ids[i] as u32, i as u32 / top_k, u32::from(weights[i])))
        .collect();
    sorted.sort_by_key(|pair| pair.0);
    let mut active: Vec<u32> = sorted.iter().map(|pair| pair.0).collect();
    active.dedup();
    let mut items = Vec::new();
    for (slot, expert) in active.iter().enumerate() {
        let first = sorted.iter().position(|pair| pair.0 == *expert).unwrap() as u32;
        let count = sorted.iter().filter(|pair| pair.0 == *expert).count() as u32;
        for group in (0..count).step_by(16) {
            items.push([slot as u32, first + group, 16.min(count - group), 0]);
        }
    }
    let pairs = sorted.iter().map(|pair| (pair.1, pair.2)).collect();
    let lists = (0..rows)
        .map(|row| {
            (0..sorted.len() as u32)
                .filter(|&p| sorted[p as usize].1 == row)
                .collect()
        })
        .collect();
    (active, items, pairs, lists)
}

#[test]
fn the_layout_places_each_section_where_javas_does() {
    // 512 experts, 10 per row, up to 8 rows: 80 pairs, 80 active experts, 85 item slots.
    let layout = DescriptorLayout::new(512, 10, 8).unwrap();
    assert_eq!(
        (
            layout.max_pairs(),
            layout.max_active(),
            layout.items_offset(),
            layout.pairs_offset(),
            layout.row_offsets_offset(),
            layout.row_pairs_offset(),
            layout.bytes()
        ),
        (80, 80, 640, 2000, 2640, 2688, 3008)
    );
    // Fewer pairs than experts bound the active experts; more experts than pairs do not.
    assert_eq!(DescriptorLayout::new(4, 10, 8).unwrap().max_active(), 4);
}

#[test]
fn the_layout_refuses_a_zero_extent_and_an_overflowing_block() {
    for (experts, top_k, rows) in [(0, 10, 8), (512, 0, 8), (512, 10, 0)] {
        assert!(
            DescriptorLayout::new(experts, top_k, rows)
                .unwrap_err()
                .to_string()
                .contains("every extent is positive")
        );
    }
    assert!(
        DescriptorLayout::new(512, 16, u32::MAX)
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
}

#[test]
fn the_plan_and_the_description_match_a_host_reference_over_many_blocks() {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
    for (experts, top_k, max_rows) in [(512, 10, 8), (16, 4, 40), (8, 8, 3), (64, 10, 33)] {
        let mut routing = Routing::new(DescriptorLayout::new(experts, top_k, max_rows).unwrap());
        for rows in [1, 2, max_rows / 2 + 1, max_rows] {
            let (ids, weights) = router_output(&mut stream, rows, experts, top_k, true);
            routing.plan(rows, |i| ids[i], |i| weights[i]).unwrap();
            let (active, items, pairs, lists) = reference(rows, experts, top_k, &ids, &weights);
            let words = written(&mut routing);
            let decoded = decode(&routing, &words, items.len(), pairs.len());
            let shape = format!("{experts} experts, {top_k} per row, {rows} of {max_rows} rows");
            assert_eq!(routing.active_experts() as usize, active.len(), "{shape}");
            for (slot, expert) in active.iter().enumerate() {
                assert_eq!(routing.active_expert(slot as u32), Some(*expert), "{shape}");
            }
            assert_eq!(routing.active_expert(active.len() as u32), None, "{shape}");
            assert_eq!(decoded.items, items, "{shape}");
            assert_eq!(routing.pair_count() as usize, pairs.len(), "{shape}");
            assert_eq!(decoded.pairs, pairs, "{shape}");
            // Each row's list of pair indices, ascending, and the offsets that delimit them (past the block: the total).
            for (row, list) in lists.iter().enumerate() {
                let (from, to) = (decoded.row_offsets[row] as usize, decoded.row_offsets[row + 1] as usize);
                assert_eq!(&decoded.row_pairs[from..to], list.as_slice(), "{shape} row {row}");
            }
            assert!(
                decoded.row_offsets[rows as usize..]
                    .iter()
                    .all(|&offset| offset as usize == pairs.len()),
                "{shape}"
            );
            // An expert's item range is what `item_start` and `item_count` say.
            let mut from = 0;
            for slot in 0..routing.active_experts() {
                assert_eq!(routing.item_start(slot), Some(from), "{shape}");
                from += routing.item_count(slot).unwrap();
            }
            assert_eq!(from as usize, items.len(), "{shape}");
        }
    }
}

#[test]
fn an_expert_with_many_pairs_is_split_into_items_of_sixteen() {
    // 20 rows of one pair each, all to expert 3: two items, 16 and 4 pairs.
    let mut routing = Routing::new(DescriptorLayout::new(8, 1, 20).unwrap());
    routing.plan(20, |_| 3, |i| i as u16).unwrap();
    let words = written(&mut routing);
    let decoded = decode(&routing, &words, 2, 20);
    assert_eq!(decoded.items, vec![[0, 0, 16, 0], [0, 16, 4, 0]]);
    assert_eq!(routing.item_count(0), Some(2));
    // Pairs keep ascending row order, and each row's list holds its one pair.
    assert_eq!(
        decoded.pairs.iter().map(|pair| pair.0).collect::<Vec<_>>(),
        (0..20).collect::<Vec<_>>()
    );
    assert_eq!(decoded.row_pairs, (0..20).collect::<Vec<_>>());
}

#[test]
fn a_block_of_padding_alone_names_no_expert() {
    let mut routing = Routing::new(DescriptorLayout::new(8, 2, 4).unwrap());
    routing.plan(3, |_| 8, |_| 1).unwrap();
    assert_eq!(
        (routing.active_experts(), routing.pair_count(), routing.rows()),
        (0, 0, 3)
    );
    let words = written(&mut routing);
    let decoded = decode(&routing, &words, 0, 0);
    assert_eq!(decoded.row_offsets, vec![0; 5]);
}

#[test]
fn the_slot_words_hold_the_record_address_low_word_first() {
    let layout = DescriptorLayout::new(8, 2, 4).unwrap();
    assert_eq!(layout.slot_offset(3), 24);
    assert_eq!(layout.slot_words(3, 0x0000_0007_89ab_cdef), [(6, 0x89ab_cdef), (7, 7)]);
}

#[test]
fn a_plan_refuses_rows_past_the_layout_and_an_id_outside_the_experts_and_then_holds_none() {
    let mut routing = Routing::new(DescriptorLayout::new(8, 2, 4).unwrap());
    routing.plan(2, |i| (i % 8) as i32, |_| 1).unwrap();
    assert_eq!(routing.active_experts(), 4);
    let message = routing.plan(5, |_| 0, |_| 0).unwrap_err().to_string();
    assert!(message.contains("5 rows exceed the 4"), "{message}");
    assert_eq!((routing.active_experts(), routing.rows()), (0, 0));
    for bad in [-1, 9] {
        let message = routing
            .plan(2, |i| if i == 3 { bad } else { 0 }, |_| 0)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains(&format!("expert id {bad} at pair 3 is outside [0, 8]")),
            "{message}"
        );
        assert_eq!((routing.active_experts(), routing.pair_count()), (0, 0));
    }
}
