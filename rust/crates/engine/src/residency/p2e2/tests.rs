use std::thread;

use super::*;

const K: u64 = 1024;

/// A valid tensor of `rows` rows whose codes come from `code(row, index)` (0 to 3, 3 being BIG), with `spare` words of
/// payload beyond what the BIG codes need.
fn tensor(rows: u64, code: impl Fn(u64, u64) -> u32, spare: u64) -> Vec<u8> {
    let codes = rows * K;
    let row_base = align(codes / 4);
    let scales = align(row_base + 4 * rows);
    let payload = align(scales + codes / 32);
    let mut primary = vec![0u32; (codes / 16) as usize];
    let mut bases = Vec::new();
    let mut units = 0u64;
    for row in 0..rows {
        bases.push(units as u32);
        for index in 0..K {
            let value = code(row, index) & 3;
            primary[(row * K + index) as usize / 16] |= value << (2 * (index % 16));
            units += u64::from(value == 3);
        }
    }
    let words = units.div_ceil(16) + PAD_WORDS + spare;
    let mut bytes = vec![0u8; (payload + 4 * words) as usize];
    for (word, value) in primary.iter().enumerate() {
        bytes[4 * word..4 * word + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (row, base) in bases.iter().enumerate() {
        let at = row_base as usize + 4 * row;
        bytes[at..at + 4].copy_from_slice(&base.to_le_bytes());
    }
    // The scale and payload planes are anything: the check does not read them.
    for (at, byte) in bytes[scales as usize..].iter_mut().enumerate() {
        *byte = at as u8;
    }
    bytes
}

fn pattern(row: u64, index: u64) -> u32 {
    ((row * 7 + index * 13 + index / 5) % 4) as u32
}

/// Feeds `bytes` in pieces of `piece` bytes, in the order `order` picks, and returns the verdict of each piece.
fn in_pieces(bytes: &[u8], rows: u64, piece: usize, reverse: bool) -> Vec<Result<(), String>> {
    let geometry = P2e2Geometry::new(rows, K, bytes.len() as u64).unwrap();
    let validation = P2e2Validation::new(geometry);
    let mut starts: Vec<usize> = (0..bytes.len()).step_by(piece).collect();
    if reverse {
        starts.reverse();
    }
    starts
        .into_iter()
        .map(|start| {
            let end = (start + piece).min(bytes.len());
            let scan = geometry.scan(start as u64, &bytes[start..end])?;
            validation.absorb(scan)
        })
        .collect()
}

#[test]
fn a_good_tensor_is_valid_whole_and_in_pieces_in_any_order() {
    let bytes = tensor(3, pattern, 0);
    assert_eq!(validate(&bytes, 3, K), Ok(()));
    for piece in [4, 260, 512, 1000, 4096] {
        for reverse in [false, true] {
            let verdicts = in_pieces(&bytes, 3, piece, reverse);
            assert!(
                verdicts.iter().all(Result::is_ok),
                "piece {piece} reverse {reverse}: {verdicts:?}"
            );
        }
    }
}

#[test]
fn a_tensor_with_more_payload_than_it_needs_is_valid() {
    assert_eq!(validate(&tensor(2, pattern, 40), 2, K), Ok(()));
    assert_eq!(validate(&tensor(2, |_, _| 3, 0), 2, K), Ok(()));
    assert_eq!(validate(&tensor(2, |_, _| 0, 0), 2, K), Ok(()));
}

#[test]
fn a_row_base_that_does_not_follow_the_primary_plane_is_named_by_its_row() {
    let mut bytes = tensor(3, pattern, 0);
    let row_base = align(3 * K / 4) as usize;
    let at = row_base + 4 * 2;
    let base = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    bytes[at..at + 4].copy_from_slice(&(base + 1).to_le_bytes());
    assert_eq!(
        validate(&bytes, 3, K),
        Err("P2E2 row base of row 2 does not match its primary plane".to_string())
    );
    // A piece cut anywhere reaches the same verdict, at the tensor's last bytes.
    for reverse in [false, true] {
        let verdicts = in_pieces(&bytes, 3, 260, reverse);
        let (last, rest) = verdicts.split_last().unwrap();
        assert!(rest.iter().all(Result::is_ok));
        assert!(last.as_ref().is_err_and(|error| error.contains("row 2")));
    }
}

#[test]
fn a_changed_primary_plane_no_longer_matches_the_row_bases() {
    let mut bytes = tensor(2, |_, _| 0, 0);
    // One BIG code in row 0: row 1's base is then wrong.
    bytes[0] = 0b11;
    assert_eq!(
        validate(&bytes, 2, K),
        Err("P2E2 row base of row 1 does not match its primary plane".to_string())
    );
}

#[test]
fn a_payload_shorter_than_the_primary_plane_needs_is_refused() {
    let full = tensor(2, |_, _| 3, 0);
    // Every code is BIG: 2048 units, 128 words. Drop 8 of them: the size still fits the shape, the codes do not.
    let short = &full[..full.len() - 32];
    assert_eq!(
        validate(short, 2, K),
        Err("P2E2 payload plane is shorter than its primary plane requires".to_string())
    );
}

#[test]
fn a_size_that_does_not_fit_the_shape_is_refused_before_any_byte_is_read() {
    let bytes = tensor(2, pattern, 0);
    assert!(P2e2Geometry::new(2, K, bytes.len() as u64 + 1).is_err());
    assert!(P2e2Geometry::new(2, K, 100).is_err());
    assert!(P2e2Geometry::new(2, 1000, bytes.len() as u64).is_err());
    assert!(P2e2Geometry::new(0, K, bytes.len() as u64).is_err());
    assert_eq!(
        validate(&bytes[..bytes.len() - 1], 2, K),
        Err("P2E2 tensor size does not match its shape".to_string())
    );
}

#[test]
fn a_piece_that_is_not_whole_words_is_refused() {
    let bytes = tensor(2, pattern, 0);
    let geometry = P2e2Geometry::new(2, K, bytes.len() as u64).unwrap();
    assert!(geometry.scan(2, &bytes[2..10]).is_err());
    assert!(geometry.scan(0, &bytes[..6]).is_err());
}

#[allow(
    clippy::disallowed_methods,
    reason = "the test runs real threads against one check to see that its verdict does not depend on their order"
)]
#[test]
fn the_pieces_of_one_tensor_are_verified_at_once() {
    let rows = 6;
    let bytes = tensor(rows, pattern, 0);
    let geometry = P2e2Geometry::new(rows, K, bytes.len() as u64).unwrap();
    let starts: Vec<usize> = (0..bytes.len()).step_by(260).collect();
    for round in 0..50 {
        // The check exists before any piece does, as the loader makes it, and every worker takes pieces of it in turn.
        let validation = P2e2Validation::new(geometry);
        let verdicts = thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|worker| {
                    let (bytes, validation, starts) = (&bytes, &validation, &starts);
                    scope.spawn(move || {
                        starts
                            .iter()
                            .skip(worker)
                            .step_by(4)
                            .map(|&start| {
                                let end = (start + 260).min(bytes.len());
                                let scan = geometry.scan(start as u64, &bytes[start..end])?;
                                validation.absorb(scan)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(verdicts.len(), starts.len());
        assert!(verdicts.iter().all(Result::is_ok), "round {round}: {verdicts:?}");
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "the test runs real threads against one check to see that its verdict does not depend on their order"
)]
#[test]
fn a_bad_tensor_is_judged_by_whichever_piece_lands_last_whatever_the_workers() {
    let rows = 6;
    let mut bytes = tensor(rows, pattern, 0);
    // A row base that does not follow the primary plane.
    let row_base = align(rows * K / 4) as usize;
    bytes[row_base + 8] ^= 1;
    let geometry = P2e2Geometry::new(rows, K, bytes.len() as u64).unwrap();
    let starts: Vec<usize> = (0..bytes.len()).step_by(260).collect();
    for _ in 0..50 {
        let validation = P2e2Validation::new(geometry);
        let failures: usize = thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|worker| {
                    let (bytes, validation, starts) = (&bytes, &validation, &starts);
                    scope.spawn(move || {
                        starts
                            .iter()
                            .skip(worker)
                            .step_by(4)
                            .filter(|&&start| {
                                let end = (start + 260).min(bytes.len());
                                geometry
                                    .scan(start as u64, &bytes[start..end])
                                    .and_then(|scan| validation.absorb(scan))
                                    .is_err()
                            })
                            .count()
                    })
                })
                .collect();
            workers.into_iter().map(|worker| worker.join().unwrap()).sum()
        });
        assert_eq!(failures, 1, "exactly the last piece to land judges the tensor");
    }
}
