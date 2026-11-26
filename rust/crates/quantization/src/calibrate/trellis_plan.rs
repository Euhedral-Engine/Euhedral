//! The parts of a trellis layer that touch no device: where a linear's rows sit in the layer's objects, how many lanes
//! the card holds, and the result in the shape the writer takes.
//!
//! The layer's weights live in fused objects (q, k and v in one, gate and up in one). A linear is rounded from its own
//! `[n, k]` buffer, gathered from the object, and its decoded rows go back to where they came from.

use super::CalibrateError;
use crate::exl3::{Quantized as Written, linears::Placement};

/// BF16 bytes of a weight.
const BF16_BYTES: u64 = 2;

/// A run of rows moved between a linear's own buffer and the object that holds them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowCopy {
    /// The object's name.
    pub object: String,
    /// Byte offset of the first row in the object.
    pub object_offset: u64,
    /// Byte offset of the first row in the linear's own buffer.
    pub linear_offset: u64,
    pub bytes: u64,
}

/// The copies that gather (or, the other way, scatter) the `rows` rows of a linear of `columns` columns, from its
/// placements. Refused unless the placements cover every row exactly once.
pub fn row_copies(placements: &[Placement], rows: u64, columns: u64) -> Result<Vec<RowCopy>, CalibrateError> {
    let row_bytes = BF16_BYTES * columns;
    let mut covered =
        vec![false; usize::try_from(rows).map_err(|_| CalibrateError::Invalid("too many rows".to_owned()))?];
    let mut copies = Vec::with_capacity(placements.len());
    for placement in placements {
        let end = placement.linear_row.checked_add(placement.rows);
        let Some(end) = end.filter(|end| *end <= rows) else {
            return Err(CalibrateError::Failed(format!(
                "{} rows from row {} lie outside a linear of {rows} rows",
                placement.rows, placement.linear_row
            )));
        };
        for row in placement.linear_row..end {
            let seen = &mut covered[row as usize];
            if *seen {
                return Err(CalibrateError::Failed(format!("row {row} of a linear is placed twice")));
            }
            *seen = true;
        }
        copies.push(RowCopy {
            object: placement.object.clone(),
            object_offset: placement.object_row * row_bytes,
            linear_offset: placement.linear_row * row_bytes,
            bytes: placement.rows * row_bytes,
        });
    }
    match covered.iter().position(|seen| !seen) {
        Some(row) => Err(CalibrateError::Failed(format!("row {row} of a linear is in no object"))),
        None => Ok(copies),
    }
}

/// How many linears the card rounds at once: as many as fit in `device` bytes and in `pinned` bytes when each takes
/// `peak_device` and `peak_pinned`, at most `most`. None when not even one fits.
pub fn lanes_that_fit(
    (peak_device, peak_pinned): (u64, u64),
    (device, pinned): (u64, u64),
    most: usize,
) -> Option<usize> {
    let by = |capacity: u64, peak: u64| if peak == 0 { u64::MAX } else { capacity / peak };
    let fit = by(device, peak_device).min(by(pinned, peak_pinned));
    let fit = usize::try_from(fit.min(most as u64)).unwrap_or(most);
    (fit >= 1).then_some(fit)
}

/// The bytes a layer holds on the card while it is rounded, besides its own weights and Hessians: a gathered copy of
/// each quantized linear's weight and its decoded result, from the linears' `numel`.
pub fn held_bytes(numel: impl IntoIterator<Item = u64>) -> u64 {
    numel.into_iter().map(|numel| 2 * BF16_BYTES * numel).sum()
}

/// The result the writer takes: the packed words as the signed 16-bit values the file stores.
pub fn written(trellis: Vec<u16>, suh: Vec<u16>, svh: Vec<u16>) -> Written {
    Written {
        trellis: trellis.into_iter().map(|word| word as i16).collect(),
        suh,
        svh,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn placement(object: &str, object_row: u64, linear_row: u64, rows: u64) -> Placement {
        Placement {
            object: object.to_owned(),
            object_row,
            linear_row,
            rows,
        }
    }

    #[test]
    fn a_linear_in_the_middle_of_an_object_is_copied_from_its_rows() {
        // k = 4: a row is 8 bytes. The linear's 3 rows are rows 5..8 of the object.
        let copies = row_copies(&[placement("o", 5, 0, 3)], 3, 4).unwrap();
        assert_eq!(
            copies,
            [RowCopy {
                object: "o".to_owned(),
                object_offset: 40,
                linear_offset: 0,
                bytes: 24
            }]
        );
    }

    #[test]
    fn a_linear_split_over_objects_is_copied_piece_by_piece_to_its_own_rows() {
        let copies = row_copies(&[placement("a", 0, 0, 2), placement("b", 7, 2, 2)], 4, 4).unwrap();
        assert_eq!(
            (
                copies[1].object.as_str(),
                copies[1].object_offset,
                copies[1].linear_offset
            ),
            ("b", 56, 16)
        );
        assert_eq!(copies.iter().map(|copy| copy.bytes).sum::<u64>(), 32);
    }

    #[test]
    fn placements_that_leave_a_row_out_or_place_one_twice_or_run_past_the_linear_are_refused() {
        assert!(
            row_copies(&[placement("o", 0, 0, 2)], 3, 4).is_err(),
            "row 2 is nowhere"
        );
        assert!(
            row_copies(&[placement("o", 0, 0, 2), placement("p", 0, 1, 2)], 3, 4).is_err(),
            "row 1 twice"
        );
        assert!(row_copies(&[placement("o", 0, 2, 2)], 3, 4).is_err(), "past the end");
        assert!(row_copies(&[], 3, 4).is_err(), "a linear no object holds");
    }

    #[test]
    fn the_lanes_that_fit_are_bound_by_the_scarcer_memory_and_by_the_most() {
        assert_eq!(lanes_that_fit((10, 1), (35, 100), 8), Some(3));
        assert_eq!(lanes_that_fit((10, 10), (100, 25), 8), Some(2));
        assert_eq!(lanes_that_fit((10, 0), (1000, 0), 4), Some(4));
        assert_eq!(lanes_that_fit((10, 1), (9, 100), 8), None, "not one fits");
    }

    #[test]
    fn a_layer_holds_a_gathered_and_a_decoded_copy_of_each_weight() {
        assert_eq!(held_bytes([10, 5]), 60);
        assert_eq!(held_bytes([]), 0);
    }

    #[test]
    fn the_words_are_written_as_the_same_bits_read_signed() {
        let out = written(vec![0xFFFF, 0x8000, 1], vec![7], vec![9, 9]);
        assert_eq!(out.trellis, [-1, i16::MIN, 1]);
        assert_eq!((out.suh, out.svh), (vec![7], vec![9, 9]));
    }
}
