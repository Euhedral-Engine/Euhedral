//! Where each part of the parameters lies in the state files. A file is a header, then sections in a fixed order, each
//! cut into pieces of at most a chunk so that no piece is larger than the lane reads at once.
//!
//! The state file (`resume.bin`) holds, for every parameter in order, its values, its first and its second moment (the
//! order Java's state has), then every parameter's gradient, which Java does not save: a state taken between two
//! updates has gradients that the next update needs. The best file holds the values only.

use crate::resume::HEADER;

/// One of the four arrays a parameter has on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Values,
    First,
    Second,
    Gradient,
}

/// Which file a layout is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum File {
    /// Everything a run continues from.
    State,
    /// The best parameters: their values.
    Best,
}

/// A piece of one parameter's array and where it lies in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub parameter: usize,
    pub part: Part,
    /// The bytes into the array where the piece starts.
    pub inside: u64,
    /// The offset of the piece in the file.
    pub offset: u64,
    pub bytes: usize,
}

/// The pieces of `file` for parameters of `counts` floats each, none larger than `chunk` bytes, in file order.
pub fn pieces(file: File, counts: &[u64], chunk: usize) -> Vec<Piece> {
    let mut order: Vec<(usize, Part)> = Vec::new();
    match file {
        File::State => {
            for parameter in 0..counts.len() {
                order.extend([Part::Values, Part::First, Part::Second].map(|part| (parameter, part)));
            }
            order.extend((0..counts.len()).map(|parameter| (parameter, Part::Gradient)));
        }
        File::Best => order.extend((0..counts.len()).map(|parameter| (parameter, Part::Values))),
    }
    let (mut offset, mut out) = (HEADER, Vec::new());
    for (parameter, part) in order {
        let total = 4 * counts[parameter];
        let mut inside = 0;
        while inside < total {
            let bytes = (total - inside).min(chunk as u64) as usize;
            out.push(Piece {
                parameter,
                part,
                inside,
                offset,
                bytes,
            });
            inside += bytes as u64;
            offset += bytes as u64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_state_holds_each_parameters_values_and_moments_then_every_gradient() {
        let found = pieces(File::State, &[3, 2], 1 << 20);
        let seen: Vec<_> = found
            .iter()
            .map(|piece| (piece.parameter, piece.part, piece.offset, piece.bytes))
            .collect();
        let at = |bytes: u64| HEADER + bytes;
        assert_eq!(
            seen,
            [
                (0, Part::Values, at(0), 12),
                (0, Part::First, at(12), 12),
                (0, Part::Second, at(24), 12),
                (1, Part::Values, at(36), 8),
                (1, Part::First, at(44), 8),
                (1, Part::Second, at(52), 8),
                (0, Part::Gradient, at(60), 12),
                (1, Part::Gradient, at(72), 8),
            ]
        );
    }

    #[test]
    fn the_best_file_holds_the_values_only() {
        let found = pieces(File::Best, &[3, 2], 1 << 20);
        let seen: Vec<_> = found
            .iter()
            .map(|piece| (piece.parameter, piece.part, piece.offset - HEADER))
            .collect();
        assert_eq!(seen, [(0, Part::Values, 0), (1, Part::Values, 12)]);
    }

    #[test]
    fn a_section_over_a_chunk_is_cut_into_pieces_that_tile_it() {
        let found = pieces(File::Best, &[7], 8);
        let sizes: Vec<_> = found.iter().map(|piece| (piece.inside, piece.bytes)).collect();
        assert_eq!(sizes, [(0, 8), (8, 8), (16, 8), (24, 4)]);
        let mut next = HEADER;
        for piece in &found {
            assert_eq!(piece.offset, next);
            next += piece.bytes as u64;
        }
        assert_eq!(next, HEADER + 28);
    }

    #[test]
    fn no_parameters_have_no_pieces() {
        assert!(pieces(File::State, &[], 8).is_empty());
        assert!(pieces(File::Best, &[0], 8).is_empty());
    }
}
