//! Where each part of the parameters lies in the state files. A file is a header, then sections in a fixed order, each
//! cut into pieces of at most a chunk so that no piece is larger than the lane reads at once.
//!
//! The state file (`resume.bin`) holds, for every parameter in order, its values, its first and its second moment (the
//! order Java's state has), then every parameter's gradient, which Java does not save: a state taken between two
//! updates has gradients that the next update needs. The best file holds the values only.

use crate::mtp::Parameter;
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

/// A piece of a file with the device address of the bytes it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub piece: Piece,
    pub address: u64,
}

/// Where the device keeps the bytes of `piece`.
pub fn address(parameters: &[Parameter], piece: &Piece) -> u64 {
    let parameter = &parameters[piece.parameter];
    let base = match piece.part {
        Part::Values => parameter.values,
        Part::First => parameter.first,
        Part::Second => parameter.second,
        Part::Gradient => parameter.gradient,
    };
    base + piece.inside
}

/// The pieces of `file` for `parameters`, none larger than `chunk` bytes, with their device addresses.
pub fn place(parameters: &[Parameter], file: File, chunk: usize) -> Vec<Placed> {
    let counts: Vec<u64> = parameters.iter().map(Parameter::count).collect();
    pieces(file, &counts, chunk)
        .into_iter()
        .map(|piece| Placed {
            piece,
            address: address(parameters, &piece),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

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

    fn parameter(rows: u32, columns: u32, base: u64) -> Parameter {
        Parameter {
            name: "test",
            rows,
            columns,
            values: base,
            gradient: base + 0x1000,
            first: base + 0x2000,
            second: base + 0x3000,
            maximum: None,
        }
    }

    #[test]
    fn a_piece_is_read_from_its_arrays_address_plus_how_far_into_the_array_it_starts() {
        let parameters = [parameter(1, 7, 0x10_0000), parameter(2, 2, 0x20_0000)];
        let placed = place(&parameters, File::State, 8);
        let seen: Vec<_> = placed
            .iter()
            .map(|placed| (placed.piece.parameter, placed.piece.part, placed.address))
            .collect();
        assert_eq!(seen[0], (0, Part::Values, 0x10_0000));
        assert_eq!(seen[1], (0, Part::Values, 0x10_0008));
        assert_eq!(seen[3], (0, Part::Values, 0x10_0018));
        let first_moment = placed
            .iter()
            .find(|p| p.piece.part == Part::First && p.piece.parameter == 1);
        assert_eq!(first_moment.map(|p| p.address), Some(0x20_2000));
        let last = placed.last().unwrap();
        assert_eq!((last.piece.parameter, last.piece.part), (1, Part::Gradient));
        assert_eq!(last.address, 0x20_1008);
    }
}
