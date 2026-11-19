//! One object's stored rounding and the `.npz` the converter reads (`--codes-from`).
//!
//! Grouped formats hold int8 codes `[rows, columns]` and float16 scales `[rows, columns / 64]`. NVFP4 holds uint8
//! codes (magnitude, with the sign in bit 3), uint8 scale codes (or SD4 table indices) `[rows, columns / 16]`, the FP32
//! tensor scale and, for SD4, the 16-entry table. This module makes and reads the archive's bytes; where they are
//! written is the caller's.

use thiserror::Error;

use crate::gptq::sd4::ENTRIES;
use crate::npy::{Dtype, NpyError};
use crate::npz::{Npz, NpzError, NpzWriter};

/// The format name of the NVFP4 states.
const NVFP4: &str = "NVFP4";

/// Why a state could not be made or read.
#[derive(Debug, Error)]
pub enum StateError {
    #[error(transparent)]
    Archive(#[from] NpzError),
    #[error(transparent)]
    Array(#[from] NpyError),
    /// The arrays do not have the sizes the shape and the format imply.
    #[error("the state's {member} holds {found} bytes, the shape needs {expected}")]
    Size {
        member: &'static str,
        expected: u64,
        found: u64,
    },
    #[error("the state's shape is not two positive dimensions")]
    Shape,
}

/// One object's rounding.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantState {
    pub format_name: String,
    pub layout: String,
    pub rows: u64,
    pub columns: u64,
    /// `rows * columns` codes, row-major.
    pub codes: Vec<u8>,
    /// The group scales, row-major: little-endian float16 for grouped formats, one byte for NVFP4.
    pub scales: Vec<u8>,
    /// The FP32 tensor scale of an NVFP4 object, else 0.
    pub global_scale: f32,
    /// The SD4 table, when the layout has one.
    pub table: Option<[u8; ENTRIES]>,
}

impl QuantState {
    /// Whether the state is NVFP4.
    pub fn is_nvfp4(&self) -> bool {
        self.format_name == NVFP4
    }

    /// The scales per row: columns over the group (16 for NVFP4, 64 otherwise).
    pub fn groups(&self) -> u64 {
        self.columns / if self.is_nvfp4() { 16 } else { 64 }
    }

    /// The file name of the state of object `name`: `/` becomes `.`.
    pub fn file_name(name: &str) -> String {
        format!("{}.npz", name.replace('/', "."))
    }

    /// The archive: format name, layout, shape, codes, scales, global scale and, with a table, the table, in that
    /// order.
    pub fn to_npz(&self) -> Result<Vec<u8>, StateError> {
        let nvfp4 = self.is_nvfp4();
        self.check_sizes()?;
        let mut writer = NpzWriter::new(Vec::new());
        writer.add_text("format_name", &self.format_name)?;
        writer.add_text("layout", &self.layout)?;
        let mut shape = Vec::with_capacity(16);
        for dimension in [self.rows, self.columns] {
            shape.extend_from_slice(&(dimension as i64).to_le_bytes());
        }
        writer.add_array("shape", Dtype::I8, &[2], &shape)?;
        let codes = if nvfp4 { Dtype::U1 } else { Dtype::I1 };
        writer.add_array("codes", codes, &[self.rows, self.columns], &self.codes)?;
        let scales = if nvfp4 { Dtype::U1 } else { Dtype::F2 };
        writer.add_array("scales", scales, &[self.rows, self.groups()], &self.scales)?;
        writer.add_array("global_scale", Dtype::F4, &[], &self.global_scale.to_le_bytes())?;
        if let Some(table) = &self.table {
            writer.add_array("table", Dtype::U1, &[ENTRIES as u64], table)?;
        }
        Ok(writer.finish()?)
    }

    /// Reads an archive `to_npz` (or the converter) wrote.
    pub fn from_npz(bytes: &[u8]) -> Result<QuantState, StateError> {
        let archive = Npz::parse(bytes)?;
        let format_name = archive.array("format_name")?.text()?;
        let layout = archive.array("layout")?.text()?;
        let shape = archive.array("shape")?.to_vec::<i64>()?;
        let (rows, columns) = match shape[..] {
            [rows, columns] if rows > 0 && columns > 0 => (rows as u64, columns as u64),
            _ => return Err(StateError::Shape),
        };
        let codes = archive.array("codes")?.data.to_vec();
        let scales = archive.array("scales")?.data.to_vec();
        let global_scale = *archive
            .array("global_scale")?
            .to_vec::<f32>()?
            .first()
            .ok_or(StateError::Shape)?;
        let table = match archive.array("table") {
            Ok(array) => {
                let data = array.data;
                Some(<[u8; ENTRIES]>::try_from(data).map_err(|_| StateError::Size {
                    member: "table",
                    expected: ENTRIES as u64,
                    found: data.len() as u64,
                })?)
            }
            Err(NpzError::Missing(_)) => None,
            Err(other) => return Err(other.into()),
        };
        let state = QuantState {
            format_name,
            layout,
            rows,
            columns,
            codes,
            scales,
            global_scale,
            table,
        };
        state.check_sizes()?;
        Ok(state)
    }

    fn check_sizes(&self) -> Result<(), StateError> {
        let scale_bytes = if self.is_nvfp4() { 1 } else { 2 };
        for (member, expected, found) in [
            ("codes", self.rows * self.columns, self.codes.len() as u64),
            ("scales", self.rows * self.groups() * scale_bytes, self.scales.len() as u64),
        ] {
            if expected != found {
                return Err(StateError::Size {
                    member,
                    expected,
                    found,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grouped() -> QuantState {
        QuantState {
            format_name: "Q3G64_F16S".into(),
            layout: "row-split-k128-v1".into(),
            rows: 2,
            columns: 64,
            codes: (0..128).map(|i| (i % 7) as u8).collect(),
            scales: vec![0x00, 0x3c, 0x00, 0x40],
            global_scale: 0.0,
            table: None,
        }
    }

    fn nvfp4() -> QuantState {
        QuantState {
            format_name: "NVFP4".into(),
            layout: "row-split-k128-sd4-v1".into(),
            rows: 2,
            columns: 32,
            codes: (0..64).map(|i| (i % 16) as u8).collect(),
            scales: vec![1, 2, 3, 4],
            global_scale: 0.0141369,
            table: Some([88, 98, 101, 103, 105, 107, 109, 112, 113, 115, 117, 120, 122, 123, 124, 126]),
        }
    }

    #[test]
    fn a_state_survives_the_archive() {
        for state in [grouped(), nvfp4()] {
            let bytes = state.to_npz().unwrap();
            assert_eq!(QuantState::from_npz(&bytes).unwrap(), state);
        }
    }

    #[test]
    fn the_members_come_in_the_order_the_converter_reads_them() {
        let bytes = nvfp4().to_npz().unwrap();
        let archive = Npz::parse(&bytes).unwrap();
        let names: Vec<&str> = archive.entries().iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(
            names,
            ["format_name.npy", "layout.npy", "shape.npy", "codes.npy", "scales.npy", "global_scale.npy", "table.npy"]
        );
        let bytes = grouped().to_npz().unwrap();
        assert!(Npz::parse(&bytes).unwrap().array("table").is_err());
    }

    #[test]
    fn a_state_whose_arrays_do_not_fit_its_shape_is_refused() {
        let mut state = grouped();
        state.scales.pop();
        assert!(matches!(state.to_npz(), Err(StateError::Size { member: "scales", .. })));
        let mut state = nvfp4();
        state.codes.push(0);
        assert!(matches!(state.to_npz(), Err(StateError::Size { member: "codes", .. })));
    }

    #[test]
    fn an_object_name_becomes_a_file_name() {
        assert_eq!(QuantState::file_name("layers/3/mlp/up"), "layers.3.mlp.up.npz");
    }
}
