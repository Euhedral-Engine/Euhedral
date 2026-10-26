//! A quantum's input record: what its rows need on the device that is not in the workspace's other slots.
//!
//! The record is one upload ahead of the quantum's first launch: its token ids as 32-bit values, padded to 8 bytes,
//! then its start position as a 64-bit value (`Workspace.inputByteSize`, `Workspace.writeInput`). RoPE, the KV append
//! and every decode attention kernel read the position from the record, so a decode quantum's launch parameters do
//! not change with its position and its graph can be replayed.

use crate::workspace::input_record_bytes;

/// The bytes of the record of `ids` and `start_position`, in the device's (native) byte order.
pub fn encode(ids: &[u32], start_position: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; input_record_bytes(ids.len() as u64) as usize];
    for (index, id) in ids.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&id.to_ne_bytes());
    }
    let position = bytes.len() - 8;
    bytes[position..].copy_from_slice(&start_position.to_ne_bytes());
    bytes
}

/// Writes the record into `out`, which must be at least [`input_record_bytes`] long; the padding is zero.
pub fn encode_into(ids: &[u32], start_position: u64, out: &mut [u8]) {
    let record = encode(ids, start_position);
    out[..record.len()].copy_from_slice(&record);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::InputRecordRef;

    #[test]
    fn the_ids_come_first_and_the_position_follows_at_an_eight_byte_boundary() {
        for rows in [1usize, 2, 3, 5, 8, 512] {
            let ids: Vec<u32> = (0..rows as u32).map(|row| 1000 + row).collect();
            let bytes = encode(&ids, 0x0123_4567_89ab_cdef);
            let offset = InputRecordRef::position_offset(rows) as usize;
            assert_eq!(offset % 8, 0);
            assert_eq!(bytes.len(), offset + 8, "{rows} rows");
            for (row, id) in ids.iter().enumerate() {
                assert_eq!(bytes[row * 4..row * 4 + 4], id.to_ne_bytes());
            }
            assert_eq!(bytes[offset..], 0x0123_4567_89ab_cdefu64.to_ne_bytes());
            assert!(
                bytes[rows * 4..offset].iter().all(|byte| *byte == 0),
                "the padding is zero"
            );
        }
    }
}
