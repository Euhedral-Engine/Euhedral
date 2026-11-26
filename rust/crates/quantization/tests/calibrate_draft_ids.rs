#![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

//! The draft head's ids read from an artifact: the tensor's values, each checked against the output head's rows.

use euhedral_inference_edrl::tiny::TinyArtifact;
use euhedral_inference_edrl::{HEADER_BYTES, Header, TensorTable};
use euhedral_inference_quantization::calibrate::draft_ids::{check_draft_ids, read_draft_ids};

fn ids_of(bytes: &[u8], vocabulary: u64) -> Result<Vec<i32>, String> {
    read_draft_ids(
        bytes.len() as u64,
        |range| Ok(bytes[range.start as usize..range.end as usize].to_vec()),
        vocabulary,
    )
}

#[test]
fn the_ids_are_the_tensors_values_and_each_must_be_in_the_output_heads_rows() {
    let tiny = TinyArtifact::new(7, false);
    let mut bytes = tiny.bytes().unwrap();
    let header = Header::parse(bytes[..HEADER_BYTES].try_into().unwrap()).unwrap();
    let length = bytes.len() as u64;
    let range = header.table_range();
    let table = &bytes[range.start as usize..range.end as usize];
    let table = TensorTable::parse(table, &header, length).unwrap();
    let at = table.get("text/draft_head_token_ids").unwrap().offset() as usize;
    let ids: Vec<u8> = [3i32, 1, 2, 0]
        .iter()
        .cycle()
        .take(16)
        .flat_map(|id| id.to_le_bytes())
        .collect();
    bytes[at..at + ids.len()].copy_from_slice(&ids);
    assert_eq!(ids_of(&bytes, 4).unwrap()[..4], [3, 1, 2, 0]);
    let why = ids_of(&bytes, 3).unwrap_err();
    assert!(why.contains("holds 3") && why.contains("outside the 3 rows"), "{why}");
    bytes[at..at + 4].copy_from_slice(&(-1i32).to_le_bytes());
    assert!(ids_of(&bytes, 4).unwrap_err().contains("holds -1"));
}

#[test]
fn bytes_that_are_not_an_artifact_are_refused() {
    assert!(ids_of(&[0; 100], 4).is_err());
}

#[test]
fn ids_are_refused_when_empty_negative_or_past_the_output_heads_rows_and_each_refusal_says_which() {
    check_draft_ids(&[0, 3, 3], 4).unwrap();
    assert!(check_draft_ids(&[], 4).unwrap_err().contains("empty"));
    let why = check_draft_ids(&[1, 4], 4).unwrap_err();
    assert!(why.contains("holds 4") && why.contains("4 rows"), "{why}");
    assert!(check_draft_ids(&[-2], 4).unwrap_err().contains("holds -2"));
}
