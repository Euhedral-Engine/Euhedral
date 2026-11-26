//! The files of the front: `front.json`, each allocation of the front and the text report.

use std::fmt::Write;

use serde_json::{Map, Value, json, to_string_pretty};

use super::super::search_state::{Measured, Progress};
use super::candidate::front_of;
use super::failure::SearchFailure;

const MIB: i64 = 1 << 20;

/// The files to publish, each a path below the out directory and its text: `front.json`, `report.txt` and
/// `allocations/front-NN.json` for the NN-th front point. `free_bytes` and `budget` head the report.
pub(super) fn render(
    measured: &[Measured],
    progress: Progress,
    budget: i64,
    free_bytes: i64,
) -> Result<Vec<(String, String)>, SearchFailure> {
    let front = front_of(measured);
    let mut files = Vec::with_capacity(front.len() + 2);
    let mut points = Vec::with_capacity(front.len());
    let mut report = String::new();
    // Writing to a String cannot fail.
    let _ = write!(
        report,
        "mixed allocation search: {} measured, {} on the front, round {}, budget {} MiB, {} MiB free\n\n",
        measured.len(),
        front.len(),
        progress.round,
        budget / MIB,
        free_bytes / MIB
    );
    let _ = writeln!(
        report,
        "      kl    +-se   MiB  prefill us/token  verify us  round  origin"
    );
    for (i, point) in front.iter().enumerate() {
        let file = format!("allocations/front-{i:02}.json");
        files.push((file.clone(), point.allocation.render()));
        let mut entry = Map::new();
        entry.insert("kl".to_owned(), json!(point.kl));
        entry.insert("klSe".to_owned(), json!(point.kl_se));
        entry.insert("bytes".to_owned(), json!(point.bytes));
        entry.insert("prefillMicrosPerToken".to_owned(), json!(point.prefill));
        entry.insert("verifyMicros".to_owned(), json!(point.decode));
        entry.insert("round".to_owned(), json!(point.round));
        entry.insert("origin".to_owned(), json!(point.origin));
        entry.insert("allocation".to_owned(), json!(file));
        points.push(Value::Object(entry));
        let _ = writeln!(
            report,
            "{:8.5} {:7.5} {:5} {:17.2} {:10.1} {:6}  {}",
            point.kl,
            point.kl_se,
            point.bytes / MIB,
            point.prefill,
            point.decode,
            point.round,
            point.origin
        );
    }
    let text =
        to_string_pretty(&json!({ "front": points })).map_err(|error| SearchFailure::Output(error.to_string()))?;
    files.push(("front.json".to_owned(), format!("{text}\n")));
    files.push(("report.txt".to_owned(), report));
    Ok(files)
}
