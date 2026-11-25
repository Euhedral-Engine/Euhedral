use std::collections::BTreeMap;

use serde_json::Value;

use super::allocation::{Allocation, parse_json};
use super::error::MixedError;
use super::option::QuantOption;

/// Allocations to measure first, from the text of a file: `{"allocations": [{"objects": {"text/layers/0/mlp/down":
/// "trellis-3.0", ...}}, ...]}`, or one allocation file (`{"objects": {...}}`). An object an allocation does not name
/// keeps the base's option.
pub fn read_seeds(text: &str) -> Result<Vec<Allocation>, MixedError> {
    let root = parse_json(text)?;
    match root.get("allocations") {
        Some(Value::Array(nodes)) => nodes.iter().map(allocation).collect(),
        Some(_) => Err(MixedError::invalid("\"allocations\" is not a list")),
        None => Ok(vec![allocation(&root)?]),
    }
}

fn allocation(node: &Value) -> Result<Allocation, MixedError> {
    let objects = node
        .get("objects")
        .and_then(Value::as_object)
        .ok_or_else(|| MixedError::invalid("an allocation has no \"objects\" map"))?;
    let mut choices = BTreeMap::new();
    for (name, value) in objects {
        let key = value.as_str().unwrap_or_default();
        let option =
            QuantOption::of(key).map_err(|failure| MixedError::invalid(format!("object {name}: {failure}")))?;
        choices.insert(name.clone(), option);
    }
    Ok(Allocation::new(choices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str, option: QuantOption) -> Allocation {
        Allocation::new(BTreeMap::from([(name.to_string(), option)]))
    }

    #[test]
    fn a_list_or_one_allocation_is_read() {
        let list = read_seeds(
            "{\"allocations\": [{\"objects\": {\"text/layers/0/mlp/down\": \"nvfp4\"}}, {\"objects\": {}}]}",
        )
        .unwrap();
        assert_eq!(
            list,
            vec![one("text/layers/0/mlp/down", QuantOption::Nvfp4), Allocation::default()]
        );
        let single = read_seeds("{\"objects\": {\"text/layers/1/mlp/down\": \"trellis-3.0\"}}").unwrap();
        assert_eq!(single, vec![one("text/layers/1/mlp/down", QuantOption::Trellis3_0)]);
    }

    #[test]
    fn a_bad_allocation_is_refused_naming_its_object() {
        let failure = read_seeds("{\"allocations\": [{\"objects\": {\"x\": \"fp8\"}}]}")
            .unwrap_err()
            .to_string();
        assert!(failure.contains("object x") && failure.contains("fp8"), "{failure}");
        assert!(read_seeds("{\"allocations\": [{}]}").is_err());
        assert!(read_seeds("{\"allocations\": 3}").is_err());
    }
}
