use std::collections::BTreeMap;

use serde_json::Value;

use super::error::MixedError;
use super::option::QuantOption;
use super::pretty::{Node, render};

/// The option chosen for each projection object, by object name; an object not listed keeps the base's. The file is
/// `{"objects": {"text/layers/0/mlp/down": "trellis-3.0", ...}}`, keys sorted.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Allocation {
    choices: BTreeMap<String, QuantOption>,
}

impl Allocation {
    /// An allocation of `choices`.
    pub fn new(choices: BTreeMap<String, QuantOption>) -> Allocation {
        Allocation { choices }
    }

    /// The choices, by object name in order.
    pub fn choices(&self) -> &BTreeMap<String, QuantOption> {
        &self.choices
    }

    /// The allocation in the text of a file; an unknown option key is refused naming it and its object.
    pub fn parse(text: &str) -> Result<Allocation, MixedError> {
        let root = parse_json(text)?;
        let objects = root
            .get("objects")
            .and_then(Value::as_object)
            .ok_or_else(|| MixedError::invalid("the allocation has no \"objects\" map"))?;
        let mut choices = BTreeMap::new();
        for (name, value) in objects {
            let key = value
                .as_str()
                .ok_or_else(|| MixedError::invalid(format!("object {name} has no option name")))?;
            let option =
                QuantOption::of(key).map_err(|failure| MixedError::invalid(format!("object {name}: {failure}")))?;
            choices.insert(name.clone(), option);
        }
        Ok(Allocation { choices })
    }

    /// The text of this allocation's file.
    pub fn render(&self) -> String {
        render(&self.node("objects"))
    }

    /// `{"<field>": {name: option}}` as a node, for the files that hold several allocations.
    pub(crate) fn node(&self, field: &str) -> Node {
        let objects = self
            .choices
            .iter()
            .map(|(name, option)| (name.clone(), Node::Text(option.key().to_string())))
            .collect();
        Node::Object(vec![(field.to_string(), Node::Object(objects))])
    }
}

pub(crate) fn parse_json(text: &str) -> Result<Value, MixedError> {
    serde_json::from_str(text).map_err(|failure| MixedError::invalid(format!("not JSON: {failure}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocation() -> Allocation {
        Allocation::new(BTreeMap::from([
            ("text/layers/1/mlp/down".to_string(), QuantOption::Trellis3_5),
            ("text/layers/0/mlp/down".to_string(), QuantOption::Nvfp4),
            ("text/layers/0/mlp/gate_up".to_string(), QuantOption::Trellis2_5),
        ]))
    }

    #[test]
    fn round_trips_with_sorted_keys() {
        let text = allocation().render();
        assert_eq!(Allocation::parse(&text).unwrap(), allocation());
        let at = |needle: &str| text.find(needle).unwrap();
        assert!(at("layers/0/mlp/down") < at("layers/0/mlp/gate_up"));
        assert!(at("layers/0/mlp/gate_up") < at("layers/1/mlp/down"));
        assert!(text.contains("\"trellis-3.5\""));
    }

    #[test]
    fn the_file_layout_is_fixed() {
        assert_eq!(
            allocation().render(),
            "{\n  \"objects\" : {\n    \"text/layers/0/mlp/down\" : \"nvfp4\",\n    \"text/layers/0/mlp/gate_up\" : \"trellis-2.5\",\n    \"text/layers/1/mlp/down\" : \"trellis-3.5\"\n  }\n}\n"
        );
        assert_eq!(Allocation::default().render(), "{\n  \"objects\" : { }\n}\n");
    }

    #[test]
    fn reads_the_documented_shape() {
        let read = Allocation::parse("{\"objects\": {\"text/layers/0/mlp/down\": \"trellis-3.0\"}}").unwrap();
        assert_eq!(
            read.choices(),
            &BTreeMap::from([("text/layers/0/mlp/down".to_string(), QuantOption::Trellis3_0)])
        );
    }

    #[test]
    fn an_unknown_option_key_is_refused_naming_it_and_its_object() {
        let failure = Allocation::parse("{\"objects\": {\"text/layers/0/mlp/down\": \"trellis-9.9\"}}")
            .unwrap_err()
            .to_string();
        assert!(failure.contains("trellis-9.9"), "{failure}");
        assert!(failure.contains("text/layers/0/mlp/down"), "{failure}");
    }

    #[test]
    fn a_file_without_the_objects_map_or_with_a_non_text_option_is_refused() {
        assert!(Allocation::parse("{}").is_err());
        assert!(Allocation::parse("{\"objects\": []}").is_err());
        let failure = Allocation::parse("{\"objects\": {\"a\": 3}}").unwrap_err().to_string();
        assert!(failure.contains("object a"), "{failure}");
        assert!(Allocation::parse("not json").is_err());
    }

    #[test]
    fn option_keys_round_trip() {
        for option in QuantOption::ALL {
            assert_eq!(QuantOption::of(option.key()).unwrap(), option);
        }
        assert_eq!(QuantOption::Trellis2_5.key(), "trellis-2.5");
        assert!(QuantOption::of("fp8").unwrap_err().to_string().contains("nvfp4"));
    }
}
