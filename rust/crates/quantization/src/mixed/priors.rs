use std::collections::BTreeMap;

use serde_json::Value;

use super::allocation::parse_json;
use super::error::MixedError;
use super::option::QuantOption;

/// The local error of each option of each object, a proxy the fidelity model starts from. The file is
/// `{"objects": {"text/layers/0/mlp/down": {"nvfp4": 0.0012, "trellis-2.5": 0.0031, ...}, ...}}`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Priors {
    objects: BTreeMap<String, BTreeMap<QuantOption, f64>>,
}

impl Priors {
    /// Priors of `objects`.
    pub fn new(objects: BTreeMap<String, BTreeMap<QuantOption, f64>>) -> Priors {
        Priors { objects }
    }

    /// The objects with their errors, in name order.
    pub fn objects(&self) -> &BTreeMap<String, BTreeMap<QuantOption, f64>> {
        &self.objects
    }

    /// The priors in the text of a file; an unknown option key is refused naming it and its object.
    pub fn parse(text: &str) -> Result<Priors, MixedError> {
        let root = parse_json(text)?;
        let objects = root
            .get("objects")
            .and_then(Value::as_object)
            .ok_or_else(|| MixedError::invalid("the priors have no \"objects\" map"))?;
        let mut read = BTreeMap::new();
        for (name, errors) in objects {
            let errors = errors
                .as_object()
                .ok_or_else(|| MixedError::invalid(format!("object {name}: the errors are not a map")))?;
            let mut by_option = BTreeMap::new();
            for (key, error) in errors {
                let number = error
                    .as_f64()
                    .filter(|_| error.is_number())
                    .ok_or_else(|| MixedError::invalid(format!("object {name}: the error of {key} is not a number")))?;
                let option =
                    QuantOption::of(key).map_err(|failure| MixedError::invalid(format!("object {name}: {failure}")))?;
                by_option.insert(option, number);
            }
            read.insert(name.clone(), by_option);
        }
        Ok(Priors { objects: read })
    }

    /// These priors of only `objects` and `options`; the refusal names what is missing.
    pub fn restrict(&self, objects: &[String], options: &[QuantOption]) -> Result<Priors, MixedError> {
        let mut kept = BTreeMap::new();
        for object in objects {
            let mut by_option = BTreeMap::new();
            for option in options {
                by_option.insert(*option, self.error(object, *option)?);
            }
            kept.insert(object.clone(), by_option);
        }
        Ok(Priors { objects: kept })
    }

    /// The error of `option` at `object`; the refusal names both when there is none.
    pub fn error(&self, object: &str, option: QuantOption) -> Result<f64, MixedError> {
        let by_option = self
            .objects
            .get(object)
            .ok_or_else(|| MixedError::invalid(format!("the priors have no object {object}")))?;
        by_option
            .get(&option)
            .copied()
            .ok_or_else(|| MixedError::invalid(format!("the priors have no error of {} for {object}", option.key())))
    }

    /// The options every object has an error for, in declaration order.
    pub fn options(&self) -> Vec<QuantOption> {
        QuantOption::ALL
            .into_iter()
            .filter(|option| self.objects.values().all(|by_option| by_option.contains_key(option)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"{"objects": {"text/layers/0/mlp/down": {"nvfp4": 0.0012, "trellis-2.5": 0.0031},
                 "text/layers/1/mlp/down": {"nvfp4": 0.002, "trellis-2.5": 0.004}}}"#;

    #[test]
    fn the_error_of_each_option_of_each_object_is_read() {
        let priors = Priors::parse(TEXT).unwrap();
        assert_eq!(
            priors.error("text/layers/0/mlp/down", QuantOption::Trellis2_5).unwrap(),
            0.0031
        );
        assert_eq!(priors.options(), vec![QuantOption::Nvfp4, QuantOption::Trellis2_5]);
        let failure = priors
            .error("text/layers/1/mlp/down", QuantOption::Trellis3_0)
            .unwrap_err()
            .to_string();
        assert!(
            failure.contains("trellis-3.0") && failure.contains("text/layers/1/mlp/down"),
            "{failure}"
        );
        assert!(priors.error("text/layers/9/mlp/down", QuantOption::Nvfp4).is_err());
    }

    #[test]
    fn an_unknown_option_is_refused_naming_its_object() {
        let failure = Priors::parse("{\"objects\": {\"text/layers/0/mlp/down\": {\"fp8\": 1}}}")
            .unwrap_err()
            .to_string();
        assert!(failure.contains("text/layers/0/mlp/down"), "{failure}");
        assert!(Priors::parse("{\"objects\": {\"a\": {\"nvfp4\": \"x\"}}}").is_err());
        assert!(Priors::parse("{}").is_err());
    }

    #[test]
    fn the_common_options_are_those_every_object_has() {
        let priors =
            Priors::parse(r#"{"objects": {"a": {"nvfp4": 1, "trellis-3.0": 2}, "b": {"nvfp4": 1, "trellis-2.5": 2}}}"#)
                .unwrap();
        assert_eq!(priors.options(), vec![QuantOption::Nvfp4]);
    }

    #[test]
    fn restricting_keeps_the_named_objects_and_options_and_refuses_what_is_missing() {
        let priors = Priors::parse(TEXT).unwrap();
        let kept = priors
            .restrict(&["text/layers/1/mlp/down".to_string()], &[QuantOption::Nvfp4])
            .unwrap();
        assert_eq!(kept.objects().len(), 1);
        assert_eq!(kept.options(), vec![QuantOption::Nvfp4]);
        assert!(
            priors
                .restrict(&["text/layers/1/mlp/down".to_string()], &[QuantOption::Trellis4_0])
                .is_err()
        );
        assert!(priors.restrict(&["nope".to_string()], &[QuantOption::Nvfp4]).is_err());
    }
}
