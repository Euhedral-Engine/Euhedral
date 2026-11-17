//! The model lists: `GET /v1/models` and `GET /v1/models/{model}`, in OpenAI's form or, for a client that sends
//! `anthropic-version`, in Anthropic's.

use euhedral_inference_api_protocol::ToJson;
use euhedral_inference_api_protocol::{anthropic, openai};

/// The one served model: its public name and when it was created, in seconds since the epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Served {
    pub name: String,
    pub created_at: i64,
}

impl Served {
    /// The JSON of the list.
    pub fn list(&self, anthropic_form: bool) -> String {
        if anthropic_form {
            anthropic::ModelList::of(anthropic::Model::of(&self.name, self.created_at)).to_json_string()
        } else {
            openai::ModelList::of(openai::Model::of(&self.name, self.created_at)).to_json_string()
        }
    }

    /// The JSON of the model `name` when it is the served one.
    pub fn one(&self, name: &str, anthropic_form: bool) -> Option<String> {
        if name != self.name {
            return None;
        }
        Some(if anthropic_form {
            anthropic::Model::of(&self.name, self.created_at).to_json_string()
        } else {
            openai::Model::of(&self.name, self.created_at).to_json_string()
        })
    }
}
