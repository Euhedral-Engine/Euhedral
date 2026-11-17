//! The model lists: `GET /v1/models` and `GET /v1/models/{model}`, in OpenAI's form or, for a client that sends
//! `anthropic-version`, in Anthropic's.

use euhedral_inference_api_protocol::ToJson;
use euhedral_inference_api_protocol::{anthropic, openai};

/// What the lists say besides the model's name: when it was created, in seconds since the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Served {
    pub created_at: i64,
}

impl Served {
    /// The JSON of the list.
    pub fn list(&self, name: &str, anthropic_form: bool) -> String {
        if anthropic_form {
            anthropic::ModelList::of(anthropic::Model::of(name, self.created_at)).to_json_string()
        } else {
            openai::ModelList::of(openai::Model::of(name, self.created_at)).to_json_string()
        }
    }

    /// The JSON of the model `name` when it is `served`, the one the backend serves.
    pub fn one(&self, served: &str, name: &str, anthropic_form: bool) -> Option<String> {
        if name != served {
            return None;
        }
        Some(if anthropic_form {
            anthropic::Model::of(name, self.created_at).to_json_string()
        } else {
            openai::Model::of(name, self.created_at).to_json_string()
        })
    }
}
