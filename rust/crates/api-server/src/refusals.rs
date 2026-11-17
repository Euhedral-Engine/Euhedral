//! What the engine backend refuses, in one place: a request that asks for something the engine cannot honour yet is
//! answered `unsupported_parameter`, naming the field of the surface the request came in on.
//!
//! Each entry is a rule: a test of the mapped request and the parameter it names on each surface. To lift a refusal,
//! delete its line from [`RULES`].

use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::generation::GenerationRequest;
use euhedral_inference_api_protocol::pipeline::ToolChoice;

use crate::planning::Endpoint;

/// One refusal: when it applies and the parameter that names it on each surface.
struct Rule {
    applies: fn(&GenerationRequest) -> bool,
    chat: &'static str,
    messages: &'static str,
    responses: &'static str,
}

impl Rule {
    fn param(&self, endpoint: Endpoint) -> &'static str {
        match endpoint {
            Endpoint::ChatCompletions => self.chat,
            Endpoint::Messages => self.messages,
            Endpoint::Responses => self.responses,
        }
    }
}

/// Every refusal, in the order they are tried.
const RULES: &[Rule] = &[
    // Sampling: the engine can sample, but only without speculation and without a readback the engine may lack, so
    // generation stays greedy until that is settled.
    Rule {
        applies: |request| !request.sampling.greedy,
        chat: "temperature",
        messages: "temperature",
        responses: "temperature",
    },
    // A JSON answer is a grammar the engine cannot constrain to.
    Rule {
        applies: |request| request.format.is_json(),
        chat: "response_format",
        messages: "output_config.format",
        responses: "text.format",
    },
    // A forced tool call is a grammar too.
    Rule {
        applies: |request| {
            request.tools.parses_output() && !matches!(request.tools.choice, ToolChoice::Auto | ToolChoice::None)
        },
        chat: "tool_choice",
        messages: "tool_choice",
        responses: "tool_choice",
    },
    // A reasoning budget ends the think block by force: a constraint on the sampled tokens.
    Rule {
        applies: |request| request.reasoning_budget.is_some(),
        chat: "reasoning_budget",
        messages: "thinking.budget_tokens",
        responses: "reasoning",
    },
];

/// The refusal of `request` as the surface words it, or `None` when the engine can run it.
pub fn refusal(endpoint: Endpoint, request: &GenerationRequest) -> Option<ApiError> {
    RULES
        .iter()
        .find(|rule| (rule.applies)(request))
        .map(|rule| ApiError::unsupported_parameter(rule.param(endpoint)))
}
