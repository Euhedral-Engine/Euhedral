//! What the engine's sampler refuses of a request: a sampling value it cannot take, answered as the surface's invalid
//! request.

use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::generation::GenerationRequest;
use euhedral_inference_engine::sampling::{SamplingConfig, SamplingError};

/// What the engine's sampler refuses of the request's sampling, as the surface's error. The mappers validate the
/// ranges the API documents, so this answers only for a value the sampler cannot take (not finite).
pub fn sampling_problem(request: &GenerationRequest) -> Option<ApiError> {
    let sampling = &request.sampling;
    let top_k = u32::try_from(sampling.top_k).unwrap_or(0);
    match SamplingConfig::new(sampling.temperature, top_k, sampling.top_p, 0, sampling.greedy) {
        Err(SamplingError::Temperature) => Some(ApiError::invalid_request(
            "'temperature' must be a finite number.",
            Some("temperature"),
        )),
        Err(SamplingError::TopP) => Some(ApiError::invalid_request(
            "'top_p' must be finite and in the range (0, 1].",
            Some("top_p"),
        )),
        _ => None,
    }
}
