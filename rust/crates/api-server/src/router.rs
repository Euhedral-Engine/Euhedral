//! The router: a method and a path become a route, and anything else becomes the JSON error of the path's surface.
//!
//! The answers follow the web layer the API had before: an unknown URL is 404 for GET and HEAD and 405 for any other
//! method (the static resource handler's rule), a known path with another method is 405, `OPTIONS` on a known path is
//! 200 with `Allow`, a body that is not JSON is 415 and an `Accept` that excludes JSON is 406. The 405, 415 and 406
//! answers carry only a content type, as every error does; the `Allow` header is sent only with an `OPTIONS` answer.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_inference_api_protocol::{ApiError, Surface};
use euhedral_net::{Handler, Method, Request, Responder, Response, no_work};

use crate::backend::Backend;
use crate::health;
use crate::limits::Limits;
use crate::models::Served;

const JSON: &str = "application/json";

/// A route that has a handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Health,
    Models,
    /// `GET /v1/models/{model}`, with the model as the client wrote it, percent-decoded.
    Model(String),
    ChatCompletions,
    Messages,
    CountTokens,
    Responses,
}

/// What a method and a path come to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    Found(Route),
    /// `OPTIONS` on a path: 200 with this `Allow` value.
    Options(&'static str),
    /// The path takes other methods than this one.
    MethodNotAllowed,
    /// No such path.
    NotFound,
}

/// The routes of a path and the methods each takes.
fn route_of(path: &str) -> Option<(Route, bool)> {
    // The bool: the route is a GET (it also answers HEAD) rather than a POST.
    match path {
        "/health" => Some((Route::Health, true)),
        "/v1/models" => Some((Route::Models, true)),
        "/v1/chat/completions" => Some((Route::ChatCompletions, false)),
        "/v1/messages" => Some((Route::Messages, false)),
        "/v1/messages/count_tokens" => Some((Route::CountTokens, false)),
        "/v1/responses" => Some((Route::Responses, false)),
        _ => {
            let model = path.strip_prefix("/v1/models/")?;
            (!model.is_empty() && !model.contains('/')).then(|| (Route::Model(percent_decode(model)), true))
        }
    }
}

/// Maps a method and the raw path (no query) to what answers it.
pub fn resolve(method: &Method, path: &str) -> Resolution {
    let is_get = matches!(method, Method::Get | Method::Head);
    let Some((route, takes_get)) = route_of(path) else {
        return match method {
            Method::Options => Resolution::Options("GET,HEAD,OPTIONS"),
            _ if is_get => Resolution::NotFound,
            _ => Resolution::MethodNotAllowed,
        };
    };
    match method {
        Method::Options => Resolution::Options(if takes_get {
            "GET, HEAD, OPTIONS"
        } else {
            "POST, OPTIONS"
        }),
        _ if is_get && takes_get => Resolution::Found(route),
        Method::Post if !takes_get => Resolution::Found(route),
        _ => Resolution::MethodNotAllowed,
    }
}

/// Decodes `%XX` escapes; an escape that is not two hex digits stays as written. Bytes that are not UTF-8 become U+FFFD.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        let hex = |i: usize| bytes.get(i).and_then(|b| char::from(*b).to_digit(16));
        if bytes[at] == b'%'
            && let (Some(high), Some(low)) = (hex(at + 1), hex(at + 2))
        {
            out.push((high * 16 + low) as u8);
            at += 3;
            continue;
        }
        out.push(bytes[at]);
        at += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A media type: `type/subtype` in lower case, and the text Spring prints for it (parameters as written, joined by `;`).
struct Media {
    kind: String,
    subtype: String,
    shown: String,
}

fn media_type(text: &str) -> Option<Media> {
    let mut parts = text.split(';');
    let essence = parts.next()?.trim();
    let (kind, subtype) = essence.split_once('/')?;
    let token = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_graphic() && !b"()<>@,;:\\\"/[]?={}".contains(&b))
    };
    if !token(kind) || !token(subtype) && subtype != "*" {
        return None;
    }
    let (kind, subtype) = (kind.to_ascii_lowercase(), subtype.to_ascii_lowercase());
    let mut shown = format!("{kind}/{subtype}");
    for parameter in parts {
        let (name, value) = parameter.split_once('=')?;
        shown.push(';');
        shown.push_str(name.trim());
        shown.push('=');
        shown.push_str(value.trim());
    }
    Some(Media { kind, subtype, shown })
}

/// The JSON reply with `status`.
fn json(status: u16, body: Vec<u8>) -> Response {
    Response::with_status(status)
        .header("Content-Type", JSON)
        .with_body(body)
}

/// The reply that carries `error` in the format of `surface`.
pub fn error_response(error: &ApiError, surface: Surface) -> Response {
    json(error.status_on(surface), error.body(surface))
}

/// The answer of every route that needs no generation, and the placeholder of the others.
#[derive(Clone)]
pub struct Router {
    backend: Arc<dyn Backend>,
    served: Served,
    limits: Limits,
}

impl Debug for Router {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Router")
            .field("served", &self.served)
            .field("limits", &self.limits)
            .finish()
    }
}

impl Router {
    pub fn new(backend: Arc<dyn Backend>, model_name: String, created_at: i64, limits: Limits) -> Router {
        Router {
            backend,
            served: Served {
                name: model_name,
                created_at,
            },
            limits,
        }
    }

    /// The whole answer to `request`.
    pub fn answer(&self, request: &Request) -> Response {
        let target = request.target.as_str();
        // An absolute-form target is the path after its authority.
        let target = target
            .strip_prefix("http://")
            .or_else(|| target.strip_prefix("https://"))
            .map_or(target, |rest| rest.find('/').map_or("/", |at| &rest[at..]));
        let path = target.split(['?', '#']).next().unwrap_or(target);
        let surface = Surface::of_path(path);
        let method = request.method.as_str();
        match resolve(&request.method, path) {
            Resolution::NotFound => error_response(&ApiError::unknown_url(method, path), surface),
            Resolution::MethodNotAllowed => error_response(&ApiError::method_not_supported(method), surface),
            Resolution::Options(allow) => Response::with_status(200).header("Allow", allow),
            Resolution::Found(route) => self.serve(route, request, surface),
        }
    }

    fn serve(&self, route: Route, request: &Request, surface: Surface) -> Response {
        let anthropic_form = request.headers.contains("anthropic-version");
        match route {
            Route::Health => {
                if let Some(refusal) = refuse_accept(request) {
                    return error_response(&refusal, surface);
                }
                let (status, body) = health::report(self.backend.is_available(), &self.served.name);
                json(status, body.into_bytes())
            }
            Route::Models => json(200, self.served.list(anthropic_form).into_bytes()),
            Route::Model(name) => match self.served.one(&name, anthropic_form) {
                Some(body) => json(200, body.into_bytes()),
                None => error_response(&ApiError::model_not_found(&name), surface),
            },
            Route::ChatCompletions | Route::Messages | Route::CountTokens | Route::Responses => {
                match self.generating(request) {
                    Ok(()) => error_response(&not_implemented(), surface),
                    Err(error) => error_response(&error, surface),
                }
            }
        }
    }

    /// The checks every generating route makes before it reads the body, in the order the web layer made them.
    fn generating(&self, request: &Request) -> Result<(), ApiError> {
        require_json(request)?;
        self.limits.check_body(request.body.len())?;
        if !self.backend.is_available() {
            return Err(ApiError::engine_unavailable("inference engine is shutting down"));
        }
        Ok(())
    }
}

/// The reply of a generating route until its change lands.
fn not_implemented() -> ApiError {
    ApiError {
        status: 501,
        kind: "server_error",
        code: Some("not_implemented"),
        param: None,
        message: "This route is not implemented yet.".to_owned(),
    }
}

/// 415 unless the request's content type is `application/json`.
fn require_json(request: &Request) -> Result<(), ApiError> {
    let Some(text) = request
        .headers
        .get("content-type")
        .filter(|text| !text.trim().is_empty())
    else {
        return Err(ApiError::unsupported_media_type("application/octet-stream"));
    };
    let Some(media) = media_type(text) else {
        return Err(ApiError::unreadable_content_type());
    };
    if media.kind == "application" && media.subtype == "json" {
        Ok(())
    } else {
        Err(ApiError::unsupported_media_type(&media.shown))
    }
}

/// 406 when `Accept` allows no JSON; an absent or empty `Accept` allows everything.
fn refuse_accept(request: &Request) -> Option<ApiError> {
    let accept: Vec<&str> = request
        .headers
        .get_all("accept")
        .flat_map(|line| line.split(','))
        .map(str::trim)
        .collect();
    if accept.iter().all(|item| item.is_empty()) {
        return None;
    }
    let mut acceptable = false;
    for item in accept.iter().filter(|item| !item.is_empty()) {
        let Some(media) = media_type(item) else {
            return Some(ApiError::unreadable_accept());
        };
        let kind = media.kind.as_str();
        let subtype = media.subtype.as_str();
        acceptable |=
            (kind == "*" && subtype == "*") || (kind == "application" && (subtype == "*" || subtype == "json"));
    }
    (!acceptable).then(|| ApiError::not_acceptable(&[JSON]))
}

/// The server's handler: answers each request at once.
impl Handler for Router {
    fn handle(&self, request: Request, reply: Responder) -> Box<dyn Frame> {
        reply.send(self.answer(&request));
        no_work()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_resolves_to_its_route_by_method() {
        assert_eq!(resolve(&Method::Get, "/health"), Resolution::Found(Route::Health));
        assert_eq!(resolve(&Method::Head, "/v1/models"), Resolution::Found(Route::Models));
        assert_eq!(
            resolve(&Method::Get, "/v1/models/m"),
            Resolution::Found(Route::Model("m".into()))
        );
        assert_eq!(
            resolve(&Method::Post, "/v1/messages/count_tokens"),
            Resolution::Found(Route::CountTokens)
        );
        assert_eq!(
            resolve(&Method::Post, "/v1/responses"),
            Resolution::Found(Route::Responses)
        );
        assert_eq!(resolve(&Method::Post, "/v1/models"), Resolution::MethodNotAllowed);
        assert_eq!(resolve(&Method::Head, "/v1/messages"), Resolution::MethodNotAllowed);
        assert_eq!(resolve(&Method::Get, "/v1/models/"), Resolution::NotFound);
        assert_eq!(resolve(&Method::Get, "/v1/models/a/b"), Resolution::NotFound);
        assert_eq!(resolve(&Method::Put, "/nothing"), Resolution::MethodNotAllowed);
        assert_eq!(
            resolve(&Method::Other("FOO".into()), "/health"),
            Resolution::MethodNotAllowed
        );
    }

    #[test]
    fn percent_escapes_are_decoded_and_bad_ones_kept() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(percent_decode("%e2%82%ac"), "\u{20ac}");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn media_types_are_read_as_the_web_layer_read_them() {
        let shown = |text: &str| media_type(text).map(|media| media.shown);
        assert_eq!(
            shown("Text/Plain; charset=UTF-8").as_deref(),
            Some("text/plain;charset=UTF-8")
        );
        assert_eq!(shown("application/json").as_deref(), Some("application/json"));
        assert_eq!(shown("*/*").as_deref(), Some("*/*"));
        assert_eq!(shown("garbage"), None);
        assert_eq!(shown("a/b;c"), None);
        assert_eq!(shown("/b"), None);
    }
}
