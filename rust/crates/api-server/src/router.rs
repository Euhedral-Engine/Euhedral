//! The router: a method and a path become a route, and anything else becomes the JSON error of the path's surface.
//!
//! An unknown URL is 404 whatever the method. A known path with a method it does not take is 405. `OPTIONS` on a
//! known path is 200 with `Allow`. A generation request whose body is not JSON is 415, and an `Accept` that excludes
//! what a route produces is 406. The 405, 415 and 406 answers carry only a content type, as every error does; the
//! `Allow` header is sent only with an `OPTIONS` answer.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_inference_api_protocol::{ApiError, Surface};
use euhedral_net::{Handler, Method, Request, Responder, Response, no_work};

use crate::answer::Answerer;
use crate::backend::Backend;
use crate::health;
use crate::limits::Limits;
use crate::metrics::{CONTENT_TYPE as METRICS_TYPE, Metrics};
use crate::models::Served;
use crate::planning::{Endpoint, Planner};

const JSON: &str = "application/json";

/// A route that has a handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Health,
    /// `GET /metrics`: the series of the server and the engine as Prometheus text.
    Metrics,
    Models,
    /// `GET /v1/models/{model}`, with the model percent-decoded.
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

/// The route of a path and whether it is a GET route (it also answers HEAD) rather than a POST one. The path is read
/// by segments, each percent-decoded, so `/v1/%6Dodels` is `/v1/models`.
fn route_of(path: &str) -> Option<(Route, bool)> {
    let segments: Vec<String> = path.strip_prefix('/')?.split('/').map(percent_decode).collect();
    let segments: Vec<&str> = segments.iter().map(String::as_str).collect();
    match segments[..] {
        ["health"] => Some((Route::Health, true)),
        ["metrics"] => Some((Route::Metrics, true)),
        ["v1", "models"] => Some((Route::Models, true)),
        ["v1", "models", model] if !model.is_empty() => Some((Route::Model(model.to_owned()), true)),
        ["v1", "chat", "completions"] => Some((Route::ChatCompletions, false)),
        ["v1", "messages"] => Some((Route::Messages, false)),
        ["v1", "messages", "count_tokens"] => Some((Route::CountTokens, false)),
        ["v1", "responses"] => Some((Route::Responses, false)),
        _ => None,
    }
}

/// Maps a method and the raw path (no query) to what answers it.
pub fn resolve(method: &Method, path: &str) -> Resolution {
    let is_get = matches!(method, Method::Get | Method::Head);
    let Some((route, takes_get)) = route_of(path) else {
        return Resolution::NotFound;
    };
    match method {
        Method::Options => Resolution::Options(if takes_get { "GET,HEAD,OPTIONS" } else { "POST,OPTIONS" }),
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

/// A media type of the HTTP types, read as clients write them: `type/subtype` in lower case, and the text it prints for it (the
/// parameters as written, joined by `;`).
struct Media {
    kind: String,
    subtype: String,
    shown: String,
}

/// Whether `text` is a token: printable ASCII with no separator.
fn is_token(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|b| b.is_ascii_graphic() && !b"()<>@,;:\\\"/[]?={}".contains(&b))
}

/// Whether `text` is a quoted string: at least two characters, between two quotes of one kind.
fn is_quoted(text: &str) -> bool {
    text.len() >= 2 && (text.starts_with('"') && text.ends_with('"') || text.starts_with('\'') && text.ends_with('\''))
}

/// Splits a list of media types at the commas outside double quotes; a backslash passes the character after it.
fn tokenize(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut quoted = false;
    let mut start = 0;
    let mut chars = text.char_indices();
    while let Some((at, ch)) = chars.next() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                parts.push(&text[start..at]);
                start = at + 1;
            }
            '\\' => {
                chars.next();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Splits the parameters of one media type at the semicolons outside double quotes.
fn parameters_of(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut quoted = false;
    let mut start = 0;
    let mut previous = ' ';
    for (at, ch) in text.char_indices() {
        if ch == '"' && previous != '\\' {
            quoted = !quoted;
        } else if ch == ';' && !quoted {
            parts.push(&text[start..at]);
            start = at + 1;
        }
        previous = ch;
    }
    parts.push(&text[start..]);
    parts
}

/// Reads one media type. A bare `*` is `*/*`; a parameter without `=` is ignored; a `q` outside 0 to 1 is an error.
fn media_type(text: &str) -> Option<Media> {
    let (head, rest) = match text.split_once(';') {
        Some((head, rest)) => (head, Some(rest)),
        None => (text, None),
    };
    let full = head.trim();
    let full = if full == "*" { "*/*" } else { full };
    let (kind, subtype) = full.split_once('/')?;
    if subtype.is_empty() || !is_token(kind) || !is_token(subtype) || (kind == "*" && subtype != "*") {
        return None;
    }
    let mut parameters: Vec<(String, String)> = Vec::new();
    for parameter in parameters_of(rest.unwrap_or("")) {
        let Some((name, value)) = parameter.trim().split_once('=') else {
            continue;
        };
        let (name, value) = (name.trim(), value.trim());
        if !is_token(name) || !(is_quoted(value) || is_token(value)) {
            return None;
        }
        if name == "q" {
            let quality = value.trim_matches(['"', '\'']);
            let quality: f64 = quality.trim_end_matches(['d', 'D', 'f', 'F']).parse().ok()?;
            if !(0.0..=1.0).contains(&quality) {
                return None;
            }
        }
        match parameters
            .iter()
            .position(|(known, _)| known.eq_ignore_ascii_case(name))
        {
            Some(at) if parameters[at].0 == name => parameters[at].1 = value.to_owned(),
            Some(at) => {
                parameters.remove(at);
                parameters.push((name.to_owned(), value.to_owned()));
            }
            None => parameters.push((name.to_owned(), value.to_owned())),
        }
    }
    let (kind, subtype) = (kind.to_ascii_lowercase(), subtype.to_ascii_lowercase());
    let mut shown = format!("{kind}/{subtype}");
    for (name, value) in &parameters {
        shown.push_str(&format!(";{name}={value}"));
    }
    Some(Media { kind, subtype, shown })
}

impl Media {
    /// The subtype is `*` or `*+suffix`.
    fn wildcard_subtype(&self) -> bool {
        self.subtype == "*" || self.subtype.starts_with("*+")
    }

    fn suffix(&self) -> Option<&str> {
        self.subtype.rsplit_once('+').map(|(_, suffix)| suffix)
    }

    /// Whether the two types can be the same one, with wildcards and `+suffix` families.
    fn compatible_with(&self, other: &Media) -> bool {
        if self.kind == "*" || other.kind == "*" {
            return true;
        }
        if self.kind != other.kind {
            return false;
        }
        if self.subtype == other.subtype {
            return true;
        }
        if !self.wildcard_subtype() && !other.wildcard_subtype() {
            return false;
        }
        if self.subtype == "*" || other.subtype == "*" {
            return true;
        }
        if self.wildcard_subtype()
            && let Some(suffix) = self.suffix()
        {
            return suffix == other.subtype || Some(suffix) == other.suffix();
        }
        if other.wildcard_subtype()
            && let Some(suffix) = other.suffix()
        {
            return suffix == self.subtype || Some(suffix) == self.suffix();
        }
        false
    }
}

/// The JSON reply with `status`.
pub(crate) fn json(status: u16, body: Vec<u8>) -> Response {
    Response::with_status(status)
        .header("Content-Type", JSON)
        .with_body(body)
}

/// The reply that carries `error` in the format of `surface`.
pub fn error_response(error: &ApiError, surface: Surface) -> Response {
    json(error.status_on(surface), error.body(surface))
}

/// The answer of every route that needs no generation; the others are planned here and answered by the answerer.
#[derive(Clone)]
pub struct Router {
    backend: Arc<dyn Backend>,
    served: Served,
    limits: Limits,
    metrics: Arc<Metrics>,
    planner: Planner,
    answerer: Answerer,
}

/// What a route that lists the model produces: `application/json` and its `+json` family.
const MODEL_TYPES: [&str; 2] = [JSON, "application/*+json"];

impl Debug for Router {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Router")
            .field("served", &self.served)
            .field("limits", &self.limits)
            .finish()
    }
}

impl Router {
    pub fn new(
        backend: Arc<dyn Backend>,
        created_at: i64,
        limits: Limits,
        metrics: Arc<Metrics>,
        planner: Planner,
        answerer: Answerer,
    ) -> Router {
        Router {
            backend,
            served: Served { created_at },
            limits,
            metrics,
            planner,
            answerer,
        }
    }

    /// The path of `request` without its query, and the surface that answers its errors.
    fn locate(request: &Request) -> (&str, Surface) {
        let target = request.target.as_str();
        // An absolute-form target is the path after its authority.
        let target = target
            .strip_prefix("http://")
            .or_else(|| target.strip_prefix("https://"))
            .map_or(target, |rest| rest.find('/').map_or("/", |at| &rest[at..]));
        let path = target.split(['?', '#']).next().unwrap_or(target);
        (path, Surface::of_path(path))
    }

    /// The whole answer to `request`, or the work that answers it: the first frame of the chain of a generating
    /// route whose request passed the checks made before its body is read.
    fn route(&self, request: Request, reply: Responder) -> Box<dyn Frame> {
        let (path, surface) = Router::locate(&request);
        let method = request.method.as_str();
        let response = match resolve(&request.method, path) {
            Resolution::NotFound => error_response(&ApiError::unknown_url(method, path), surface),
            Resolution::MethodNotAllowed => error_response(&ApiError::method_not_supported(method), surface),
            Resolution::Options(allow) => Response::with_status(200).header("Allow", allow),
            Resolution::Found(Route::ChatCompletions) => {
                return self.generating(Some(Endpoint::ChatCompletions), request, reply, surface);
            }
            Resolution::Found(Route::Messages) => {
                return self.generating(Some(Endpoint::Messages), request, reply, surface);
            }
            Resolution::Found(Route::Responses) => {
                return self.generating(Some(Endpoint::Responses), request, reply, surface);
            }
            Resolution::Found(Route::CountTokens) => {
                return self.generating(None, request, reply, surface);
            }
            Resolution::Found(route) => self.serve(route, &request, surface),
        };
        reply.send(response);
        no_work()
    }

    /// The first frame of a generating route's chain once the checks made before the body is read pass, or the error
    /// that ends the request. `endpoint` `None` counts the tokens of a Messages request.
    fn generating(
        &self,
        endpoint: Option<Endpoint>,
        request: Request,
        reply: Responder,
        surface: Surface,
    ) -> Box<dyn Frame> {
        if let Err(error) = self.check_generating(&request) {
            reply.send(error_response(&error, surface));
            return no_work();
        }
        match endpoint {
            Some(endpoint) => {
                let arrived = Instant::now();
                let answerer = self.answerer.clone();
                self.planner.plan(
                    endpoint,
                    request.body,
                    Box::new(move |planned| match planned {
                        Ok(planned) => answerer.answer(planned, reply, surface, arrived),
                        Err(error) => {
                            answerer.planning_failed(endpoint, &error);
                            reply.send(error_response(&error, surface));
                        }
                    }),
                )
            }
            None => self.planner.count_tokens(
                request.body,
                Box::new(move |counted| {
                    reply.send(match counted {
                        Ok(tokens) => json(200, format!("{{\"input_tokens\":{tokens}}}").into_bytes()),
                        Err(error) => error_response(&error, surface),
                    });
                }),
            ),
        }
    }

    fn serve(&self, route: Route, request: &Request, surface: Surface) -> Response {
        let anthropic_form = request.headers.contains("anthropic-version");
        match route {
            Route::Health => {
                if let Some(refusal) = refuse_accept(request, &[JSON]) {
                    return error_response(&refusal, surface);
                }
                let (status, body) = health::report(self.backend.is_available(), self.backend.model_id());
                json(status, body.into_bytes())
            }
            Route::Metrics => {
                if let Some(refusal) = refuse_accept(request, &[METRICS_TYPE]) {
                    return error_response(&refusal, surface);
                }
                let counts = self.answerer.counts();
                self.metrics.set_queue(counts.active, counts.waiting);
                let text = self.metrics.render(
                    self.backend.model_id(),
                    self.backend.context_length(),
                    &self.backend.engine_report(),
                    Metrics::wall_clock_ms(),
                );
                Response::with_status(200)
                    .header("Content-Type", METRICS_TYPE)
                    .with_body(text.into_bytes())
            }
            Route::Models => {
                if let Some(refusal) = refuse_accept(request, &MODEL_TYPES) {
                    return error_response(&refusal, surface);
                }
                json(
                    200,
                    self.served.list(self.backend.model_id(), anthropic_form).into_bytes(),
                )
            }
            Route::Model(name) => match self.served.one(self.backend.model_id(), &name, anthropic_form) {
                Some(body) => match refuse_accept(request, &MODEL_TYPES) {
                    Some(refusal) => error_response(&refusal, surface),
                    None => json(200, body.into_bytes()),
                },
                None => error_response(&ApiError::model_not_found(&name), surface),
            },
            // The generating routes are started by `route`, which has the body's owner and the responder.
            Route::ChatCompletions | Route::Messages | Route::CountTokens | Route::Responses => {
                error_response(&ApiError::server_error(), surface)
            }
        }
    }

    /// The checks every generating route makes before it reads the body, in the order a client sees their errors: type, size, readiness.
    fn check_generating(&self, request: &Request) -> Result<(), ApiError> {
        require_json(request)?;
        self.limits.check_body(request.body.len())?;
        if !self.backend.is_available() {
            return Err(ApiError::engine_unavailable("inference engine is shutting down"));
        }
        Ok(())
    }
}

/// 415 unless the request's content type is `application/json`. A missing or empty header is read as no type at all.
fn require_json(request: &Request) -> Result<(), ApiError> {
    let Some(text) = request.headers.get("content-type").filter(|text| !text.is_empty()) else {
        return Err(ApiError::unsupported_media_type("null"));
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

/// 406 when `Accept` allows none of `produces`; an absent or empty `Accept` allows everything, and one that cannot be
/// read is refused too.
fn refuse_accept(request: &Request, produces: &[&str]) -> Option<ApiError> {
    let lines: Vec<&str> = request.headers.get_all("accept").collect();
    let mut accepted = Vec::new();
    for item in tokenize(&lines.join(",")) {
        if item.trim().is_empty() {
            continue;
        }
        match media_type(item) {
            Some(media) => accepted.push(media),
            None => return Some(ApiError::unreadable_accept()),
        }
    }
    if accepted.is_empty() {
        return None;
    }
    let produced: Vec<Media> = produces.iter().filter_map(|text| media_type(text)).collect();
    let fits = accepted
        .iter()
        .any(|wanted| produced.iter().any(|offered| wanted.compatible_with(offered)));
    (!fits).then(|| ApiError::not_acceptable(produces))
}

/// The server's handler: answers a request at once, or returns the first frame of the chain that plans it.
impl Handler for Router {
    fn handle(&self, request: Request, reply: Responder) -> Box<dyn Frame> {
        self.route(request, reply)
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
        assert_eq!(resolve(&Method::Put, "/nothing"), Resolution::NotFound);
        assert_eq!(resolve(&Method::Post, "/nothing"), Resolution::NotFound);
        assert_eq!(resolve(&Method::Options, "/nothing"), Resolution::NotFound);
        assert_eq!(resolve(&Method::Get, "/v1/%6Dodels"), Resolution::Found(Route::Models));
        assert_eq!(
            resolve(&Method::Options, "/v1/models"),
            Resolution::Options("GET,HEAD,OPTIONS")
        );
        assert_eq!(
            resolve(&Method::Options, "/v1/responses"),
            Resolution::Options("POST,OPTIONS")
        );
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
    fn media_types_are_read_with_their_parameters_and_wildcards() {
        let shown = |text: &str| media_type(text).map(|media| media.shown);
        assert_eq!(
            shown("Text/Plain; charset=UTF-8").as_deref(),
            Some("text/plain;charset=UTF-8")
        );
        assert_eq!(shown("application/json").as_deref(), Some("application/json"));
        assert_eq!(shown("*/*").as_deref(), Some("*/*"));
        assert_eq!(shown("garbage"), None);
        assert_eq!(shown("a/b;c").as_deref(), Some("a/b"));
        assert_eq!(shown("/b"), None);
    }

    /// Whether `Accept: header` (none: no header) lets a route that produces `produces` answer.
    fn allows(header: Option<&str>, produces: &[&str]) -> Result<bool, String> {
        let mut request = Request::new(Method::Get, "/");
        if let Some(header) = header {
            request = request.header("Accept", header);
        }
        match refuse_accept(&request, produces) {
            None => Ok(true),
            Some(error) if error.status == 406 && error.message.starts_with("Acceptable") => Ok(false),
            Some(error) => Err(error.message),
        }
    }

    #[test]
    fn an_accept_header_is_read_as_clients_write_it() {
        let json = [JSON];
        let family = MODEL_TYPES;
        let unreadable = Err("Could not parse Accept header.".to_owned());
        type Case<'a> = (Option<&'a str>, &'a [&'a str], Result<bool, String>);
        let table: [Case; 28] = [
            (None, &json, Ok(true)),
            (Some(""), &json, Ok(true)),
            (Some(" , "), &json, Ok(true)),
            (Some("*/*"), &json, Ok(true)),
            (Some("*"), &json, Ok(true)),
            (Some("*; q=.2"), &json, Ok(true)),
            (Some("application/*"), &json, Ok(true)),
            (Some("application/json"), &json, Ok(true)),
            (Some("APPLICATION/JSON;charset=utf-8"), &json, Ok(true)),
            (Some("text/plain"), &json, Ok(false)),
            (Some("text/*"), &json, Ok(false)),
            (Some("text/html, application/json;q=0.5"), &json, Ok(true)),
            (Some("application/*+json"), &json, Ok(true)),
            (Some("application/vnd.x+json"), &json, Ok(false)),
            (Some("application/vnd.x+json"), &family, Ok(true)),
            (Some("application/*+xml"), &family, Ok(false)),
            (Some("text/plain"), &family, Ok(false)),
            (Some("text/plain, nonsense"), &json, unreadable.clone()),
            (Some("nonsense"), &json, unreadable.clone()),
            (Some("*/json"), &json, unreadable.clone()),
            (Some("application/"), &json, unreadable.clone()),
            (Some("text/plain;q=2"), &json, unreadable.clone()),
            (Some("text/plain;q=abc"), &json, unreadable.clone()),
            (Some("text/plain;a="), &json, unreadable.clone()),
            (Some("text/plain;a=b c"), &json, unreadable.clone()),
            (Some("text/plain;ignored, application/json"), &json, Ok(true)),
            (Some("text/plain;note=\"a,b;c\", application/json"), &json, Ok(true)),
            (Some("application/js on"), &json, unreadable),
        ];
        for (header, produces, expected) in table {
            assert_eq!(allows(header, produces), expected, "Accept: {header:?}");
        }
    }

    #[test]
    fn several_accept_lines_are_one_list() {
        let request = Request::new(Method::Get, "/")
            .header("Accept", "text/plain")
            .header("Accept", "application/json");
        assert!(refuse_accept(&request, &[JSON]).is_none());
    }

    #[test]
    fn a_media_type_is_printed_with_its_parameters_as_written() {
        let shown = |text: &str| media_type(text).map(|media| media.shown);
        assert_eq!(shown("a/b;c=d;e").as_deref(), Some("a/b;c=d"));
        assert_eq!(shown("a/b;X=1;x=2").as_deref(), Some("a/b;x=2"));
        assert_eq!(shown("a/b;q=\"0.5\"").as_deref(), Some("a/b;q=\"0.5\""));
        assert_eq!(shown("*").as_deref(), Some("*/*"));
    }
}
