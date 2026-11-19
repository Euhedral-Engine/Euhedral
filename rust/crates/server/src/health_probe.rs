//! `euhedral-health-probe`: the container's health check. The image has no shell and no curl, so the check is this
//! binary: it asks `GET /health` of the server on this machine and exits 0 on `200`, else 1.
//!
//! The port is `EUHEDRAL_SERVER_PORT`, else `PORT`, else 1738, the variables and the default the server reads. Connecting,
//! writing and reading each give up after two seconds.

use std::env::var;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::process::ExitCode;
use std::time::Duration;

/// The port the server listens on when nothing sets it.
const DEFAULT_PORT: u16 = 1738;
/// The bound on connecting, on writing the request and on reading the answer.
const TIMEOUT: Duration = Duration::from_secs(2);

/// The port the variables name: `server_port`, else `port`, else the default.
fn port_from(server_port: Option<String>, port: Option<String>) -> Result<u16, String> {
    let Some(text) = server_port.or(port) else {
        return Ok(DEFAULT_PORT);
    };
    text.trim()
        .parse()
        .map_err(|_| format!("the port {text:?} is not a port number"))
}

/// Asks `GET /health` of `address`; `Ok` only for a `200` answer.
fn check(address: SocketAddr, timeout: Duration) -> Result<(), String> {
    let mut stream =
        TcpStream::connect_timeout(&address, timeout).map_err(|error| format!("cannot connect: {error}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| error.to_string())?;
    stream
        .write_all(format!("GET /health HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n").as_bytes())
        .map_err(|error| format!("cannot send the request: {error}"))?;
    let mut head = Vec::new();
    let mut chunk = [0_u8; 256];
    while !head.contains(&b'\n') && head.len() < 1024 {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => head.extend_from_slice(&chunk[..count]),
            Err(error) => return Err(format!("cannot read the answer: {error}")),
        }
    }
    let line = String::from_utf8_lossy(&head);
    let line = line.lines().next().unwrap_or("");
    match line.split_whitespace().nth(1) {
        Some("200") => Ok(()),
        _ => Err(format!("unhealthy: {line:?}")),
    }
}

fn main() -> ExitCode {
    let result = port_from(var("EUHEDRAL_SERVER_PORT").ok(), var("PORT").ok())
        .and_then(|port| check(SocketAddr::from((Ipv4Addr::LOCALHOST, port)), TIMEOUT));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            euhedral_inference_server::say(&message);
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread::{JoinHandle, spawn};

    /// Serves one connection: reads the request, writes `reply`, and returns the request.
    fn serve_once(reply: &'static str) -> (SocketAddr, JoinHandle<String>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let address = listener.local_addr().expect("address");
        let handle = spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = vec![0_u8; 512];
            let count = stream.read(&mut request).expect("read");
            stream.write_all(reply.as_bytes()).expect("write");
            String::from_utf8_lossy(&request[..count]).into_owned()
        });
        (address, handle)
    }

    #[test]
    fn a_healthy_reply_passes_and_asks_for_health() {
        let (address, server) = serve_once("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}");
        assert_eq!(check(address, TIMEOUT), Ok(()));
        assert!(server.join().expect("server").starts_with("GET /health HTTP/1.1\r\n"));
    }

    #[test]
    fn an_unhealthy_reply_fails() {
        let (address, server) = serve_once("HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n");
        let error = check(address, TIMEOUT).expect_err("503 is unhealthy");
        assert!(error.contains("503"), "{error}");
        server.join().expect("server");
    }

    #[test]
    fn a_reply_that_is_not_http_fails() {
        let (address, server) = serve_once("garbage\r\n");
        assert!(check(address, TIMEOUT).is_err());
        server.join().expect("server");
    }

    #[test]
    fn a_refused_connection_fails() {
        let address = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .expect("bind")
            .local_addr()
            .expect("address");
        // The listener is closed: nothing listens on the port now.
        let error = check(address, TIMEOUT).expect_err("refused");
        assert!(error.contains("cannot connect"), "{error}");
    }

    #[test]
    fn a_silent_server_times_out() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let address = listener.local_addr().expect("address");
        let error = check(address, Duration::from_millis(200)).expect_err("no answer");
        assert!(error.contains("cannot read"), "{error}");
    }

    #[test]
    fn the_port_comes_from_the_server_variable_then_port_then_the_default() {
        assert_eq!(port_from(Some("2000".into()), Some("3000".into())), Ok(2000));
        assert_eq!(port_from(None, Some("3000".into())), Ok(3000));
        assert_eq!(port_from(None, None), Ok(1738));
        assert!(port_from(Some("x".into()), None).is_err());
        assert!(port_from(Some("70000".into()), None).is_err());
    }
}
