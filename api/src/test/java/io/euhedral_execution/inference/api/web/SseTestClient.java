package io.euhedral_execution.inference.api.web;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/// Minimal HTTP/1.1 client over a raw socket so tests control exactly when the connection drops.
final class SseTestClient implements AutoCloseable {
    private final Socket socket;
    private final BufferedReader reader;

    private SseTestClient(Socket socket) throws IOException {
        this.socket = socket;
        this.socket.setSoTimeout(30_000);
        this.reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
    }

    static SseTestClient post(int port, String path, String json) throws IOException {
        var client = new SseTestClient(new Socket("localhost", port));
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        OutputStream output = client.socket.getOutputStream();
        output.write(("POST " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                        + "Accept: application/json\r\nContent-Length: " + body.length + "\r\n\r\n")
                .getBytes(StandardCharsets.US_ASCII));
        output.write(body);
        output.flush();
        return client;
    }

    /// Reads the raw response (headers and chunked framing included) until a line matches.
    List<String> readUntil(Predicate<String> stop) throws IOException {
        List<String> lines = new ArrayList<>();
        String line;
        while ((line = this.reader.readLine()) != null) {
            lines.add(line);
            if (stop.test(line)) return lines;
        }
        throw new IOException("connection closed before the expected line; saw " + lines);
    }

    /// Closes the connection in order (FIN), as a client that gives up and closes its socket would.
    void closeGracefully() throws IOException {
        this.socket.close();
    }

    /// Drops the TCP connection, as a client that navigates away or is killed would.
    @Override
    public void close() throws IOException {
        if (this.socket.isClosed()) return;
        this.socket.setSoLinger(true, 0);
        this.socket.close();
    }
}
