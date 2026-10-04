package io.euhedral_execution.inference.api.web;

import io.euhedral_execution.inference.api.chat.ClientLink;
import jakarta.servlet.ReadListener;
import jakarta.servlet.ServletInputStream;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// A request's connection, probed by a non-blocking read.
///
/// Tomcat does not watch an asynchronous request's connection once its body is read, so a client that leaves is
/// noticed only when a response is written, which for a JSON response is the end of the generation. Once the
/// request is asynchronous, a no-op `ReadListener` is installed; with one installed, `available()` reads the socket
/// without blocking, and a closed or reset connection reports data available. This serves JSON responses; once a
/// stream's response is committed the probe reads nothing, and streams are watched by their writes instead. A pipelined
/// next request reports
/// data too and would read as gone; HTTP/1.1 clients do not pipeline behind a pending response.
///
/// Probes and `close` are serialized: Tomcat recycles a completed request for the next one, so no probe may run
/// after the request's completion closed the link.
final class ServletClientLink implements ClientLink {
    static final String ATTRIBUTE = ServletClientLink.class.getName();
    private static final Logger LOG = LoggerFactory.getLogger(ServletClientLink.class);

    private HttpServletRequest request;
    private ServletInputStream input;
    private boolean closed;

    /// A link for `request`, watched once the request becomes asynchronous ([Interceptor]).
    static ServletClientLink of(HttpServletRequest request) {
        var link = new ServletClientLink();
        request.setAttribute(ATTRIBUTE, link);
        return link;
    }

    /// Notes that the request is asynchronous; runs on the container thread that started it. The listener is
    /// installed by the first probe, which only a waiting or generating request makes: a request that failed while
    /// it was planned completes without one.
    synchronized void watch(HttpServletRequest request) {
        if (!this.closed && this.request == null) this.request = request;
    }

    private void install() {
        HttpServletRequest request = this.request;
        this.request = null;
        try {
            ServletInputStream stream = request.getInputStream();
            stream.setReadListener(new ReadListener() {
                @Override
                public void onDataAvailable() {}

                @Override
                public void onAllDataRead() {}

                @Override
                public void onError(Throwable failure) {}
            });
            this.input = stream;
        } catch (IOException | IllegalStateException refused) {
            LOG.debug("Client connection cannot be watched", refused);
        }
    }

    @Override
    public synchronized boolean gone() {
        if (this.closed) return false;
        if (this.input == null && this.request != null) install();
        if (this.input == null) return false;
        try {
            return this.input.available() > 0;
        } catch (IOException | RuntimeException failure) {
            return true;
        }
    }

    @Override
    public synchronized void close() {
        this.closed = true;
        this.request = null;
        this.input = null;
    }

    /// Watches the link of each generation request when its asynchronous processing starts.
    static final class Interceptor implements org.springframework.web.servlet.AsyncHandlerInterceptor {
        @Override
        public void afterConcurrentHandlingStarted(
                HttpServletRequest request, jakarta.servlet.http.HttpServletResponse response, Object handler) {
            if (request.getAttribute(ATTRIBUTE) instanceof ServletClientLink link) link.watch(request);
        }
    }
}
