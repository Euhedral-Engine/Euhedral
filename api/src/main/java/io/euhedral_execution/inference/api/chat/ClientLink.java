package io.euhedral_execution.inference.api.chat;

/// Whether the client of a request is still connected, asked without writing to it.
///
/// `gone` may be asked from workers at any time until `close`, which the request's completion calls; after it,
/// `gone` answers false without touching the connection.
public interface ClientLink {
    ClientLink NONE = new ClientLink() {
        @Override
        public boolean gone() {
            return false;
        }

        @Override
        public void close() {}
    };

    boolean gone();

    void close();
}
