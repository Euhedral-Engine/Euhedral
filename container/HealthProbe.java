import java.net.HttpURLConnection;
import java.net.URI;
import java.util.Map;

/// Container health probe; uses only the Java runtime, not a shell or curl.
public final class HealthProbe {
    public static void main(String[] args) throws Exception {
        Map<String, String> environment = System.getenv();
        String configured = environment.getOrDefault("EUHEDRAL_SERVER_PORT", environment.get("PORT"));
        int port = Integer.parseInt(configured == null ? "1738" : configured);
        HttpURLConnection connection = (HttpURLConnection) URI.create("http://127.0.0.1:" + port + "/health")
                .toURL()
                .openConnection();
        connection.setConnectTimeout(2000);
        connection.setReadTimeout(2000);
        try {
            if (connection.getResponseCode() != 200) System.exit(1);
        } finally {
            connection.disconnect();
        }
    }
}
