package io.euhedral_execution.inference.quantization.mixed;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.euhedral_execution.inference.quantization.checkpoint.ResumeFile;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;

/// What a search has measured and where it stands, in one [ResumeFile] that is overwritten in place.
///
/// The header holds the search's fingerprint, the round in progress, the explore share, the rounds without a
/// new front point, the last round's held-out error, the seed, the device's free bytes at the start, and how
/// many measured candidates the file holds and where they end. Each candidate is appended after the previous
/// ones as a length and its JSON, and the header is rewritten to cover it only once the record is on disk, so
/// a stop part way through a record leaves the file as it was before that candidate. A candidate already
/// recorded is refused, so a restarted run never scores one twice nor records it twice.
public final class SearchState implements AutoCloseable {

    /// A candidate that was scored: its allocation, KL against the reference and the KL's standard error, the
    /// bytes of the composite artifact, the cost table's prefill microseconds per token and verification
    /// microseconds, whether the plan at the target context keeps every object on the device, and the round
    /// and origin that proposed it.
    public record Measured(
            Allocation a,
            double kl,
            double klSe,
            long bytes,
            double prefill,
            double decode,
            boolean fits,
            int round,
            String origin) {}

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int LENGTH_BYTES = Integer.BYTES;

    private final ResumeFile file;
    private final Properties header;
    private final List<Measured> measured = new ArrayList<>();
    private final Set<Allocation> known = new HashSet<>();
    private long end = ResumeFile.HEADER;

    private SearchState(ResumeFile file, Properties header) {
        this.file = file;
        this.header = header;
    }

    /// The state in `path`, or a new one with `seed` and `freeBytes` recorded.
    ///
    /// @throws IllegalStateException when the file belongs to a different search (another `fingerprint`)
    public static SearchState open(Path path, String fingerprint, long seed, long freeBytes) throws IOException {
        ResumeFile file = new ResumeFile(path);
        try {
            Properties header = file.header();
            SearchState state = new SearchState(file, header);
            if (header.isEmpty()) {
                header.setProperty("kind", "mixed-search");
                header.setProperty("fingerprint", fingerprint);
                header.setProperty("seed", Long.toString(seed));
                header.setProperty("freeBytes", Long.toString(freeBytes));
                header.setProperty("count", "0");
                header.setProperty("end", Long.toString(ResumeFile.HEADER));
                file.header(header);
                return state;
            }
            if (!"mixed-search".equals(header.getProperty("kind"))
                    || !fingerprint.equals(header.getProperty("fingerprint")))
                throw new IllegalStateException(path + " holds a different search; give --state a new file");
            state.load();
            return state;
        } catch (IOException | RuntimeException failure) {
            file.close();
            throw failure;
        }
    }

    private void load() throws IOException {
        int count = Integer.parseInt(header.getProperty("count"));
        long recorded = Long.parseLong(header.getProperty("end"));
        try (Arena arena = Arena.ofConfined()) {
            long at = ResumeFile.HEADER;
            MemorySegment size = arena.allocate(LENGTH_BYTES);
            for (int i = 0; i < count; i++) {
                at = file.read(at, size);
                int length = size.get(ValueLayout.JAVA_INT_UNALIGNED, 0);
                MemorySegment body = arena.allocate(length);
                at = file.read(at, body);
                Measured read = decode(new String(body.toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8));
                measured.add(read);
                known.add(read.a());
            }
            if (at != recorded)
                throw new IllegalStateException("the search state's records end at " + at + ", not " + recorded);
        }
        end = recorded;
    }

    /// Records `candidate`; false, and nothing recorded, when its allocation is already.
    public boolean add(Measured candidate) throws IOException {
        if (!known.add(candidate.a())) return false;
        byte[] body = encode(candidate).getBytes(StandardCharsets.UTF_8);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment record = arena.allocate(LENGTH_BYTES + body.length);
            record.set(ValueLayout.JAVA_INT_UNALIGNED, 0, body.length);
            MemorySegment.copy(MemorySegment.ofArray(body), 0, record, LENGTH_BYTES, body.length);
            file.write(end, record);
            end += record.byteSize();
        }
        measured.add(candidate);
        header.setProperty("count", Integer.toString(measured.size()));
        header.setProperty("end", Long.toString(end));
        file.header(header);
        return true;
    }

    /// Writes bytes after the recorded end without recording them, as a stop during a record would leave.
    void writeUnrecordedBytesForTest(byte[] bytes) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            file.write(end, arena.allocateFrom(ValueLayout.JAVA_BYTE, bytes));
        }
    }

    private static String encode(Measured m) {
        ObjectNode node = JSON.createObjectNode();
        ObjectNode objects = node.putObject("objects");
        new TreeMap<>(m.a().choices()).forEach((name, option) -> objects.put(name, option.key()));
        node.put("kl", m.kl());
        node.put("klSe", m.klSe());
        node.put("bytes", m.bytes());
        node.put("prefill", m.prefill());
        node.put("decode", m.decode());
        node.put("fits", m.fits());
        node.put("round", m.round());
        node.put("origin", m.origin());
        return node.toString();
    }

    private static Measured decode(String json) throws IOException {
        JsonNode node = JSON.readTree(json);
        Map<String, Option> choices = new TreeMap<>();
        for (var entry : node.get("objects").properties())
            choices.put(entry.getKey(), Option.of(entry.getValue().asText()));
        return new Measured(
                new Allocation(choices),
                node.get("kl").asDouble(),
                node.get("klSe").asDouble(),
                node.get("bytes").asLong(),
                node.get("prefill").asDouble(),
                node.get("decode").asDouble(),
                node.get("fits").asBoolean(),
                node.get("round").asInt(),
                node.get("origin").asText());
    }

    /// Every measured candidate, in the order they were measured.
    public List<Measured> measured() {
        return List.copyOf(measured);
    }

    /// Whether `allocation` was measured.
    public boolean contains(Allocation allocation) {
        return known.contains(allocation);
    }

    public long seed() {
        return Long.parseLong(header.getProperty("seed"));
    }

    public long freeBytes() {
        return Long.parseLong(header.getProperty("freeBytes"));
    }

    /// The round in progress: 0 while the seeds are measured.
    public int round() {
        return Integer.parseInt(header.getProperty("round", "0"));
    }

    public void round(int round) throws IOException {
        put("round", Integer.toString(round));
    }

    /// The share of each batch proposed for what the ensemble is unsure of.
    public double explore() {
        return Double.parseDouble(header.getProperty("explore", "NaN"));
    }

    public void explore(double share) throws IOException {
        put("explore", Double.toString(share));
    }

    /// Rounds in a row without a new front point.
    public int stale() {
        return Integer.parseInt(header.getProperty("stale", "0"));
    }

    public void stale(int rounds) throws IOException {
        put("stale", Integer.toString(rounds));
    }

    /// The held-out error of the last round that finished, NaN before the first.
    public double heldOut() {
        return Double.parseDouble(header.getProperty("heldOut", "NaN"));
    }

    public void heldOut(double error) throws IOException {
        put("heldOut", Double.toString(error));
    }

    private void put(String key, String value) throws IOException {
        header.setProperty(key, value);
        file.header(header);
    }

    @Override
    public void close() throws IOException {
        file.close();
    }
}
