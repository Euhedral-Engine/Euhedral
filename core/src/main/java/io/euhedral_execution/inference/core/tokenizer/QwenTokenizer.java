package io.euhedral_execution.inference.core.tokenizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// Immutable Qwen BPE tokenizer loaded from one checkpoint's tokenizer assets.
///
/// <p>Load it once from a directory containing {@code tokenizer.json} and
/// {@code tokenizer_config.json}, then share the instance. Encoding does not access files or
/// depend on inference execution, scheduling, GPU, or sequence state.
public final class QwenTokenizer {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int BYTE_UNICODE_OFFSET = 256;

    private final List<String> tokensById;
    private final Map<String, Integer> vocabulary;
    private final Map<Merge, Integer> mergeRanks;
    private final Map<String, Integer> controlTokenIds;
    private final Set<Integer> controlTokenIdsById;
    private final Set<Integer> specialTokenIds;
    private final Pattern pretokenPattern;
    private final Pattern controlTokenPattern;
    private final int eosTokenId;
    private final OptionalInt bosTokenId;
    private final Set<Integer> generationEosTokenIds;
    private final OptionalInt generationBosTokenId;
    private final boolean addBosToken;
    private final boolean addEosToken;
    private final int[] byteByCodePoint;
    private final int[] codePointByByte;
    private volatile byte[][] generatedTokenBytes;

    private QwenTokenizer(
            List<String> tokensById,
            Map<String, Integer> vocabulary,
            Map<Merge, Integer> mergeRanks,
            Map<String, Integer> controlTokenIds,
            Set<Integer> specialTokenIds,
            Pattern pretokenPattern,
            int eosTokenId,
            OptionalInt bosTokenId,
            Set<Integer> generationEosTokenIds,
            OptionalInt generationBosTokenId,
            boolean addBosToken,
            boolean addEosToken,
            int[] byteByCodePoint) {
        this.tokensById = List.copyOf(tokensById);
        this.vocabulary = Map.copyOf(vocabulary);
        this.mergeRanks = Map.copyOf(mergeRanks);
        this.controlTokenIds = Map.copyOf(controlTokenIds);
        this.controlTokenIdsById = Set.copyOf(controlTokenIds.values());
        this.specialTokenIds = Set.copyOf(specialTokenIds);
        this.pretokenPattern = pretokenPattern;
        this.eosTokenId = eosTokenId;
        this.bosTokenId = bosTokenId;
        this.generationEosTokenIds = Set.copyOf(generationEosTokenIds);
        this.generationBosTokenId = generationBosTokenId;
        this.addBosToken = addBosToken;
        this.addEosToken = addEosToken;
        this.byteByCodePoint = byteByCodePoint.clone();
        this.codePointByByte = codePointByByte(byteByCodePoint);
        this.controlTokenPattern = compileControlPattern(controlTokenIds.keySet());
    }

    /// Loads the Qwen tokenizer data from the selected checkpoint directory.
    public static QwenTokenizer load(Path checkpointDirectory) throws IOException {
        Objects.requireNonNull(checkpointDirectory, "checkpointDirectory");
        Path tokenizerPath = checkpointDirectory.resolve("tokenizer.json");
        Path configPath = checkpointDirectory.resolve("tokenizer_config.json");
        Path generationConfigPath = checkpointDirectory.resolve("generation_config.json");
        JsonNode tokenizer = JSON.readTree(tokenizerPath.toFile());
        JsonNode config = JSON.readTree(configPath.toFile());
        JsonNode generationConfig = JSON.readTree(generationConfigPath.toFile());

        JsonNode model = tokenizer.path("model");
        if (!"BPE".equals(model.path("type").asText())) {
            throw new IOException(
                    "Unsupported Qwen tokenizer model: " + model.path("type").asText());
        }
        if (!"NFC".equals(tokenizer.path("normalizer").path("type").asText())) {
            throw new IOException("Qwen tokenizer must use NFC normalization");
        }
        if (!"Sequence".equals(tokenizer.path("pre_tokenizer").path("type").asText())) {
            throw new IOException("Unsupported Qwen tokenizer pre-tokenizer sequence");
        }
        JsonNode pretokenizers = tokenizer.path("pre_tokenizer").path("pretokenizers");
        JsonNode split = pretokenizers.isArray() && pretokenizers.size() > 0 ? pretokenizers.get(0) : null;
        if (split == null || !"Split".equals(split.path("type").asText())) {
            throw new IOException("Unsupported Qwen tokenizer pre-tokenizer");
        }
        if (pretokenizers.size() != 2
                || !"ByteLevel".equals(pretokenizers.get(1).path("type").asText())) {
            throw new IOException("Qwen tokenizer must use byte-level BPE after its source split pattern");
        }
        String splitRegex = split.path("pattern").path("Regex").asText(null);
        if (splitRegex == null) throw new IOException("Qwen tokenizer is missing its split expression");
        if (!"Isolated".equals(split.path("behavior").asText())
                || split.path("invert").asBoolean(true)) {
            throw new IOException("Unsupported Qwen tokenizer split behavior");
        }

        int[] byteByCodePoint = byteByCodePoint();
        Map<Integer, String> tokensByIdMap = new HashMap<>();
        Map<String, Integer> vocabulary = new HashMap<>();
        IteratorFields.readVocabulary(model.path("vocab"), (token, id) -> {
            if (vocabulary.put(token, id) != null || tokensByIdMap.put(id, token) != null) {
                throw new IOException("Duplicate Qwen vocabulary token or ID");
            }
        });
        if (tokensByIdMap.isEmpty()) throw new IOException("Qwen tokenizer vocabulary is empty");
        int largestId = tokensByIdMap.keySet().stream()
                .mapToInt(Integer::intValue)
                .max()
                .orElseThrow();
        List<String> tokensById = new ArrayList<>(java.util.Collections.nCopies(largestId + 1, null));
        tokensByIdMap.forEach(tokensById::set);

        Map<Merge, Integer> mergeRanks = new HashMap<>();
        JsonNode merges = model.path("merges");
        if (!merges.isArray()) throw new IOException("Qwen tokenizer is missing BPE merges");
        for (int rank = 0; rank < merges.size(); rank++) {
            String merge = merges.get(rank).asText();
            int separator = merge.indexOf(' ');
            if (separator <= 0 || separator == merge.length() - 1) {
                throw new IOException("Invalid Qwen BPE merge at rank " + rank);
            }
            mergeRanks.put(new Merge(merge.substring(0, separator), merge.substring(separator + 1)), rank);
        }

        Map<String, Integer> controlTokenIds = new HashMap<>();
        Set<Integer> specialTokenIds = new HashSet<>();
        JsonNode addedTokens = tokenizer.path("added_tokens");
        if (!addedTokens.isArray()) throw new IOException("Qwen tokenizer is missing added-token metadata");
        for (JsonNode added : addedTokens) {
            String content = added.path("content").asText();
            int id = added.path("id").asInt(-1);
            if (id < 0 || content.isEmpty() || controlTokenIds.put(content, id) != null) {
                throw new IOException("Invalid Qwen added-token metadata");
            }
            while (tokensById.size() <= id) tokensById.add(null);
            if (tokensById.get(id) != null) throw new IOException("Added-token ID collides with vocabulary");
            tokensById.set(id, content);
            if (added.path("special").asBoolean(false)) specialTokenIds.add(id);
        }

        String eosToken = config.path("eos_token").asText(null);
        Integer eosTokenId = eosToken == null ? null : controlTokenIds.get(eosToken);
        if (eosTokenId == null || !specialTokenIds.contains(eosTokenId)) {
            throw new IOException("Qwen tokenizer EOS token is absent from special-token metadata");
        }
        String bosToken = config.path("bos_token").asText(null);
        Integer resolvedBosTokenId = bosToken == null ? null : controlTokenIds.get(bosToken);
        if (bosToken != null && resolvedBosTokenId == null) {
            throw new IOException("Qwen tokenizer BOS token is absent from added-token metadata");
        }
        Set<Integer> generationEosTokenIds = readTokenIds(generationConfig.path("eos_token_id"));
        Integer generationBosTokenId = nullableTokenId(generationConfig.path("bos_token_id"));
        for (int tokenId : generationEosTokenIds) {
            if (tokenId < 0 || tokenId >= tokensById.size() || tokensById.get(tokenId) == null) {
                throw new IOException("Qwen generation EOS ID is absent from tokenizer vocabulary: " + tokenId);
            }
        }
        if (generationBosTokenId != null
                && (generationBosTokenId < 0
                        || generationBosTokenId >= tokensById.size()
                        || tokensById.get(generationBosTokenId) == null)) {
            throw new IOException("Qwen generation BOS ID is absent from tokenizer vocabulary");
        }

        try {
            return new QwenTokenizer(
                    tokensById,
                    vocabulary,
                    mergeRanks,
                    controlTokenIds,
                    specialTokenIds,
                    Pattern.compile(splitRegex, Pattern.UNICODE_CHARACTER_CLASS),
                    eosTokenId,
                    resolvedBosTokenId == null ? OptionalInt.empty() : OptionalInt.of(resolvedBosTokenId),
                    generationEosTokenIds,
                    generationBosTokenId == null ? OptionalInt.empty() : OptionalInt.of(generationBosTokenId),
                    config.path("add_bos_token").asBoolean(false),
                    config.path("add_eos_token").asBoolean(false),
                    byteByCodePoint);
        } catch (RuntimeException failure) {
            throw new IOException("Invalid Qwen tokenizer metadata", failure);
        }
    }

    /// Encodes text exactly as the checkpoint tokenizer, without automatic BOS/EOS insertion.
    public int[] encodeText(String text) {
        Pretokens pretokens = pretokenize(text);
        return encode(pretokens, 0, pretokens.size());
    }

    /// Encodes text with only the BOS/EOS tokens enabled by tokenizer_config.json.
    ///
    /// <p>Generation-config BOS/stop IDs are exposed separately and are not inserted implicitly.
    public int[] encodeWithModelSpecialTokens(String text) {
        return withModelSpecialTokens(encodeText(text));
    }

    /// A text split into the pieces BPE encodes independently, in order: each piece is either an added
    /// control token or one NFC-normalized pre-token. Encoding the pieces in any grouping and concatenating
    /// the results in order equals [#encodeText].
    public static final class Pretokens {
        private final String[] texts;
        private final int[] controlIds;

        private Pretokens(String[] texts, int[] controlIds) {
            this.texts = texts;
            this.controlIds = controlIds;
        }

        public int size() {
            return this.texts.length;
        }
    }

    /// Splits text on control tokens, normalizes each run between them, and splits it into pre-tokens.
    public Pretokens pretokenize(String text) {
        Objects.requireNonNull(text, "text");
        List<String> texts = new ArrayList<>();
        List<Integer> controls = new ArrayList<>();
        Matcher control = this.controlTokenPattern.matcher(text);
        int cursor = 0;
        while (control.find()) {
            pretokenizeNormalized(text.substring(cursor, control.start()), texts, controls);
            texts.add(null);
            controls.add(this.controlTokenIds.get(control.group()));
            cursor = control.end();
        }
        pretokenizeNormalized(text.substring(cursor), texts, controls);
        return new Pretokens(
                texts.toArray(String[]::new),
                controls.stream().mapToInt(Integer::intValue).toArray());
    }

    /// The token IDs of pieces [`from`, `to`) of `pretokens`.
    public int[] encode(Pretokens pretokens, int from, int to) {
        Objects.requireNonNull(pretokens, "pretokens");
        Objects.checkFromToIndex(from, to, pretokens.size());
        List<Integer> ids = new ArrayList<>();
        for (int index = from; index < to; index++) {
            String piece = pretokens.texts[index];
            if (piece == null) ids.add(pretokens.controlIds[index]);
            else encodePretoken(piece, ids);
        }
        return ids.stream().mapToInt(Integer::intValue).toArray();
    }

    /// `plain` with the BOS/EOS tokens enabled by tokenizer_config.json added.
    public int[] withModelSpecialTokens(int[] plain) {
        Objects.requireNonNull(plain, "plain");
        int prefix = this.addBosToken ? 1 : 0;
        int suffix = this.addEosToken ? 1 : 0;
        int[] result = new int[plain.length + prefix + suffix];
        int offset = 0;
        if (prefix != 0) result[offset++] = this.bosTokenId.orElseThrow();
        System.arraycopy(plain, 0, result, offset, plain.length);
        if (suffix != 0) result[result.length - 1] = this.eosTokenId;
        return result;
    }

    /// Decodes IDs to text and preserves all control-token spellings, including EOS.
    public String decode(int[] tokenIds) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        IncrementalDecoder decoder = newIncrementalDecoder();
        StringBuilder text = new StringBuilder();
        text.append(decoder.append(tokenIds));
        text.append(decoder.finish());
        return text.toString();
    }

    public OptionalInt specialTokenId(String token) {
        Integer id = this.controlTokenIds.get(Objects.requireNonNull(token, "token"));
        return id != null && this.specialTokenIds.contains(id) ? OptionalInt.of(id) : OptionalInt.empty();
    }

    /// Returns an added control-token ID, including non-special control markers.
    public OptionalInt controlTokenId(String token) {
        Integer id = this.controlTokenIds.get(Objects.requireNonNull(token, "token"));
        return id == null ? OptionalInt.empty() : OptionalInt.of(id);
    }

    public int eosTokenId() {
        return this.eosTokenId;
    }

    /// Returns the EOS ID declared by tokenizer_config.json, not generation stop IDs.
    public boolean isEosToken(int tokenId) {
        return tokenId == this.eosTokenId;
    }

    /// Returns the generation_config.json stop IDs without changing tokenizer EOS semantics.
    public Set<Integer> generationEosTokenIds() {
        return this.generationEosTokenIds;
    }

    public boolean isGenerationEosToken(int tokenId) {
        return this.generationEosTokenIds.contains(tokenId);
    }

    public OptionalInt bosTokenId() {
        return this.bosTokenId;
    }

    /// Returns the generation-config BOS ID; it is not implicitly inserted by tokenizer encoding.
    public OptionalInt generationBosTokenId() {
        return this.generationBosTokenId;
    }

    public TokenKind tokenKind(int tokenId) {
        requireTokenId(tokenId);
        if (tokenId == this.eosTokenId) return TokenKind.EOS;
        if (this.controlTokenIdsById.contains(tokenId)) return TokenKind.CONTROL;
        return TokenKind.NORMAL;
    }

    public IncrementalDecoder newIncrementalDecoder() {
        return new IncrementalDecoder(this);
    }

    void requireTokenId(int tokenId) {
        if (tokenId < 0 || tokenId >= this.tokensById.size() || this.tokensById.get(tokenId) == null) {
            throw new IllegalArgumentException("Unknown Qwen token ID: " + tokenId);
        }
    }

    boolean isControlTokenId(int tokenId) {
        return this.controlTokenIdsById.contains(tokenId);
    }

    /// Returns borrowed, immutable-by-convention BPE bytes; control and absent IDs have no text bytes.
    byte[] generationTokenBytes(int tokenId) {
        byte[][] cached = this.generatedTokenBytes;
        if (cached == null) {
            synchronized (this) {
                cached = this.generatedTokenBytes;
                if (cached == null) {
                    cached = new byte[this.tokensById.size()][];
                    for (int id = 0; id < cached.length; id++) {
                        String token = this.tokensById.get(id);
                        if (token == null || isControlTokenId(id)) continue;
                        byte[] bytes = new byte[token.codePointCount(0, token.length())];
                        for (int index = 0, offset = 0; index < token.length(); offset++) {
                            int codePoint = token.codePointAt(index);
                            int value = byteForCodePoint(codePoint);
                            if (value < 0) throw new IllegalStateException("Qwen BPE token has no byte mapping");
                            bytes[offset] = (byte) value;
                            index += Character.charCount(codePoint);
                        }
                        cached[id] = bytes;
                    }
                    this.generatedTokenBytes = cached;
                }
            }
        }
        return tokenId < 0 || tokenId >= cached.length ? null : cached[tokenId];
    }

    String tokenText(int tokenId) {
        return this.tokensById.get(tokenId);
    }

    int byteForCodePoint(int codePoint) {
        if (codePoint < 0 || codePoint >= this.byteByCodePoint.length) return -1;
        return this.byteByCodePoint[codePoint];
    }

    private void pretokenizeNormalized(String text, List<String> texts, List<Integer> controls) {
        if (text.isEmpty()) return;
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFC);
        Matcher matcher = this.pretokenPattern.matcher(normalized);
        int cursor = 0;
        while (matcher.find()) {
            if (matcher.start() > cursor) addPretoken(normalized.substring(cursor, matcher.start()), texts, controls);
            addPretoken(matcher.group(), texts, controls);
            cursor = matcher.end();
        }
        if (cursor < normalized.length()) addPretoken(normalized.substring(cursor), texts, controls);
    }

    private static void addPretoken(String pretoken, List<String> texts, List<Integer> controls) {
        if (pretoken.isEmpty()) return;
        texts.add(pretoken);
        controls.add(-1);
    }

    private void encodePretoken(String text, List<Integer> ids) {
        if (text.isEmpty()) return;
        String byteEncoded = encodeBytes(text, this.codePointByByte);
        Node first = null;
        Node previous = null;
        PriorityQueue<MergeCandidate> candidates = new PriorityQueue<>();
        for (int index = 0; index < byteEncoded.length(); ) {
            int codePoint = byteEncoded.codePointAt(index);
            Node node = new Node(new String(Character.toChars(codePoint)), index);
            node.previous = previous;
            if (previous == null) first = node;
            else previous.next = node;
            enqueueMerge(previous, node, candidates);
            previous = node;
            index += Character.charCount(codePoint);
        }
        while (!candidates.isEmpty()) {
            MergeCandidate candidate = candidates.remove();
            Node left = candidate.left();
            Node right = candidate.right();
            if (!left.active
                    || !right.active
                    || left.next != right
                    || !left.token.equals(candidate.merge().left())
                    || !right.token.equals(candidate.merge().right())) continue;
            left.token += right.token;
            left.next = right.next;
            if (right.next != null) right.next.previous = left;
            right.active = false;
            enqueueMerge(left.previous, left, candidates);
            enqueueMerge(left, left.next, candidates);
        }
        for (Node node = first; node != null; node = node.next) {
            Integer id = this.vocabulary.get(node.token);
            if (id == null) throw new IllegalStateException("Qwen BPE produced a token absent from its vocabulary");
            ids.add(id);
        }
    }

    private void enqueueMerge(Node left, Node right, PriorityQueue<MergeCandidate> candidates) {
        if (left == null || right == null) return;
        Merge merge = new Merge(left.token, right.token);
        Integer rank = this.mergeRanks.get(merge);
        if (rank != null) candidates.add(new MergeCandidate(merge, rank, left.startIndex, left, right));
    }

    private static Set<Integer> readTokenIds(JsonNode node) throws IOException {
        Set<Integer> ids = new HashSet<>();
        if (node.isIntegralNumber()) ids.add(node.asInt());
        else if (node.isArray()) {
            for (JsonNode value : node) {
                if (!value.isIntegralNumber()) throw new IOException("Invalid Qwen generation EOS ID");
                ids.add(value.asInt());
            }
        } else {
            throw new IOException("Qwen generation_config.json is missing EOS token IDs");
        }
        if (ids.isEmpty()) throw new IOException("Qwen generation_config.json has no EOS token IDs");
        return ids;
    }

    private static Integer nullableTokenId(JsonNode node) {
        return node.isIntegralNumber() ? node.asInt() : null;
    }

    private static String encodeBytes(String text, int[] codePointByByte) {
        StringBuilder encoded = new StringBuilder(text.length());
        byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        for (byte value : bytes) encoded.appendCodePoint(codePointByByte[value & 0xff]);
        return encoded.toString();
    }

    private static int[] codePointByByte(int[] byteByCodePoint) {
        int[] codePointByByte = new int[256];
        Arrays.fill(codePointByByte, -1);
        for (int codePoint = 0; codePoint < byteByCodePoint.length; codePoint++) {
            int value = byteByCodePoint[codePoint];
            if (value >= 0) codePointByByte[value] = codePoint;
        }
        for (int codePoint : codePointByByte) {
            if (codePoint < 0) throw new IllegalArgumentException("Incomplete Qwen byte-level vocabulary mapping");
        }
        return codePointByByte;
    }

    private static Pattern compileControlPattern(Set<String> controlTokens) {
        if (controlTokens.isEmpty()) return Pattern.compile("(?!)");
        String alternatives = controlTokens.stream()
                .sorted((left, right) -> Integer.compare(right.length(), left.length()))
                .map(Pattern::quote)
                .collect(java.util.stream.Collectors.joining("|"));
        return Pattern.compile(alternatives);
    }

    private static int[] byteByCodePoint() {
        boolean[] visible = new boolean[256];
        for (int value = 33; value <= 126; value++) visible[value] = true;
        for (int value = 161; value <= 172; value++) visible[value] = true;
        for (int value = 174; value <= 255; value++) visible[value] = true;
        int[] mapping = new int[768];
        Arrays.fill(mapping, -1);
        int extra = 0;
        for (int value = 0; value < visible.length; value++) {
            int codePoint = visible[value] ? value : BYTE_UNICODE_OFFSET + extra++;
            mapping[codePoint] = value;
        }
        return mapping;
    }

    private record Merge(String left, String right) {}

    private record MergeCandidate(Merge merge, int rank, int startIndex, Node left, Node right)
            implements Comparable<MergeCandidate> {
        @Override
        public int compareTo(MergeCandidate other) {
            int rankOrder = Integer.compare(this.rank, other.rank);
            return rankOrder != 0 ? rankOrder : Integer.compare(this.startIndex, other.startIndex);
        }
    }

    private static final class Node {
        private String token;
        private final int startIndex;
        private Node previous;
        private Node next;
        private boolean active = true;

        private Node(String token, int startIndex) {
            this.token = token;
            this.startIndex = startIndex;
        }
    }

    private static final class IteratorFields {
        private static void readVocabulary(JsonNode vocabulary, VocabularyConsumer consumer) throws IOException {
            if (!vocabulary.isObject()) throw new IOException("Qwen tokenizer vocabulary is missing");
            var fields = vocabulary.fields();
            while (fields.hasNext()) {
                var entry = fields.next();
                consumer.accept(entry.getKey(), entry.getValue().asInt(-1));
            }
        }
    }
}
