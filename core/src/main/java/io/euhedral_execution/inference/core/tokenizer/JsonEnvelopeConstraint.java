package io.euhedral_execution.inference.core.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/// Request-local byte grammar for a JSON answer or tool-call envelope. Token candidates are checked
/// against borrowed Qwen BPE bytes, not independently decoded strings: a UTF-8 character may span tokens.
/// This enforces envelope syntax and offered names, not the full JSON Schema of tool arguments. It serves as
/// a token constraint of its own, or as the byte grammar of an answer that follows reasoning.
public final class JsonEnvelopeConstraint implements TokenConstraint, ByteGrammar {
    private static final byte[] CALL_PREFIX = ascii("{\"tool_calls\":[{\"name\":\"");
    private static final byte[] CONTENT_PREFIX = ascii("{\"content\":\"");
    private static final byte[] AFTER_NAME = ascii(",\"arguments\":{");
    private static final byte[] NEXT_CALL = ascii("{\"name\":\"");

    private final QwenTokenizer tokenizer;
    private final List<String> toolNames;
    private final boolean requiresCall;
    private final boolean parallel;
    private final State current = new State();
    private final State scratch = new State();

    public JsonEnvelopeConstraint(QwenTokenizer tokenizer, List<String> toolNames, boolean requiresCall) {
        this(tokenizer, toolNames, requiresCall, true);
    }

    public JsonEnvelopeConstraint(
            QwenTokenizer tokenizer, List<String> toolNames, boolean requiresCall, boolean parallel) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.toolNames = List.copyOf(toolNames);
        if (this.toolNames.isEmpty()) throw new IllegalArgumentException("tool names must not be empty");
        this.requiresCall = requiresCall;
        this.parallel = parallel;
    }

    /// Does not change the committed grammar state; sampling may query every vocabulary token.
    @Override
    public boolean allows(int tokenId) {
        if (this.tokenizer.isGenerationEosToken(tokenId)) return complete();
        byte[] bytes = this.tokenizer.generationTokenBytes(tokenId);
        if (bytes == null || bytes.length == 0) return false;
        return allows(bytes, 0);
    }

    /// Commits only the token that sampling selected, on the generation thread.
    @Override
    public void accept(int tokenId) {
        if (!allows(tokenId)) throw new IllegalArgumentException("token violates the JSON envelope constraint");
        if (this.tokenizer.isGenerationEosToken(tokenId)) return;
        accept(this.tokenizer.generationTokenBytes(tokenId), 0);
    }

    @Override
    public boolean allows(byte[] bytes, int from) {
        this.scratch.copyFrom(this.current);
        for (int index = from; index < bytes.length; index++)
            if (!this.scratch.feed(bytes[index] & 0xff, this.toolNames, this.requiresCall, this.parallel)) return false;
        return true;
    }

    @Override
    public void accept(byte[] bytes, int from) {
        for (int index = from; index < bytes.length; index++) {
            if (!this.current.feed(bytes[index] & 0xff, this.toolNames, this.requiresCall, this.parallel))
                throw new IllegalStateException("committed token failed JSON validation");
        }
    }

    @Override
    public boolean complete() {
        return this.current.phase == Phase.DONE;
    }

    private static byte[] ascii(String value) {
        return value.getBytes(StandardCharsets.US_ASCII);
    }

    private enum Phase {
        PREFIX,
        NAME,
        AFTER_NAME,
        ARGUMENTS,
        CALL_END,
        AFTER_CALL,
        NEXT_CALL,
        ROOT_END,
        CONTENT_STRING,
        CONTENT_END,
        DONE
    }

    private enum Lex {
        NORMAL,
        STRING_KEY,
        STRING_VALUE,
        ESCAPE_KEY,
        ESCAPE_VALUE,
        UNICODE_KEY,
        UNICODE_VALUE,
        LITERAL,
        NUMBER
    }

    private static final int OBJECT_KEY_OR_END = 0;
    private static final int OBJECT_KEY = 1;
    private static final int OBJECT_COLON = 2;
    private static final int OBJECT_VALUE = 3;
    private static final int OBJECT_COMMA_OR_END = 4;
    private static final int ARRAY_VALUE_OR_END = 5;
    private static final int ARRAY_VALUE = 6;
    private static final int ARRAY_COMMA_OR_END = 7;

    private static final int NUM_MINUS = 0;
    private static final int NUM_ZERO = 1;
    private static final int NUM_INTEGER = 2;
    private static final int NUM_DOT = 3;
    private static final int NUM_FRACTION = 4;
    private static final int NUM_EXP = 5;
    private static final int NUM_EXP_SIGN = 6;
    private static final int NUM_EXP_DIGITS = 7;

    private static final class State {
        private Phase phase = Phase.PREFIX;
        private Lex lex = Lex.NORMAL;
        private int[] stack = new int[64];
        private final char[] name = new char[64];
        private int depth;
        private int nameLength;
        private int prefixLength;
        private int prefixOptions = 3;
        private int literalLength;
        private int unicodeDigits;
        private int utf8Remaining;
        private int utf8Minimum;
        private int utf8Maximum;
        private int numberState;
        private String literal;
        private int literalIndex;

        private void copyFrom(State other) {
            this.phase = other.phase;
            this.lex = other.lex;
            this.depth = other.depth;
            if (this.stack.length < other.depth) this.stack = Arrays.copyOf(this.stack, other.stack.length);
            System.arraycopy(other.stack, 0, this.stack, 0, other.depth);
            this.nameLength = other.nameLength;
            System.arraycopy(other.name, 0, this.name, 0, other.nameLength);
            this.prefixLength = other.prefixLength;
            this.prefixOptions = other.prefixOptions;
            this.literalLength = other.literalLength;
            this.unicodeDigits = other.unicodeDigits;
            this.utf8Remaining = other.utf8Remaining;
            this.utf8Minimum = other.utf8Minimum;
            this.utf8Maximum = other.utf8Maximum;
            this.numberState = other.numberState;
            this.literal = other.literal;
            this.literalIndex = other.literalIndex;
        }

        private boolean feed(int value, List<String> names, boolean requiresCall, boolean parallel) {
            return switch (this.phase) {
                case PREFIX -> prefix(value, requiresCall);
                case NAME -> name(value, names);
                case AFTER_NAME -> literal(value, AFTER_NAME, Phase.ARGUMENTS);
                case ARGUMENTS -> json(value);
                case CALL_END -> {
                    if (value != '}') yield false;
                    this.phase = Phase.AFTER_CALL;
                    yield true;
                }
                case AFTER_CALL -> {
                    if (value == ']') this.phase = Phase.ROOT_END;
                    else if (value == ',' && parallel) this.phase = Phase.NEXT_CALL;
                    else yield false;
                    yield true;
                }
                case NEXT_CALL -> literal(value, NEXT_CALL, Phase.NAME);
                case ROOT_END -> {
                    if (value != '}') yield false;
                    this.phase = Phase.DONE;
                    yield true;
                }
                case CONTENT_STRING ->
                    switch (this.lex) {
                        case ESCAPE_VALUE -> escape(value, false);
                        case UNICODE_VALUE -> unicode(value, false);
                        default -> string(value, false);
                    };
                case CONTENT_END -> {
                    if (value != '}') yield false;
                    this.phase = Phase.DONE;
                    yield true;
                }
                case DONE -> false;
            };
        }

        private boolean prefix(int value, boolean requiresCall) {
            int possible = this.prefixOptions & (requiresCall ? 1 : 3);
            if ((possible & 1) != 0
                    && (this.prefixLength >= CALL_PREFIX.length || value != (CALL_PREFIX[this.prefixLength] & 0xff)))
                possible &= ~1;
            if ((possible & 2) != 0
                    && (this.prefixLength >= CONTENT_PREFIX.length
                            || value != (CONTENT_PREFIX[this.prefixLength] & 0xff))) possible &= ~2;
            if (possible == 0) return false;
            this.prefixOptions = possible;
            this.prefixLength++;
            if (possible == 1 && this.prefixLength == CALL_PREFIX.length) this.phase = Phase.NAME;
            else if (possible == 2 && this.prefixLength == CONTENT_PREFIX.length) this.phase = Phase.CONTENT_STRING;
            return true;
        }

        private boolean name(int value, List<String> names) {
            if (value == '"') {
                for (String offered : names) {
                    if (offered.length() == this.nameLength && matchesName(offered)) {
                        this.phase = Phase.AFTER_NAME;
                        this.literalLength = 0;
                        this.nameLength = 0;
                        return true;
                    }
                }
                return false;
            }
            if (value > 0x7f || this.nameLength == this.name.length) return false;
            this.name[this.nameLength++] = (char) value;
            for (String offered : names) if (offered.length() >= this.nameLength && matchesName(offered)) return true;
            return false;
        }

        private boolean matchesName(String offered) {
            for (int index = 0; index < this.nameLength; index++)
                if (offered.charAt(index) != this.name[index]) return false;
            return true;
        }

        private boolean literal(int value, byte[] expected, Phase next) {
            if (value != (expected[this.literalLength] & 0xff)) return false;
            if (++this.literalLength == expected.length) {
                this.phase = next;
                this.literalLength = 0;
                if (next == Phase.ARGUMENTS) {
                    this.depth = 1;
                    this.stack[0] = OBJECT_KEY_OR_END;
                    this.lex = Lex.NORMAL;
                }
            }
            return true;
        }

        private boolean json(int value) {
            switch (this.lex) {
                case STRING_KEY, STRING_VALUE -> {
                    return string(value, this.lex == Lex.STRING_KEY);
                }
                case ESCAPE_KEY, ESCAPE_VALUE -> {
                    return escape(value, this.lex == Lex.ESCAPE_KEY);
                }
                case UNICODE_KEY, UNICODE_VALUE -> {
                    return unicode(value, this.lex == Lex.UNICODE_KEY);
                }
                case LITERAL -> {
                    if (value != this.literal.charAt(this.literalIndex++)) return false;
                    if (this.literalIndex == this.literal.length()) {
                        this.lex = Lex.NORMAL;
                        valueCompleted();
                    }
                    return true;
                }
                case NUMBER -> {
                    return number(value);
                }
                case NORMAL -> {
                    /* Follow the container state below. */
                }
            }
            if (value == ' ' || value == '\t' || value == '\n' || value == '\r') return true;
            int state = this.stack[this.depth - 1];
            return switch (state) {
                case OBJECT_KEY_OR_END, OBJECT_KEY -> {
                    if (value == '"') {
                        this.lex = Lex.STRING_KEY;
                        yield true;
                    }
                    yield state == OBJECT_KEY_OR_END && value == '}' && closeContainer();
                }
                case OBJECT_COLON -> {
                    if (value != ':') yield false;
                    this.stack[this.depth - 1] = OBJECT_VALUE;
                    yield true;
                }
                case OBJECT_VALUE, ARRAY_VALUE, ARRAY_VALUE_OR_END -> {
                    if (state == ARRAY_VALUE_OR_END && value == ']') yield closeContainer();
                    yield startValue(value);
                }
                case OBJECT_COMMA_OR_END -> {
                    if (value == '}') yield closeContainer();
                    if (value != ',') yield false;
                    this.stack[this.depth - 1] = OBJECT_KEY;
                    yield true;
                }
                case ARRAY_COMMA_OR_END -> {
                    if (value == ']') yield closeContainer();
                    if (value != ',') yield false;
                    this.stack[this.depth - 1] = ARRAY_VALUE;
                    yield true;
                }
                default -> false;
            };
        }

        private boolean startValue(int value) {
            if (value == '{' || value == '[') {
                if (this.depth == this.stack.length) this.stack = Arrays.copyOf(this.stack, this.stack.length * 2);
                this.stack[this.depth++] = value == '{' ? OBJECT_KEY_OR_END : ARRAY_VALUE_OR_END;
                return true;
            }
            if (value == '"') {
                this.lex = Lex.STRING_VALUE;
                return true;
            }
            if (value == 't' || value == 'f' || value == 'n') {
                this.literal = value == 't' ? "true" : value == 'f' ? "false" : "null";
                this.literalIndex = 1;
                this.lex = Lex.LITERAL;
                return true;
            }
            if (value == '-') {
                this.numberState = NUM_MINUS;
                this.lex = Lex.NUMBER;
                return true;
            }
            if (value == '0' || value >= '1' && value <= '9') {
                this.numberState = value == '0' ? NUM_ZERO : NUM_INTEGER;
                this.lex = Lex.NUMBER;
                return true;
            }
            return false;
        }

        private boolean number(int value) {
            if (value >= '0' && value <= '9') {
                this.numberState = switch (this.numberState) {
                    case NUM_MINUS -> value == '0' ? NUM_ZERO : NUM_INTEGER;
                    case NUM_INTEGER -> NUM_INTEGER;
                    case NUM_DOT, NUM_FRACTION -> NUM_FRACTION;
                    case NUM_EXP, NUM_EXP_SIGN, NUM_EXP_DIGITS -> NUM_EXP_DIGITS;
                    default -> -1;
                };
                return this.numberState >= 0;
            }
            if (value == '.' && (this.numberState == NUM_ZERO || this.numberState == NUM_INTEGER)) {
                this.numberState = NUM_DOT;
                return true;
            }
            if ((value == 'e' || value == 'E')
                    && (this.numberState == NUM_ZERO
                            || this.numberState == NUM_INTEGER
                            || this.numberState == NUM_FRACTION)) {
                this.numberState = NUM_EXP;
                return true;
            }
            if ((value == '+' || value == '-') && this.numberState == NUM_EXP) {
                this.numberState = NUM_EXP_SIGN;
                return true;
            }
            if (this.numberState != NUM_ZERO
                    && this.numberState != NUM_INTEGER
                    && this.numberState != NUM_FRACTION
                    && this.numberState != NUM_EXP_DIGITS) return false;
            this.lex = Lex.NORMAL;
            valueCompleted();
            return json(value);
        }

        private boolean string(int value, boolean key) {
            if (this.utf8Remaining > 0) {
                if (value < this.utf8Minimum || value > this.utf8Maximum) return false;
                this.utf8Remaining--;
                this.utf8Minimum = 0x80;
                this.utf8Maximum = 0xbf;
                return true;
            }
            if (value == '"') {
                if (this.phase == Phase.CONTENT_STRING) this.phase = Phase.CONTENT_END;
                else if (key) this.stack[this.depth - 1] = OBJECT_COLON;
                else valueCompleted();
                this.lex = Lex.NORMAL;
                return true;
            }
            if (value == '\\') {
                this.lex = key ? Lex.ESCAPE_KEY : Lex.ESCAPE_VALUE;
                return true;
            }
            if (value < 0x20) return false;
            if (value < 0x80) return true;
            this.utf8Minimum = 0x80;
            this.utf8Maximum = 0xbf;
            if (value >= 0xc2 && value <= 0xdf) {
                this.utf8Remaining = 1;
            } else if (value >= 0xe0 && value <= 0xef) {
                this.utf8Remaining = 2;
                if (value == 0xe0) this.utf8Minimum = 0xa0;
                if (value == 0xed) this.utf8Maximum = 0x9f;
            } else if (value >= 0xf0 && value <= 0xf4) {
                this.utf8Remaining = 3;
                if (value == 0xf0) this.utf8Minimum = 0x90;
                if (value == 0xf4) this.utf8Maximum = 0x8f;
            } else return false;
            return true;
        }

        private boolean escape(int value, boolean key) {
            if (value == 'u') {
                this.lex = key ? Lex.UNICODE_KEY : Lex.UNICODE_VALUE;
                this.unicodeDigits = 0;
                return true;
            }
            if (value != '"'
                    && value != '\\'
                    && value != '/'
                    && value != 'b'
                    && value != 'f'
                    && value != 'n'
                    && value != 'r'
                    && value != 't') return false;
            this.lex = key ? Lex.STRING_KEY : Lex.STRING_VALUE;
            return true;
        }

        private boolean unicode(int value, boolean key) {
            if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F')))
                return false;
            if (++this.unicodeDigits == 4) this.lex = key ? Lex.STRING_KEY : Lex.STRING_VALUE;
            return true;
        }

        private boolean closeContainer() {
            this.depth--;
            if (this.depth == 0) this.phase = Phase.CALL_END;
            else valueCompleted();
            return true;
        }

        private void valueCompleted() {
            int state = this.stack[this.depth - 1];
            if (state == OBJECT_VALUE) this.stack[this.depth - 1] = OBJECT_COMMA_OR_END;
            else if (state == ARRAY_VALUE || state == ARRAY_VALUE_OR_END)
                this.stack[this.depth - 1] = ARRAY_COMMA_OR_END;
            else throw new IllegalStateException("JSON value has no awaiting container");
        }
    }
}
