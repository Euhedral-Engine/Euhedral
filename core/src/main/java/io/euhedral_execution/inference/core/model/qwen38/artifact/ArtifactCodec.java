package io.euhedral_execution.inference.core.model.qwen38.artifact;

import io.euhedral_execution.inference.core.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.CompactTensorLayout;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/// Metadata and tensor-table codec of the compact version 2 container.
///
/// Metadata is encoded in the same order as `Qwen38Config`: int32 scalar fields, IEEE 754 binary64
/// floating-point fields, a length-prefixed UTF-8 activation name, a counted sequence of layer-type
/// ordinals, the remaining int32 fields, two boolean bytes, and the MTP layer count.
/// `DataOutputStream` and `ByteBuffer` both use the explicit big-endian order selected here.
final class ArtifactCodec {

    static final int MAX_COUNT = 1_000_000;
    static final int MAX_RANK = 64;
    static final int MAX_STRING_BYTES = 1 << 20;
    static final long MAX_METADATA_BYTES = 8L << 20;
    static final ByteOrder BYTE_ORDER = ByteOrder.BIG_ENDIAN;

    private ArtifactCodec() {}

    static byte[] encodeMetadata(Qwen38Config config) throws ArtifactFormatException {
        validateConfig(config);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream output = new DataOutputStream(bytes)) {
            output.writeInt(config.vocabSize());
            output.writeInt(config.hiddenSize());
            output.writeInt(config.numHiddenLayers());
            output.writeInt(config.numAttentionHeads());
            output.writeInt(config.numKeyValueHeads());
            output.writeInt(config.attentionHeadDim());
            output.writeInt(config.intermediateSize());
            output.writeInt(config.linearNumKeyHeads());
            output.writeInt(config.linearNumValueHeads());
            output.writeInt(config.linearKeyHeadDim());
            output.writeInt(config.linearValueHeadDim());
            output.writeInt(config.linearConvKernelDim());
            output.writeDouble(config.rmsNormEpsilon());
            output.writeDouble(config.ropeTheta());
            output.writeDouble(config.partialRotaryFactor());
            output.writeInt(config.maxPositionEmbeddings());
            writeString(output, config.hiddenActivation(), "hidden activation");
            output.writeInt(config.layerTypes().length);
            for (LayerType layerType : config.layerTypes()) {
                output.writeInt(layerType.ordinal());
            }
            output.writeInt(config.numExperts());
            output.writeInt(config.numExpertsPerToken());
            output.writeInt(config.moeIntermediateSize());
            output.writeInt(config.sharedExpertIntermediateSize());
            output.writeByte(config.tieWordEmbeddings() ? 1 : 0);
            output.writeByte(config.attentionOutputGate() ? 1 : 0);
            output.writeInt(config.mtpLayerCount());
        } catch (IOException exception) {
            throw new AssertionError("Byte array metadata encoding failed", exception);
        }
        if (bytes.size() > MAX_METADATA_BYTES) {
            throw invalid("metadata is too large");
        }
        return bytes.toByteArray();
    }

    static Qwen38Config decodeMetadata(byte[] metadata) throws ArtifactFormatException {
        ByteBuffer input = ByteBuffer.wrap(metadata).order(BYTE_ORDER);
        int vocabSize = readInt(input, "vocab size");
        int hiddenSize = readInt(input, "hidden size");
        int numHiddenLayers = readInt(input, "hidden layer count");
        int numAttentionHeads = readInt(input, "attention head count");
        int numKeyValueHeads = readInt(input, "key/value head count");
        int attentionHeadDim = readInt(input, "attention head dimension");
        int intermediateSize = readInt(input, "intermediate size");
        int linearNumKeyHeads = readInt(input, "linear key head count");
        int linearNumValueHeads = readInt(input, "linear value head count");
        int linearKeyHeadDim = readInt(input, "linear key head dimension");
        int linearValueHeadDim = readInt(input, "linear value head dimension");
        int linearConvKernelDim = readInt(input, "linear convolution kernel dimension");
        double rmsNormEpsilon = readDouble(input, "RMS norm epsilon");
        double ropeTheta = readDouble(input, "rope theta");
        double partialRotaryFactor = readDouble(input, "partial rotary factor");
        int maxPositionEmbeddings = readInt(input, "maximum position embeddings");
        String hiddenActivation = readString(input, "hidden activation");
        int layerCount = readCount(input, "layer type count");
        LayerType[] layerTypes = new LayerType[layerCount];
        for (int i = 0; i < layerCount; i++) {
            layerTypes[i] = readEnum(input, LayerType.values(), "layer type");
        }
        int numExperts = readInt(input, "expert count");
        int numExpertsPerToken = readInt(input, "experts per token");
        int moeIntermediateSize = readInt(input, "MoE intermediate size");
        int sharedExpertIntermediateSize = readInt(input, "shared expert intermediate size");
        boolean tieWordEmbeddings = readBoolean(input, "tie word embeddings");
        boolean attentionOutputGate = readBoolean(input, "attention output gate");
        int mtpLayerCount = readInt(input, "MTP layer count");

        if (input.hasRemaining()) {
            throw invalid("metadata contains trailing bytes");
        }

        Qwen38Config config = new Qwen38Config(
                vocabSize,
                hiddenSize,
                numHiddenLayers,
                numAttentionHeads,
                numKeyValueHeads,
                attentionHeadDim,
                intermediateSize,
                linearNumKeyHeads,
                linearNumValueHeads,
                linearKeyHeadDim,
                linearValueHeadDim,
                linearConvKernelDim,
                rmsNormEpsilon,
                ropeTheta,
                partialRotaryFactor,
                maxPositionEmbeddings,
                hiddenActivation,
                layerTypes,
                numExperts,
                numExpertsPerToken,
                moeIntermediateSize,
                sharedExpertIntermediateSize,
                tieWordEmbeddings,
                attentionOutputGate,
                mtpLayerCount);
        validateConfig(config);
        return config;
    }

    static byte[] encodeTensorTable(TensorDescriptor[] tensors) throws ArtifactFormatException {
        validateTensorCount(tensors);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream output = new DataOutputStream(bytes)) {
            for (TensorDescriptor tensor : tensors) {
                validateTensorLayout(tensor);
                byte[] name = encodeString(tensor.name(), "tensor name");
                output.writeInt(name.length);
                output.write(name);
                output.writeInt(tensor.shape().length);
                for (long dimension : tensor.shape()) {
                    output.writeLong(dimension);
                }
                output.writeInt(tensor.dataType().ordinal());
                output.writeInt(tensor.format().ordinal());
                output.writeInt(tensor.layout().ordinal());
                output.writeLong(tensor.dataOffset());
                output.writeLong(tensor.byteSize());
            }
        } catch (ArtifactFormatException exception) {
            throw exception;
        } catch (IOException exception) {
            throw new AssertionError("Byte array tensor table encoding failed", exception);
        }
        validateNonOverlapping(tensors);
        return bytes.toByteArray();
    }

    static int readInt(ByteBuffer input, String field) throws ArtifactFormatException {
        requireRemaining(input, Integer.BYTES, field);
        return input.getInt();
    }

    static long readLong(ByteBuffer input, String field) throws ArtifactFormatException {
        requireRemaining(input, Long.BYTES, field);
        return input.getLong();
    }

    static double readDouble(ByteBuffer input, String field) throws ArtifactFormatException {
        requireRemaining(input, Double.BYTES, field);
        return input.getDouble();
    }

    static String readString(ByteBuffer input, String field) throws ArtifactFormatException {
        int length = readInt(input, field + " length");
        if (length < 0 || length > MAX_STRING_BYTES) {
            throw invalid(field + " length is outside the supported range: " + length);
        }
        requireRemaining(input, length, field);
        byte[] bytes = new byte[length];
        input.get(bytes);
        return decodeUtf8(bytes, field);
    }

    static <E extends Enum<E>> E readEnum(ByteBuffer input, E[] values, String field) throws ArtifactFormatException {
        return enumValue(readInt(input, field), values, field);
    }

    static <E extends Enum<E>> E enumValue(int ordinal, E[] values, String field) throws ArtifactFormatException {
        if (ordinal < 0 || ordinal >= values.length) {
            throw invalid(field + " value is outside the supported range: " + ordinal);
        }
        return values[ordinal];
    }

    static String decodeUtf8(byte[] bytes, String field) throws ArtifactFormatException {
        try {
            return StandardCharsets.UTF_8
                    .newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes))
                    .toString();
        } catch (CharacterCodingException exception) {
            throw invalid(field + " is not valid UTF-8");
        }
    }

    static boolean readBoolean(ByteBuffer input, String field) throws ArtifactFormatException {
        requireRemaining(input, 1, field);
        int value = Byte.toUnsignedInt(input.get());
        if (value > 1) {
            throw invalid(field + " must be 0 or 1, but was " + value);
        }
        return value == 1;
    }

    static void validateNonOverlapping(TensorDescriptor[] tensors) throws ArtifactFormatException {
        for (int i = 0; i < tensors.length; i++) {
            TensorDescriptor first = tensors[i];
            if (first.byteSize() == 0) {
                continue;
            }
            long firstEnd = ArtifactFileAccess.checkedEnd(first.dataOffset(), first.byteSize(), "tensor data");
            for (int j = i + 1; j < tensors.length; j++) {
                TensorDescriptor second = tensors[j];
                if (second.byteSize() == 0) {
                    continue;
                }
                long secondEnd = ArtifactFileAccess.checkedEnd(second.dataOffset(), second.byteSize(), "tensor data");
                if (first.dataOffset() < secondEnd && second.dataOffset() < firstEnd) {
                    throw invalid("tensor data ranges overlap: " + first.name() + " and " + second.name());
                }
            }
        }
    }

    static ArtifactFormatException invalid(String message) {
        return new ArtifactFormatException(message);
    }

    private static int readCount(ByteBuffer input, String field) throws ArtifactFormatException {
        int count = readInt(input, field);
        if (count < 0 || count > MAX_COUNT) {
            throw invalid(field + " is outside the supported range: " + count);
        }
        return count;
    }

    private static void requireRemaining(ByteBuffer input, int bytes, String field) throws ArtifactFormatException {
        if (bytes < 0 || input.remaining() < bytes) {
            throw invalid("metadata is truncated while reading " + field);
        }
    }

    private static void validateConfig(Qwen38Config config) throws ArtifactFormatException {
        if (config == null) {
            throw invalid("config is null");
        }
        int[] counts = {
            config.vocabSize(),
            config.hiddenSize(),
            config.numHiddenLayers(),
            config.numAttentionHeads(),
            config.numKeyValueHeads(),
            config.attentionHeadDim(),
            config.intermediateSize(),
            config.linearNumKeyHeads(),
            config.linearNumValueHeads(),
            config.linearKeyHeadDim(),
            config.linearValueHeadDim(),
            config.linearConvKernelDim(),
            config.maxPositionEmbeddings(),
            config.numExperts(),
            config.numExpertsPerToken(),
            config.moeIntermediateSize(),
            config.sharedExpertIntermediateSize(),
            config.mtpLayerCount()
        };
        for (int count : counts) {
            if (count < 0 || count > MAX_COUNT) {
                throw invalid("config count or dimension is outside the supported range: " + count);
            }
        }
        if (!Double.isFinite(config.rmsNormEpsilon())
                || !Double.isFinite(config.ropeTheta())
                || !Double.isFinite(config.partialRotaryFactor())) {
            throw invalid("config floating-point values must be finite");
        }
        if (config.layerTypes() == null) {
            throw invalid("layer types are null");
        }
        if (config.layerTypes().length != config.numHiddenLayers()) {
            throw invalid("layer type count does not match hidden layer count");
        }
        if (config.layerTypes().length > MAX_COUNT) {
            throw invalid("layer type count is too large");
        }
        for (LayerType layerType : config.layerTypes()) {
            if (layerType == null) {
                throw invalid("layer types contain null");
            }
        }
        encodeString(config.hiddenActivation(), "hidden activation");
    }

    private static void validateTensorCount(TensorDescriptor[] tensors) throws ArtifactFormatException {
        if (tensors == null) {
            throw invalid("tensor descriptors are null");
        }
        if (tensors.length > MAX_COUNT) {
            throw invalid("tensor count is too large: " + tensors.length);
        }
    }

    private static void validateTensor(TensorDescriptor tensor) throws ArtifactFormatException {
        if (tensor == null) {
            throw invalid("tensor descriptor is null");
        }
        byte[] name = encodeString(tensor.name(), "tensor name");
        if (name.length == 0) {
            throw invalid("tensor name is empty");
        }
        if (tensor.shape() == null || tensor.shape().length > MAX_RANK) {
            throw invalid("tensor rank is outside the supported range");
        }
        for (long dimension : tensor.shape()) {
            if (dimension < 0) {
                throw invalid("tensor shape contains a negative dimension");
            }
        }
        if (tensor.dataType() == null || tensor.format() == null) {
            throw invalid("tensor data type and format are required");
        }
        if (tensor.dataOffset() < 0 || tensor.byteSize() < 0) {
            throw invalid("tensor data offset and byte size must be non-negative");
        }
        ArtifactFileAccess.checkedEnd(tensor.dataOffset(), tensor.byteSize(), "tensor data");
    }

    private static void validateTensorLayout(TensorDescriptor tensor) throws ArtifactFormatException {
        validateTensor(tensor);
        if (tensor.layout() == null) {
            throw invalid("compact tensor layout is required");
        }
        try {
            if (!CompactTensorLayout.acceptsByteSize(
                    tensor.shape(), tensor.dataType(), tensor.format(), tensor.layout(), tensor.byteSize())) {
                throw invalid("compact tensor byte size does not match its shape, format, and layout: " + tensor.name()
                        + " was " + tensor.byteSize());
            }
        } catch (IllegalArgumentException exception) {
            throw invalid("unsupported compact tensor '" + tensor.name() + "': " + exception.getMessage());
        }
    }

    private static byte[] encodeString(String value, String field) throws ArtifactFormatException {
        if (value == null) {
            throw invalid(field + " is null");
        }
        try {
            ByteBuffer encoded = StandardCharsets.UTF_8
                    .newEncoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .encode(CharBuffer.wrap(value));
            byte[] bytes = new byte[encoded.remaining()];
            encoded.get(bytes);
            if (bytes.length > MAX_STRING_BYTES) {
                throw invalid(field + " is too long");
            }
            return bytes;
        } catch (CharacterCodingException exception) {
            throw invalid(field + " is not valid UTF-8");
        }
    }

    private static void writeString(DataOutputStream output, String value, String field) throws IOException {
        byte[] bytes = encodeString(value, field);
        output.writeInt(bytes.length);
        output.write(bytes);
    }
}
