package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/// The calibrated artifacts a mixed artifact is assembled from, one per [Option], and the base whose objects an
/// allocation does not name.
///
/// The choosable objects are the quantized linear objects of `text/layers/*` ([#projections]); the quantized linears
/// of the MTP layer are listed apart ([#drafterProjections]). Every other object comes from the base. The objects
/// derived directly from the checkpoint (norms, GDN state tensors and convolution, the small BF16 GDN projections,
/// the draft head's token ids) must be bit for bit the same in every source, which [#open] verifies by the hash of
/// their payload. The quantized objects outside the layers (the token embedding, the output and draft heads, the MTP
/// layer) differ between sources by construction, each source having been calibrated on its own, and are not
/// compared.
public final class Sources {

    /// The suffixes of an object's companions, which travel with it: trellis sign vectors and segment ends.
    static final List<String> COMPANION_SUFFIXES = List.of(".suh", ".svh", ".segments");

    private static final String LAYERS = "text/layers/";
    private static final String DRAFTER_LAYER = "mtp/layer/";
    private static final int HASH_BUFFER_BYTES = 1 << 20;

    private final Map<Option, Path> paths;
    private final Map<Option, Artifact> artifacts;
    private final Map<Option, Map<String, TensorDescriptor>> descriptors = new EnumMap<>(Option.class);
    private final Option base;
    private final List<String> projections, drafterProjections;

    private Sources(Map<Option, Path> paths, Map<Option, Artifact> artifacts, Option base) {
        this.paths = new EnumMap<>(paths);
        this.artifacts = new EnumMap<>(artifacts);
        this.base = base;
        artifacts.forEach((option, artifact) -> {
            Map<String, TensorDescriptor> byName = new HashMap<>();
            for (TensorDescriptor tensor : artifact.tensors()) byName.put(tensor.name(), tensor);
            descriptors.put(option, byName);
        });
        this.projections = quantizedLinears(artifacts.get(base), LAYERS);
        this.drafterProjections = quantizedLinears(artifacts.get(base), DRAFTER_LAYER);
    }

    /// Opens `artifacts` (an artifact file for each option that can be chosen) and checks them against each other.
    ///
    /// @throws IllegalArgumentException naming the object when a source lacks one the base has among the
    /// projections or
    ///     the objects compared, when their shapes or formats differ where they must agree, or when a compared
    ///     object's bytes differ
    public static Sources open(Map<Option, Path> artifacts, Option base) throws IOException {
        if (!artifacts.containsKey(base)) throw new IllegalArgumentException("no source for the base " + base.key());
        Map<Option, Artifact> read = new EnumMap<>(Option.class);
        for (var entry : artifacts.entrySet()) read.put(entry.getKey(), ArtifactReader.read(entry.getValue()));
        Sources sources = new Sources(artifacts, read, base);
        sources.verify();
        return sources;
    }

    private void verify() throws IOException {
        Set<String> compared = compared(base);
        for (Option option : paths.keySet()) {
            if (option == base) continue;
            Set<String> projectionsOf = new TreeSet<>(quantizedLinears(artifacts.get(option), LAYERS));
            if (!projectionsOf.equals(new TreeSet<>(projections)))
                throw new IllegalArgumentException(option.key() + " and " + base.key()
                        + " do not have the same projections, first difference: "
                        + firstDifference(projectionsOf, new TreeSet<>(projections)));
            Set<String> comparedOf = compared(option);
            if (!comparedOf.equals(compared))
                throw new IllegalArgumentException(option.key() + " and " + base.key()
                        + " do not have the same shared objects, first difference: "
                        + firstDifference(comparedOf, compared));
            for (String name : projections) {
                TensorDescriptor mine = descriptor(option, name), theirs = descriptor(base, name);
                if (!Arrays.equals(mine.shape(), theirs.shape()))
                    throw new IllegalArgumentException(name + " has the shape " + Arrays.toString(mine.shape()) + " in "
                            + option.key() + " and " + Arrays.toString(theirs.shape()) + " in " + base.key());
            }
            for (String name : compared) {
                TensorDescriptor mine = descriptor(option, name), theirs = descriptor(base, name);
                if (!Arrays.equals(mine.shape(), theirs.shape())
                        || mine.dataType() != theirs.dataType()
                        || mine.format() != theirs.format()
                        || mine.layout() != theirs.layout()
                        || mine.byteSize() != theirs.byteSize()
                        || !Arrays.equals(hash(option, mine), hash(base, theirs)))
                    throw new IllegalArgumentException(
                            name + " differs between " + option.key() + " and " + base.key());
            }
        }
    }

    /// The option the objects an allocation does not name come from.
    public Option base() {
        return base;
    }

    /// The options that have a source.
    public Set<Option> options() {
        return Set.copyOf(paths.keySet());
    }

    /// The file of `option`'s source.
    public Path path(Option option) {
        Path path = paths.get(option);
        if (path == null) throw new IllegalArgumentException("no source for " + option.key());
        return path;
    }

    /// The artifact of `option`'s source.
    public Artifact artifact(Option option) {
        path(option);
        return artifacts.get(option);
    }

    /// The choosable objects, in the base's order: its quantized linear objects of `text/layers/*`.
    public List<String> projections() {
        return projections;
    }

    /// The quantized linear objects of the MTP layer, in the base's order. An allocation does not choose them.
    public List<String> drafterProjections() {
        return drafterProjections;
    }

    /// The descriptor of `name` in `option`'s source.
    public TensorDescriptor descriptor(Option option, String name) {
        path(option);
        TensorDescriptor descriptor = descriptors.get(option).get(name);
        if (descriptor == null) throw new IllegalArgumentException(option.key() + " has no object " + name);
        return descriptor;
    }

    /// The descriptors of `name` and of its companions in `option`'s source: the companions travel with their object.
    public List<TensorDescriptor> objectWithCompanions(Option option, String name) {
        List<TensorDescriptor> all = new ArrayList<>();
        all.add(descriptor(option, name));
        for (String suffix : COMPANION_SUFFIXES) {
            TensorDescriptor companion = descriptors.get(option).get(name + suffix);
            if (companion != null) all.add(companion);
        }
        return all;
    }

    private boolean companion(Map<String, TensorDescriptor> byName, String name) {
        for (String suffix : COMPANION_SUFFIXES) {
            if (name.endsWith(suffix) && byName.containsKey(name.substring(0, name.length() - suffix.length())))
                return true;
        }
        return false;
    }

    private static boolean quantized(WeightFormat format) {
        return switch (format) {
            case NVFP4, TRELLIS, Q3_G64_FP16, Q4_G64_FP16, Q5_G64_FP16 -> true;
            default -> false;
        };
    }

    private List<String> quantizedLinears(Artifact artifact, String prefix) {
        Map<String, TensorDescriptor> byName = new HashMap<>();
        for (TensorDescriptor tensor : artifact.tensors()) byName.put(tensor.name(), tensor);
        List<String> names = new ArrayList<>();
        for (TensorDescriptor tensor : artifact.tensors())
            if (tensor.name().startsWith(prefix) && quantized(tensor.format()) && !companion(byName, tensor.name()))
                names.add(tensor.name());
        return List.copyOf(names);
    }

    /// The objects of `option`'s source that come from the checkpoint as they are: raw (not quantized) objects that
    /// are not the MTP layer's and not companions.
    private Set<String> compared(Option option) {
        Map<String, TensorDescriptor> byName = descriptors.get(option);
        Set<String> names = new LinkedHashSet<>();
        for (TensorDescriptor tensor : artifacts.get(option).tensors()) {
            String name = tensor.name();
            if (name.startsWith("mtp/") || quantized(tensor.format()) || companion(byName, name)) continue;
            names.add(name);
        }
        return new TreeSet<>(names);
    }

    /// The first name in only one of `a` and `b`.
    private static String firstDifference(Set<String> a, Set<String> b) {
        Set<String> difference = new TreeSet<>(a);
        difference.addAll(b);
        Set<String> both = new TreeSet<>(a);
        both.retainAll(b);
        difference.removeAll(both);
        return difference.iterator().next();
    }

    private byte[] hash(Option option, TensorDescriptor tensor) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException failure) {
            throw new IllegalStateException(failure);
        }
        ByteBuffer buffer = ByteBuffer.allocate(HASH_BUFFER_BYTES);
        try (FileChannel channel = FileChannel.open(path(option), StandardOpenOption.READ)) {
            long position = tensor.dataOffset(), end = tensor.dataOffset() + tensor.byteSize();
            while (position < end) {
                buffer.clear().limit((int) Math.min(buffer.capacity(), end - position));
                int read = channel.read(buffer, position);
                if (read < 0) throw new IOException(path(option) + " is truncated in " + tensor.name());
                digest.update(buffer.array(), 0, read);
                position += read;
            }
        }
        return digest.digest();
    }
}
