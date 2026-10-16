package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen38.artifact.CompactArtifactWriter;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeMap;

/// A mixed artifact from calibrated [Sources] and an [Allocation]: every object of the base, except that the
/// projections the allocation names come from the source of their option, their companions with them.
public final class Assemble {

    /// Where an assembled file's payloads start, as in the artifacts the converter writes.
    private static final long PAYLOAD_ALIGNMENT = 4096;

    /// The default of `--copy-buffer-mib`.
    public static final long DEFAULT_COPY_BUFFER_MIB = 64;

    private Assemble() {}

    /// The composite artifact: the base's objects in its order, the chosen projections' descriptors and companions
    /// from their sources, and in `sources` the file each object is read from. `draftDepth`, when given, is the
    /// composite's stored MTP draft depth; otherwise the base's (if any) stays.
    ///
    /// @throws IllegalArgumentException naming the object (or option) for an object that is unknown, is not a
    ///     projection, is one of the MTP layer's, or whose option has no source
    public static Artifact view(Sources sources, Allocation allocation, OptionalInt draftDepth) {
        Option base = sources.base();
        Set<String> choosable = new HashSet<>(sources.projections());
        Map<String, Option> chosen = new TreeMap<>();
        for (var choice : new TreeMap<>(allocation.choices()).entrySet()) {
            String name = choice.getKey();
            Option option = choice.getValue();
            if (!choosable.contains(name)) {
                if (sources.drafterProjections().contains(name))
                    throw new IllegalArgumentException(name + " is a projection of the MTP layer, which is not chosen");
                boolean known = Arrays.stream(sources.artifact(base).tensors())
                        .anyMatch(tensor -> tensor.name().equals(name));
                throw new IllegalArgumentException(known ? name + " is not a projection" : "unknown object " + name);
            }
            if (!sources.options().contains(option))
                throw new IllegalArgumentException("no source for " + option.key() + " (chosen for " + name + ")");
            chosen.put(name, option);
        }
        Artifact baseArtifact = sources.artifact(base);
        List<TensorDescriptor> tensors = new ArrayList<>();
        Map<String, Path> files = new HashMap<>();
        for (TensorDescriptor tensor : baseArtifact.tensors()) {
            String name = tensor.name();
            Option option = chosen.get(name);
            if (option != null && option != base) {
                for (TensorDescriptor taken : sources.objectWithCompanions(option, name)) {
                    tensors.add(taken);
                    files.put(taken.name(), sources.path(option));
                }
            } else if (!companionOfReplaced(name, chosen, base)) {
                tensors.add(tensor);
                files.put(name, sources.path(base));
            }
        }
        var config = draftDepth.isPresent()
                ? baseArtifact.config().withDraftDepth(draftDepth.getAsInt())
                : baseArtifact.config();
        return new Artifact(baseArtifact.header(), config, tensors.toArray(TensorDescriptor[]::new), files);
    }

    private static boolean companionOfReplaced(String name, Map<String, Option> chosen, Option base) {
        for (String suffix : Sources.COMPANION_SUFFIXES) {
            if (!name.endsWith(suffix)) continue;
            Option option = chosen.get(name.substring(0, name.length() - suffix.length()));
            if (option != null && option != base) return true;
        }
        return false;
    }

    /// The payload bytes the assembled artifact holds: the chosen projections' and everything else of the base.
    public static long residentBytes(Sources sources, Allocation allocation) {
        long bytes = 0;
        for (TensorDescriptor tensor :
                view(sources, allocation, OptionalInt.empty()).tensors()) bytes += tensor.byteSize();
        return bytes;
    }

    /// [#write] with the default copy chunk, [#DEFAULT_COPY_BUFFER_MIB].
    public static void write(Sources sources, Allocation allocation, OptionalInt draftDepth, Path out)
            throws IOException {
        write(sources, allocation, draftDepth, out, DEFAULT_COPY_BUFFER_MIB << 20);
    }

    /// Writes the composite as a standalone artifact file at `out`: header, metadata and tensor table, then each
    /// payload copied exactly from its source file in chunks of at most `copyBufferBytes`, so no payload is ever
    /// held in memory. The file appears at `out` once complete.
    public static void write(
            Sources sources, Allocation allocation, OptionalInt draftDepth, Path out, long copyBufferBytes)
            throws IOException {
        if (copyBufferBytes < 1) throw new IllegalArgumentException("the copy buffer must be at least one byte");
        Artifact composite = view(sources, allocation, draftDepth);
        TensorDescriptor[] source = composite.tensors();
        long metadataSize = CompactArtifactWriter.metadataSize(composite.config());
        long tableOffset = ArtifactHeader.BYTE_SIZE + metadataSize;
        // The table's size does not depend on the offsets, so it is measured with offsets that cannot overlap.
        TensorDescriptor[] sized = new TensorDescriptor[source.length];
        long provisional = 0;
        for (int i = 0; i < source.length; i++) {
            sized[i] = at(source[i], provisional);
            provisional += source[i].byteSize();
        }
        long dataOffset = tableOffset + CompactArtifactWriter.tensorTableSize(sized);
        TensorDescriptor[] placed = new TensorDescriptor[source.length];
        long next = align(dataOffset);
        for (int i = 0; i < source.length; i++) {
            placed[i] = at(source[i], next);
            next = align(next + source[i].byteSize());
        }
        var header = new ArtifactHeader(
                ArtifactHeader.MAGIC,
                ArtifactHeader.COMPACT_VERSION,
                ArtifactHeader.BYTE_SIZE,
                metadataSize,
                tableOffset,
                placed.length,
                dataOffset);
        Path partial = out.resolveSibling(out.getFileName() + ".partial");
        Map<Path, FileChannel> inputs = new HashMap<>();
        try (FileChannel channel =
                CompactArtifactWriter.writeLayout(partial, new Artifact(header, composite.config(), placed))) {
            for (int i = 0; i < source.length; i++) {
                Path file = composite.pathOf(out, source[i]);
                FileChannel input = inputs.get(file);
                if (input == null) {
                    input = FileChannel.open(file, StandardOpenOption.READ);
                    inputs.put(file, input);
                }
                copy(
                        input,
                        source[i].dataOffset(),
                        channel,
                        placed[i].dataOffset(),
                        source[i].byteSize(),
                        copyBufferBytes);
            }
            channel.force(true);
        } catch (IOException | RuntimeException failure) {
            Files.deleteIfExists(partial);
            throw failure;
        } finally {
            for (FileChannel input : inputs.values()) input.close();
        }
        Files.move(partial, out, StandardCopyOption.REPLACE_EXISTING);
    }

    private static TensorDescriptor at(TensorDescriptor tensor, long offset) {
        return new TensorDescriptor(
                tensor.name(),
                tensor.shape(),
                tensor.dataType(),
                tensor.format(),
                tensor.layout(),
                offset,
                tensor.byteSize());
    }

    private static long align(long value) {
        return (value + PAYLOAD_ALIGNMENT - 1) / PAYLOAD_ALIGNMENT * PAYLOAD_ALIGNMENT;
    }

    private static void copy(FileChannel from, long fromOffset, FileChannel to, long toOffset, long bytes, long chunk)
            throws IOException {
        long done = 0;
        to.position(toOffset);
        while (done < bytes) {
            long moved = from.transferTo(fromOffset + done, Math.min(chunk, bytes - done), to);
            if (moved <= 0) throw new IOException("source ended " + (bytes - done) + " bytes early");
            done += moved;
        }
    }
}
