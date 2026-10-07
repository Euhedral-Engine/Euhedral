package io.euhedral_execution.inference.core.model.qwen4.loader;

import io.euhedral_execution.inference.core.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.artifact.Nvfp4Layout;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertProjection;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.CRC32;

/// Validates a parsed artifact against the inventory its configuration requires, and (optionally) the payload of
/// every object against the CRC-32 its table records.
public final class Validator {

    private static final int MAX_REPORTED = 20;
    private static final long BUFFER_BYTES = 64L << 20;

    private Validator() {}

    /// What a validation looked at, for logs and tests.
    public record Report(int tensors, int banks, int expertRecords, long checkedBytes) {}

    /// Checks the inventory: no missing or unexpected object, and every shape, format, layout and group as expected;
    /// expert banks (record count, projections, record geometry); the n-gram tables and their addressing; and the
    /// group sums the planner relies on. Throws with every problem found (the first few).
    public static Report validateInventory(Artifact artifact) throws ArtifactFormatException {
        Qwen4Config config = artifact.config();
        ExpectedInventory.Inventory expected = ExpectedInventory.expected(config);
        List<String> problems = new ArrayList<>();
        Map<String, Tensor> actual = artifact.tensorsByName();
        for (Map.Entry<String, ExpectedInventory.Expected> entry :
                expected.tensors().entrySet()) {
            Tensor tensor = actual.get(entry.getKey());
            ExpectedInventory.Expected want = entry.getValue();
            if (tensor == null) {
                problems.add("missing object " + entry.getKey());
                continue;
            }
            if (!Arrays.equals(tensor.shape(), want.shape()))
                problems.add(entry.getKey() + " has shape " + Arrays.toString(tensor.shape()) + ", expected "
                        + Arrays.toString(want.shape()));
            if (tensor.format() != want.format() || tensor.layout() != want.layout())
                problems.add(entry.getKey() + " is stored as " + tensor.format() + "/" + tensor.layout() + ", expected "
                        + want.format() + "/" + want.layout());
            if (tensor.group() != want.group())
                problems.add(entry.getKey() + " is in group " + tensor.group() + ", expected " + want.group());
        }
        for (String name : actual.keySet())
            if (!expected.tensors().containsKey(name)) problems.add("unexpected object " + name);

        Set<String> seenBanks = new HashSet<>();
        for (ExpectedInventory.ExpectedBank want : expected.banks()) {
            ExpertBank bank = artifact.bank(want.name()).orElse(null);
            if (bank == null) {
                problems.add("missing expert bank " + want.name());
                continue;
            }
            seenBanks.add(bank.name());
            checkBank(bank, want, problems);
        }
        for (ExpertBank bank : artifact.banks())
            if (!seenBanks.contains(bank.name())) problems.add("unexpected expert bank " + bank.name());

        checkNgram(config, problems);
        if (!problems.isEmpty()) {
            StringBuilder message =
                    new StringBuilder("artifact inventory is invalid (" + problems.size() + " problems):");
            for (int i = 0; i < Math.min(problems.size(), MAX_REPORTED); i++)
                message.append("\n  ").append(problems.get(i));
            if (problems.size() > MAX_REPORTED) message.append("\n  ...");
            throw new ArtifactFormatException(message.toString());
        }
        int records = 0;
        for (ExpertBank bank : artifact.banks()) records += bank.expertCount();
        return new Report(artifact.tensors().length, artifact.banks().length, records, 0);
    }

    private static void checkBank(ExpertBank bank, ExpectedInventory.ExpectedBank want, List<String> problems) {
        if (bank.group() != want.group())
            problems.add(bank.name() + " is in group " + bank.group() + ", expected " + want.group());
        if (bank.layer() != want.layer())
            problems.add(bank.name() + " is for layer " + bank.layer() + ", expected " + want.layer());
        if (bank.expertCount() != want.experts())
            problems.add(bank.name() + " has " + bank.expertCount() + " experts, expected " + want.experts());
        if (bank.projections().size() != want.projections().size())
            problems.add(bank.name() + " has " + bank.projections().size() + " projections, expected "
                    + want.projections().size());
        long cursor = 0;
        for (Map.Entry<String, long[]> projection : want.projections().entrySet()) {
            ExpertProjection actual = null;
            for (ExpertProjection candidate : bank.projections())
                if (candidate.name().equals(projection.getKey())) actual = candidate;
            if (actual == null) {
                problems.add(bank.name() + " lacks projection " + projection.getKey());
                continue;
            }
            if (!Arrays.equals(actual.shape(), projection.getValue()))
                problems.add(bank.name() + "/" + projection.getKey() + " has shape " + Arrays.toString(actual.shape())
                        + ", expected " + Arrays.toString(projection.getValue()));
            if (actual.format() != WeightFormat.NVFP4 || actual.layout() != WeightLayout.ROW_SPLIT_K128_V1)
                problems.add(bank.name() + "/" + projection.getKey() + " is not row-split NVFP4");
            if (actual.recordOffset() < cursor)
                problems.add(
                        bank.name() + "/" + projection.getKey() + " overlaps the previous projection in the record");
            cursor = actual.recordOffset() + actual.byteSize();
        }
        // Expert `e` is the e-th record: records follow one another in expert order, so an id is its position.
        for (int expert = 0; expert < bank.expertCount(); expert++) {
            if (bank.recordBytes(expert) < cursor) {
                problems.add(bank.name() + " expert " + expert + " has a record smaller than its projections");
                break;
            }
            if (expert > 0 && bank.fileOffset(expert) < bank.fileOffset(expert - 1) + bank.recordBytes(expert - 1)) {
                problems.add(bank.name() + " expert " + expert + " does not follow expert " + (expert - 1));
                break;
            }
        }
    }

    /// The tables address `heads` slices of one table of `shardRows * splitParts` rows: they must be contiguous and
    /// fit.
    private static void checkNgram(Qwen4Config config, List<String> problems) {
        Qwen4Config.Ngram ngram = config.ngram();
        for (int head = 0; head < ngram.heads(); head++) {
            long end = ngram.headsOffsets()[head] + ngram.headsVocabSizes()[head];
            if (end > ngram.totalRows())
                problems.add(
                        "n-gram head " + head + " addresses row " + end + " beyond the " + ngram.totalRows() + " rows");
        }
    }

    /// Reads every object's payload (in parallel) and compares its CRC-32 with the table's; the NVFP4 fixed tensors
    /// and every expert projection are also checked structurally (no NaN block scale, a finite non-negative global
    /// scale: [Nvfp4Layout#validate]).
    public static Report verifyChecksums(Path path, Artifact artifact, int threads) throws IOException {
        List<long[]> work = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<Integer> crcs = new ArrayList<>();
        // What to check structurally in each object: its NVFP4 matrices as {offsetInObject, rows, k, bytes}.
        List<long[][]> matrices = new ArrayList<>();
        for (Tensor tensor : artifact.tensors()) {
            work.add(new long[] {tensor.dataOffset(), tensor.byteSize()});
            labels.add(tensor.name());
            crcs.add(tensor.crc32());
            boolean rowSplit =
                    tensor.format() == WeightFormat.NVFP4 && tensor.layout() == WeightLayout.ROW_SPLIT_K128_V1;
            matrices.add(
                    rowSplit
                            ? new long[][] {{0, tensor.shape()[0], tensor.shape()[1], tensor.byteSize()}}
                            : new long[0][]);
        }
        long records = 0;
        for (ExpertBank bank : artifact.banks()) {
            for (int expert = 0; expert < bank.expertCount(); expert++) {
                work.add(new long[] {bank.fileOffset(expert), bank.recordBytes(expert)});
                labels.add(bank.name() + "#" + expert);
                crcs.add(bank.crc32(expert));
                long[][] projections = new long[bank.projections().size()][];
                for (int p = 0; p < projections.length; p++) {
                    ExpertProjection projection = bank.projections().get(p);
                    projections[p] = new long[] {
                        projection.recordOffset(),
                        projection.shape()[0],
                        projection.shape()[1],
                        projection.byteSize()
                    };
                }
                matrices.add(projections);
                records++;
            }
        }
        AtomicLong bytes = new AtomicLong();
        int workers = Math.max(1, threads);
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (int worker = 0; worker < workers; worker++) {
                final int start = worker;
                futures.add(pool.submit(() -> {
                    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
                            Arena arena = Arena.ofConfined()) {
                        MemorySegment buffer = arena.allocate(BUFFER_BYTES);
                        for (int index = start; index < work.size(); index += workers) {
                            long offset = work.get(index)[0];
                            long size = work.get(index)[1];
                            CRC32 crc = new CRC32();
                            for (long done = 0; done < size; ) {
                                int chunk = (int) Math.min(buffer.byteSize(), size - done);
                                ByteBuffer view = buffer.asSlice(0, chunk).asByteBuffer();
                                ArtifactFileAccess.readFully(channel, offset + done, view, labels.get(index));
                                view.flip();
                                crc.update(view);
                                done += chunk;
                            }
                            if (size <= buffer.byteSize()) {
                                // The whole object is in the buffer: check its matrices.
                                for (long[] matrix : matrices.get(index)) {
                                    try {
                                        Nvfp4Layout.validate(
                                                buffer.asSlice(matrix[0], matrix[3]), matrix[1], matrix[2]);
                                    } catch (IllegalArgumentException | IndexOutOfBoundsException invalid) {
                                        throw new ArtifactFormatException("invalid NVFP4 data in " + labels.get(index)
                                                + ": " + invalid.getMessage());
                                    }
                                }
                            }
                            if ((int) crc.getValue() != crcs.get(index))
                                throw new ArtifactFormatException("CRC-32 of " + labels.get(index) + " is "
                                        + Integer.toHexString((int) crc.getValue()) + ", the table records "
                                        + Integer.toHexString(crcs.get(index)));
                            bytes.addAndGet(size);
                        }
                    }
                    return null;
                }));
            }
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException failure) {
                    if (failure.getCause() instanceof IOException io) throw io;
                    throw new IOException("checksum verification failed", failure.getCause());
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IOException("checksum verification was interrupted", interrupted);
                }
            }
        } finally {
            pool.shutdownNow();
        }
        return new Report(artifact.tensors().length, artifact.banks().length, (int) records, bytes.get());
    }
}
