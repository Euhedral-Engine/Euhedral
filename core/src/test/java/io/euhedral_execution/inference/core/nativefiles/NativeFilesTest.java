package io.euhedral_execution.inference.core.nativefiles;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.nio.file.attribute.PosixFileAttributeView;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NativeFilesTest {
    private static final String PLATFORM = "test-x64";

    @TempDir
    Path directory;

    private Path cache;
    private URLClassLoader resources;

    /// A jar's native resources: a product directory under `META-INF/euhedral-native/<platform>/` with its
    /// `files.sha256`, on a class loader of its own.
    @BeforeEach
    void productResources() throws IOException {
        Path classes = this.directory.resolve("classes");
        Path product = classes.resolve("META-INF/euhedral-native").resolve(PLATFORM);
        Map<String, byte[]> files = new TreeMap<>();
        files.put("lib/" + System.mapLibraryName("euhedral_cuda"), bytes("library", 4096));
        files.put("lib/" + System.mapLibraryName("llguidance"), bytes("guidance", 100));
        files.put("share/euhedral_cuda/attention/kernels.cu", bytes("kernel", 10));
        files.put("cuda/include/cuda_fp4.h", bytes("header", 10));
        StringBuilder hashes = new StringBuilder();
        for (var file : files.entrySet()) {
            Path path = product.resolve(file.getKey());
            Files.createDirectories(path.getParent());
            Files.write(path, file.getValue());
            hashes.append(sha256(file.getValue()))
                    .append("  ")
                    .append(file.getKey())
                    .append('\n');
        }
        Files.writeString(product.resolve(NativeFiles.HASHES), hashes);
        this.resources = new URLClassLoader(new URL[] {classes.toUri().toURL()}, null);
        this.cache = this.directory.resolve("cache");
    }

    private NativeFiles unpack() {
        return NativeFiles.unpack(this.cache, this.resources, PLATFORM);
    }

    @Test
    void unpacksTheLayoutIntoAHashNamedDirectory() throws IOException {
        NativeFiles files = unpack();
        assertEquals(this.cache, files.root().getParent());
        assertEquals(16, files.root().getFileName().toString().length());
        assertArrayEquals(bytes("library", 4096), Files.readAllBytes(files.library()));
        assertTrue(Files.isRegularFile(files.guidanceLibrary()));
        assertTrue(Files.isRegularFile(files.kernelSources().resolve("attention/kernels.cu")));
        assertTrue(Files.isRegularFile(files.cudaHeaders().resolve("cuda_fp4.h")));
        assertEquals(files.cudaHeaders(), NativeFiles.headersBeside(files.library()));
    }

    @Test
    void aSecondUnpackReusesTheDirectoryAndWritesNothing() throws IOException {
        NativeFiles first = unpack();
        FileTime written = Files.getLastModifiedTime(first.library());
        List<Path> before = listing(this.cache);
        // A cache that cannot be written still serves a complete directory.
        boolean readOnly = writable(this.cache, false);
        try {
            NativeFiles second = unpack();
            assertEquals(first, second);
        } finally {
            if (readOnly) writable(this.cache, true);
        }
        assertEquals(written, Files.getLastModifiedTime(first.library()));
        assertEquals(before, listing(this.cache));
    }

    @Test
    void concurrentUnpacksShareOneDirectoryAndLeaveNoTemporaryOne() throws Exception {
        CountDownLatch start = new CountDownLatch(1);
        List<AtomicReference<Object>> results = List.of(new AtomicReference<>(), new AtomicReference<>());
        List<Thread> threads = new ArrayList<>();
        for (AtomicReference<Object> result : results) {
            threads.add(Thread.startVirtualThread(() -> {
                try {
                    start.await();
                    result.set(unpack());
                } catch (Throwable failure) {
                    result.set(failure);
                }
            }));
        }
        start.countDown();
        for (Thread thread : threads) thread.join();
        assertTrue(
                results.get(0).get() instanceof NativeFiles,
                String.valueOf(results.get(0).get()));
        assertEquals(results.get(0).get(), results.get(1).get());
        NativeFiles files = (NativeFiles) results.get(0).get();
        try (Stream<Path> entries = Files.list(this.cache)) {
            assertEquals(List.of(files.root()), entries.toList());
        }
    }

    @Test
    void anUnwritableCacheNamesItsSetting() throws IOException {
        Files.createDirectories(this.cache);
        assumeTrue(writable(this.cache, false), "the file system has no POSIX permissions");
        try {
            assumeFalse(Files.isWritable(this.cache), "the cache stays writable (running as root)");
            var failure = assertThrows(NativeFiles.CacheException.class, this::unpack);
            assertTrue(failure.getMessage().startsWith("host.native-cache"), failure.getMessage());
            assertTrue(failure.getMessage().contains(this.cache.toString()), failure.getMessage());
        } finally {
            writable(this.cache, true);
        }
    }

    @Test
    void aTruncatedFileIsUnpackedAgain() throws IOException {
        NativeFiles first = unpack();
        Files.write(first.library(), bytes("library", 100));
        NativeFiles second = unpack();
        assertEquals(first, second);
        assertArrayEquals(bytes("library", 4096), Files.readAllBytes(second.library()));
        try (Stream<Path> entries = Files.list(this.cache)) {
            assertEquals(List.of(second.root()), entries.toList());
        }
    }

    @Test
    void theBuildLayoutIsUsedWhereItIs() {
        Path root = this.directory.resolve("build/native/linux-x64");
        NativeFiles files = NativeFiles.at(root);
        assertEquals(root.resolve("lib").resolve(System.mapLibraryName("euhedral_cuda")), files.library());
        assertEquals(root.resolve("cuda/include"), NativeFiles.headersBeside(files.library()));
        assertEquals(root.resolve("share/euhedral_cuda"), files.kernelSources());
    }

    /// Makes `directory` writable or read-only; false where the file system has no POSIX permissions (Windows).
    private static boolean writable(Path directory, boolean writable) throws IOException {
        if (!Files.getFileStore(directory).supportsFileAttributeView(PosixFileAttributeView.class)) return false;
        Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString(writable ? "rwxr-xr-x" : "r-xr-xr-x"));
        return true;
    }

    private static List<Path> listing(Path root) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.sorted().toList();
        }
    }

    private static byte[] bytes(String text, int length) {
        byte[] seed = text.getBytes(StandardCharsets.UTF_8);
        byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) bytes[i] = seed[i % seed.length];
        return bytes;
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
