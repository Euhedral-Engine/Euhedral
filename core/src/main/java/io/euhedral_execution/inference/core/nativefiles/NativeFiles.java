package io.euhedral_execution.inference.core.nativefiles;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import io.euhedral_execution.hardware_utils.util.OSName;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

/// The native files the engine loads, laid out under one root: `lib/` (the CUDA library, llguidance, the CUDA
/// runtime and NVRTC), `share/euhedral_cuda/` (the kernel sources NVRTC compiles) and `cuda/include/` (the CUDA
/// headers they include). The server jar carries this layout for its platform under
/// `META-INF/euhedral-native/<platform>/` with a `files.sha256` listing every file's hash; [#unpack] copies it into
/// the native cache. Builds lay it out under `build/native/<platform>/` ([#at]).
///
/// @param root the directory holding `lib/`, `share/` and `cuda/`
public record NativeFiles(Path root) {
    /// The hash list at a layout's root: `<sha256>  <relative path>` per file.
    public static final String HASHES = "files.sha256";

    private static final String RESOURCES = "META-INF/euhedral-native/";
    private static final String SETTING = "host.native-cache";
    private static final int ATTEMPTS = 8;

    public NativeFiles {
        Objects.requireNonNull(root, "root");
    }

    /// The layout at `root`, as a build leaves it.
    public static NativeFiles at(Path root) {
        return new NativeFiles(root);
    }

    /// The CUDA library: `lib/libeuhedral_cuda.so`, or `lib/euhedral_cuda.dll` on Windows.
    public Path library() {
        return this.root.resolve("lib").resolve(System.mapLibraryName("euhedral_cuda"));
    }

    /// The constrained-decoding library, beside the CUDA library.
    public Path guidanceLibrary() {
        return this.root.resolve("lib").resolve(System.mapLibraryName("llguidance"));
    }

    /// The kernel sources NVRTC compiles.
    public Path kernelSources() {
        return this.root.resolve("share").resolve("euhedral_cuda");
    }

    /// The CUDA headers the kernels compile against.
    public Path cudaHeaders() {
        return headersBeside(library());
    }

    /// The CUDA headers of the layout that holds `library`: `cuda/include` beside its `lib/` directory.
    public static Path headersBeside(Path library) {
        return library.toAbsolutePath().getParent().resolveSibling("cuda").resolve("include");
    }

    /// The user's cache directory for native files: `%LOCALAPPDATA%\euhedral` on Windows, else
    /// `$XDG_CACHE_HOME/euhedral`, else `~/.cache/euhedral`.
    public static Path defaultCache() {
        if (OSName.isWindows()) {
            String local = System.getenv("LOCALAPPDATA");
            if (local != null && !local.isBlank()) return Path.of(local, "euhedral");
            return Path.of(System.getProperty("user.home"), "AppData", "Local", "euhedral");
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        if (xdg != null && !xdg.isBlank() && Path.of(xdg).isAbsolute()) return Path.of(xdg, "euhedral");
        return Path.of(System.getProperty("user.home"), ".cache", "euhedral");
    }

    /// The platform of the running JVM, as the jars name it: `linux-x64` or `windows-x64`, the only products.
    public static String platform() {
        if (SystemInfo.isX86() && OSName.isLinux()) return "linux-x64";
        if (SystemInfo.isX86() && OSName.isWindows()) return "windows-x64";
        throw new IllegalStateException("unsupported platform " + OSName.CURRENT_OS
                + (SystemInfo.isX86() ? "" : " (not x86-64)") + ": the server runs on Linux or Windows on x86-64");
    }

    /// The NVIDIA driver library the CUDA library links: `libcuda.so.1`, or `nvcuda.dll` on Windows.
    public static String driverLibrary() {
        return OSName.isWindows() ? "nvcuda.dll" : "libcuda.so.1";
    }

    /// Unpacks the running jar's native files for this platform into `cacheRoot/<content hash>/` and returns them.
    /// A complete directory already there is reused without writing; a damaged one is replaced.
    public static NativeFiles unpack(Path cacheRoot) {
        return unpack(cacheRoot, NativeFiles.class.getClassLoader(), platform());
    }

    /// Unpacks the files under `META-INF/euhedral-native/<platform>/` of `loader`. Each unpack writes a temporary
    /// sibling directory and renames it into place, so concurrent unpacks, in one process or several, never
    /// expose a partial directory; the first rename wins and the others discard their copy.
    static NativeFiles unpack(Path cacheRoot, ClassLoader loader, String platform) {
        String prefix = RESOURCES + platform + "/";
        byte[] listing = resource(loader, prefix + HASHES, platform);
        Map<String, String> hashes = parse(listing);
        Path target =
                cacheRoot.resolve(HexFormat.of().formatHex(sha256(listing)).substring(0, 16));
        for (int attempt = 0; attempt < ATTEMPTS; attempt++) {
            if (complete(target, hashes)) return new NativeFiles(target);
            Path temporary = null;
            try {
                Files.createDirectories(cacheRoot);
                temporary = Files.createTempDirectory(cacheRoot, target.getFileName() + ".unpack-");
                write(loader, prefix, hashes, temporary);
                Files.write(temporary.resolve(HASHES), listing);
                if (Files.exists(target) && !complete(target, hashes)) discard(cacheRoot, target);
                try {
                    Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
                    temporary = null;
                } catch (FileSystemException taken) {
                    // Another unpack renamed its copy first: reuse it if it is complete, else try again.
                }
            } catch (IOException | UncheckedIOException failure) {
                throw new CacheException(
                        SETTING + ": cannot unpack the native files into " + cacheRoot + ": " + failure.getMessage()
                                + "; set " + SETTING + " to a writable directory",
                        failure);
            } finally {
                if (temporary != null) deleteTree(temporary);
            }
        }
        if (complete(target, hashes)) return new NativeFiles(target);
        throw new CacheException(
                SETTING + ": the native files in " + target + " keep changing while they unpack", null);
    }

    /// Moves a damaged directory aside under a temporary name and deletes it; a concurrent unpack may have done so
    /// already.
    private static void discard(Path cacheRoot, Path target) throws IOException {
        Path aside = Files.createTempDirectory(cacheRoot, target.getFileName() + ".discard-");
        Files.delete(aside);
        try {
            Files.move(target, aside, StandardCopyOption.ATOMIC_MOVE);
        } catch (NoSuchFileException gone) {
            return;
        }
        deleteTree(aside);
    }

    private static void write(ClassLoader loader, String prefix, Map<String, String> hashes, Path directory)
            throws IOException {
        for (var file : hashes.entrySet()) {
            Path path = directory.resolve(file.getKey()).normalize();
            if (!path.startsWith(directory))
                throw new IllegalStateException("a native file escapes its directory: " + file.getKey());
            Files.createDirectories(path.getParent());
            MessageDigest digest = digest();
            try (InputStream source = loader.getResourceAsStream(prefix + file.getKey())) {
                if (source == null)
                    throw new IllegalStateException("the jar lacks its native file " + prefix + file.getKey());
                try (InputStream hashed = new DigestInputStream(source, digest);
                        OutputStream output = Files.newOutputStream(path)) {
                    hashed.transferTo(output);
                }
            }
            if (!HexFormat.of().formatHex(digest.digest()).equals(file.getValue()))
                throw new IllegalStateException("the jar's native file " + file.getKey() + " does not match its hash");
        }
    }

    /// Whether `directory` holds every listed file with its listed hash.
    private static boolean complete(Path directory, Map<String, String> hashes) {
        if (!Files.isDirectory(directory)) return false;
        try {
            for (var file : hashes.entrySet()) {
                Path path = directory.resolve(file.getKey());
                if (!Files.isRegularFile(path)) return false;
                MessageDigest digest = digest();
                try (InputStream input = new DigestInputStream(Files.newInputStream(path), digest)) {
                    input.transferTo(OutputStream.nullOutputStream());
                }
                if (!HexFormat.of().formatHex(digest.digest()).equals(file.getValue())) return false;
            }
            return true;
        } catch (IOException unreadable) {
            return false;
        }
    }

    private static Map<String, String> parse(byte[] listing) {
        Map<String, String> hashes = new LinkedHashMap<>();
        for (String line : new String(listing, StandardCharsets.UTF_8).split("\n")) {
            if (line.isBlank()) continue;
            int separator = line.indexOf("  ");
            if (separator != 64) throw new IllegalStateException("malformed " + HASHES + " line: " + line);
            hashes.put(line.substring(separator + 2), line.substring(0, separator));
        }
        if (hashes.isEmpty()) throw new IllegalStateException(HASHES + " lists no files");
        return hashes;
    }

    private static byte[] resource(ClassLoader loader, String name, String platform) {
        try (InputStream input = loader.getResourceAsStream(name)) {
            if (input == null)
                throw new IllegalStateException(
                        "this jar carries no native files for " + platform + "; use the jar built for " + platform);
            return input.readAllBytes();
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }

    private static void deleteTree(Path root) {
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(path);
        } catch (IOException ignored) {
            // A leftover temporary directory is harmless: its name never matches a content hash.
        }
    }

    private static byte[] sha256(byte[] bytes) {
        return digest().digest(bytes);
    }

    private static MessageDigest digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }

    /// The native cache cannot hold the unpacked files; the message names `host.native-cache`.
    public static final class CacheException extends RuntimeException {
        CacheException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
