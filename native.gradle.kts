import groovy.json.JsonSlurper
import java.io.File
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import java.util.zip.ZipFile
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.Zip
import org.tukaani.xz.XZInputStream

buildscript {
    repositories { mavenCentral() }
    dependencies {
        classpath("org.tukaani:xz:1.10")
        classpath("org.apache.commons:commons-compress:1.26.1")
    }
}

val manifest = JsonSlurper().parse(file("native/native-products.json")) as Map<*, *>
val products = (manifest["products"] as List<*>).map { it as Map<*, *> }
val redistributables = manifest["redistributables"] as Map<*, *>
val nativeRoot = layout.buildDirectory.dir("native")
val os = System.getProperty("os.name").lowercase(Locale.ROOT).let {
    when {
        it.startsWith("linux") -> "linux"
        it.startsWith("windows") -> "windows"
        else -> "unsupported"
    }
}
val architecture = System.getProperty("os.arch").lowercase(Locale.ROOT)
val host = products.firstOrNull { it["hostOs"] == os && architecture in listOf("amd64", "x86_64") }
val selected = providers.gradleProperty("euhedral.native.products").orNull
    ?.split(",")?.map(String::trim) ?: products.map { it["id"] as String }
require(selected.isNotEmpty() && selected.distinct().size == selected.size &&
    selected.all { id -> products.any { it["id"] == id } }) {
    "euhedral.native.products must contain unique supported IDs: ${products.map { it["id"] }}"
}

fun checksum(path: Path): String {
    val digest = MessageDigest.getInstance("SHA-256")
    Files.newInputStream(path).use { input ->
        val bytes = ByteArray(65536)
        while (true) {
            val length = input.read(bytes)
            if (length < 0) break
            digest.update(bytes, 0, length)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

fun fetch(cache: Path, relative: String, sha: String): Path {
    Files.createDirectories(cache)
    val archive = cache.resolve(relative.substringAfterLast('/'))
    if (Files.isRegularFile(archive) && checksum(archive) == sha) return archive
    val temporary = Files.createTempFile(cache, "cuda-", ".download")
    try {
        val connection = URI("https://developer.download.nvidia.com/compute/cuda/redist/$relative")
            .toURL().openConnection().apply { connectTimeout = 30000; readTimeout = 120000 }
        connection.getInputStream().use { source -> Files.newOutputStream(temporary).use { source.copyTo(it) } }
        check(checksum(temporary) == sha) { "CUDA download checksum mismatch: $relative" }
        Files.move(temporary, archive, StandardCopyOption.REPLACE_EXISTING)
    } finally { Files.deleteIfExists(temporary) }
    return archive
}

fun unpack(archive: Path, destination: Path, windows: Boolean) {
    fun install(name: String, source: java.io.InputStream) {
        val header = name.substringAfter("/include/", "")
        val base = name.substringAfterLast('/')
        val path = when {
            header.isNotEmpty() && !header.contains("..") -> "include/$header"
            windows && name.contains("/lib/x64/") && base in setOf("cuda.lib", "cudart.lib", "nvrtc.lib") -> "lib/$base"
            windows && name.contains("/bin/x64/") && base in setOf(
                "cudart64_13.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_131.dll") -> "runtime/$base"
            !windows && name.contains("/lib/stubs/") && base == "libcuda.so" -> "lib/libcuda.so"
            !windows && name.contains("/lib/") && base in setOf(
                "libcudart.so.13.1.80", "libnvrtc.so.13.1.80", "libnvrtc-builtins.so.13.1.80") -> when (base) {
                "libcudart.so.13.1.80" -> "lib/libcudart.so"
                "libnvrtc.so.13.1.80" -> "lib/libnvrtc.so"
                else -> "runtime/$base"
            }
            else -> return
        }
        val output = destination.resolve(path)
        Files.createDirectories(output.parent)
        Files.newOutputStream(output).use { source.copyTo(it) }
    }
    if (windows) {
        ZipFile(archive.toFile()).use { zip ->
            val entries = zip.entries()
            while (entries.hasMoreElements()) {
                val entry = entries.nextElement()
                if (!entry.isDirectory) zip.getInputStream(entry).use { install(entry.name, it) }
            }
        }
    } else {
        TarArchiveInputStream(XZInputStream(Files.newInputStream(archive))).use { tar ->
            while (true) {
                val entry = tar.nextEntry ?: break
                if (entry.isFile) install(entry.name, tar)
            }
        }
    }
}

fun headerVersion(include: File, header: String, macro: String): Int? = include.resolve(header)
    .takeIf(File::isFile)?.readText()?.let {
        Regex("#define\\s+$macro\\s+(\\d+)").find(it)?.groupValues?.get(1)?.toIntOrNull()
    }

fun coherentCudaInputs(include: File, library: File, windows: Boolean): Boolean {
    val runtime = headerVersion(include, "cuda_runtime_api.h", "CUDART_VERSION") ?: return false
    val driver = headerVersion(include, "cuda.h", "CUDA_VERSION") ?: return false
    val libraryRoot = if (windows && library.name.equals("x64", ignoreCase = true))
        library.canonicalFile.parentFile?.parentFile else library.canonicalFile.parentFile
    // NVIDIA's installed toolkit and the downloaded redistributables both colocate these inputs.
    return runtime >= 13010 && driver >= 13010 && runtime / 10 == driver / 10 &&
        include.canonicalFile.parentFile == libraryRoot && include.resolve("nvrtc.h").isFile
}

fun taskSuffix(id: String) = id.split('-').joinToString("") { it.replaceFirstChar(Char::uppercase) }

fun binaryImports(data: ByteArray, windows: Boolean): Set<String> {
    val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
    fun unsignedInt(at: Int) = buffer.getInt(at).toLong() and 0xffffffffL
    fun text(at: Int): String {
        check(at in data.indices) { "Invalid native import name offset $at" }
        var end = at
        while (end < data.size && data[end] != 0.toByte()) end++
        return String(data, at, end - at, Charsets.US_ASCII)
    }
    if (windows) {
        val pe = buffer.getInt(0x3c)
        val coff = pe + 4
        val optional = coff + 20
        check(buffer.getShort(optional).toInt() == 0x20b) { "Expected PE32+ image" }
        val importRva = unsignedInt(optional + 120)
        val sections = optional + (buffer.getShort(coff + 16).toInt() and 0xffff)
        val count = buffer.getShort(coff + 2).toInt() and 0xffff
        fun offset(rva: Long): Int {
            for (index in 0 until count) {
                val section = sections + index * 40
                val address = unsignedInt(section + 12)
                val size = minOf(unsignedInt(section + 8), unsignedInt(section + 16))
                if (rva >= address && rva < address + size)
                    return (unsignedInt(section + 20) + rva - address).toInt()
            }
            error("Unmapped PE import RVA $rva")
        }
        val imports = mutableSetOf<String>()
        var descriptor = offset(importRva)
        while ((0 until 5).any { unsignedInt(descriptor + it * 4) != 0L }) {
            imports += text(offset(unsignedInt(descriptor + 12)))
            descriptor += 20
        }
        return imports
    }
    val headers = mutableListOf<Triple<Int, Long, Long>>()
    val programOffset = buffer.getLong(32).toInt()
    val programSize = buffer.getShort(54).toInt() and 0xffff
    val programCount = buffer.getShort(56).toInt() and 0xffff
    var dynamic = 0
    var dynamicLength = 0
    for (index in 0 until programCount) {
        val entry = programOffset + index * programSize
        val type = buffer.getInt(entry)
        if (type == 1) headers += Triple(buffer.getLong(entry + 8).toInt(),
            buffer.getLong(entry + 16), buffer.getLong(entry + 32))
        if (type == 2) {
            dynamic = buffer.getLong(entry + 8).toInt()
            dynamicLength = buffer.getLong(entry + 32).toInt()
        }
    }
    fun offset(address: Long): Int {
        for ((fileOffset, virtual, size) in headers)
            if (address >= virtual && address < virtual + size) return fileOffset + (address - virtual).toInt()
        error("Unmapped ELF import address $address")
    }
    check(dynamicLength > 0) { "Missing ELF dynamic section" }
    val stringTable = (0 until dynamicLength step 16).firstNotNullOfOrNull { position ->
        if (buffer.getLong(dynamic + position) == 5L) buffer.getLong(dynamic + position + 8) else null
    } ?: error("Missing ELF dynamic string table")
    val table = offset(stringTable)
    return (0 until dynamicLength step 16).mapNotNull { position ->
        if (buffer.getLong(dynamic + position) == 1L)
            text(table + buffer.getLong(dynamic + position + 8).toInt()) else null
    }.toSet()
}

val nativeTasks = products.associate { product ->
    val id = product["id"] as String
    val windows = product["hostOs"] == "windows"
    val prefix = "euhedral.cuda.$id"
    val explicitInclude = providers.gradleProperty("$prefix.include-dir").orNull
        ?: if (product == host) providers.gradleProperty("euhedral.cuda.include-dir").orNull else null
    val explicitLibrary = providers.gradleProperty("$prefix.library-dir").orNull
        ?: if (product == host) providers.gradleProperty("euhedral.cuda.library-dir").orNull else null
    require((explicitInclude == null) == (explicitLibrary == null)) {
        "Both $prefix.include-dir and $prefix.library-dir are required for $id"
    }
    // An arbitrary host default (for example /usr/local/cuda) can produce PTX newer than the
    // installed driver accepts. Use the pinned runtime unless the caller selects a toolkit.
    val toolkitHome = if (product == host) {
        providers.environmentVariable("CUDA_HOME").orNull ?: providers.environmentVariable("CUDA_PATH").orNull
    } else null
    val toolkit = toolkitHome?.let(::file)
    val toolkitInclude = toolkit?.resolve(if (windows) "include" else "targets/x86_64-linux/include")
    val toolkitLibrary = toolkit?.resolve(if (windows) "lib/x64" else "targets/x86_64-linux/lib")
    val toolkitReady = explicitInclude == null &&
        !providers.gradleProperty("euhedral.cuda.force-download").map(String::toBoolean).getOrElse(false) &&
        toolkitInclude != null && toolkitLibrary != null &&
        coherentCudaInputs(toolkitInclude, toolkitLibrary, windows) &&
        toolkitLibrary.resolve(if (windows) "cudart.lib" else "libcudart.so").isFile &&
        toolkitLibrary.resolve(if (windows) "nvrtc.lib" else "libnvrtc.so").isFile &&
        toolkitLibrary.resolve(if (windows) "cuda.lib" else "stubs/libcuda.so").isFile
    val downloaded = layout.buildDirectory.dir("cuda-dev/$id")
    val include = explicitInclude?.let(::file) ?: if (toolkitReady) toolkitInclude!! else downloaded.get().dir("include").asFile
    val library = explicitLibrary?.let(::file) ?: if (toolkitReady) toolkitLibrary!! else downloaded.get().dir("lib").asFile
    val driverDirectory = providers.gradleProperty("$prefix.driver-library-dir").orNull?.let(::file)
        ?: if (windows) library else if (toolkitReady || explicitLibrary != null && library.resolve("stubs/libcuda.so").isFile)
            library.resolve("stubs") else library
    if (product == host) {
        extra["euhedral.native.host.include"] = include.absolutePath
        extra["euhedral.native.host.runtime"] = if (windows) {
            if (explicitLibrary != null || toolkitReady) {
                // The Windows toolkit installs import libraries in lib/x64 and DLLs in bin/x64.
                library.canonicalFile.parentFile.parentFile.resolve("bin/x64").absolutePath
            } else downloaded.get().dir("runtime").asFile.absolutePath
        } else if (explicitLibrary != null || toolkitReady) library.absolutePath
            else downloaded.get().dir("runtime").asFile.absolutePath
    }
    val verifyInputs = tasks.register("cudaInputs${taskSuffix(id)}") {
        group = "build"
        description = "Resolve CUDA 13.1+ build inputs for $id, without a GPU driver."
        if (explicitInclude == null && !toolkitReady) {
            outputs.dir(downloaded)
            val runtimeFiles = if (windows) listOf("cudart64_13.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_131.dll")
                else listOf("libcudart.so.13", "libnvrtc.so.13", "libnvrtc-builtins.so.13.1")
            outputs.upToDateWhen { runtimeFiles.all { downloaded.get().file("runtime/$it").asFile.isFile } }
            doLast {
                val platform = redistributables[product["cudaPlatform"]] as Map<*, *>
                val destination = downloaded.get().asFile.toPath()
                val cache = gradle.gradleUserHomeDir.toPath().resolve("caches/euhedral-cuda/${manifest["cudaVersion"]}")
                for (component in listOf("cudart", "nvrtc", "crt", "cccl")) {
                    val spec = platform[component] as Map<*, *>
                    unpack(fetch(cache, spec["path"] as String, spec["sha256"] as String), destination, windows)
                }
                if (!windows) {
                    for (stem in listOf("libcudart", "libnvrtc")) {
                        val runtime = destination.resolve("runtime/$stem.so.13")
                        Files.createDirectories(runtime.parent)
                        Files.copy(destination.resolve("lib/$stem.so"), runtime, StandardCopyOption.REPLACE_EXISTING)
                    }
                    Files.copy(destination.resolve("runtime/libnvrtc-builtins.so.13.1.80"),
                        destination.resolve("runtime/libnvrtc-builtins.so.13.1"), StandardCopyOption.REPLACE_EXISTING)
                }
            }
        }
    }
    val imports = layout.buildDirectory.dir("cuda-import/$id").get().asFile
    val driverImport = if (windows) tasks.register("cudaDriverImport${taskSuffix(id)}") {
        group = "build"
        dependsOn(verifyInputs)
        inputs.files(rootProject.file("native/nvcuda.def"), rootProject.file("native/cudart.def"))
        outputs.files(imports.resolve("nvcuda.lib"), imports.resolve("cudart.lib"))
        doLast {
            Files.createDirectories(imports.toPath())
            for (stem in listOf("nvcuda", "cudart")) {
                val zig = providers.environmentVariable("ZIG").orNull
                val launcher = listOf(zig ?: "zig")
                val command = launcher + listOf("dlltool", "-m", "i386:x86-64", "-d",
                    rootProject.file("native/$stem.def").absolutePath, "-l", imports.resolve("$stem.lib").absolutePath)
                check(ProcessBuilder(command).directory(rootProject.projectDir).inheritIO().start().waitFor() == 0) {
                    "Zig dlltool failed to generate $stem import library"
                }
            }
        }
    } else null
    val task = tasks.register<Exec>("nativeBuild${taskSuffix(id)}") {
        group = "build"
        description = "Cross-build the $id CUDA runtime product with pinned Zig."
        dependsOn(driverImport ?: verifyInputs)
        workingDir(rootProject.file("native"))
        val zig = providers.environmentVariable("ZIG").orNull
        val launcher = listOf(zig ?: "zig")
        commandLine(launcher + listOf("build", "-Dproduct-target=${product["zigTarget"]}",
            "-Doptimize=ReleaseSafe", "-Dcuda-include-dir=${include.absolutePath}",
            "-Dcuda-lib-dir=${library.absolutePath}",
            "-Dcuda-driver-lib-dir=${if (windows) imports.absolutePath else driverDirectory.absolutePath}",
            "--prefix", nativeRoot.get().dir(id).asFile.absolutePath))
        inputs.dir(rootProject.file("native/src"))
        inputs.dir(rootProject.file("native/include"))
        inputs.file(rootProject.file("native/build.zig"))
        inputs.file(rootProject.file("native/native-products.json"))
        inputs.dir(include)
        inputs.dir(library)
        if (windows) inputs.dir(imports)
        outputs.dir(nativeRoot.map { it.dir(id) })
        // Zig installs but never removes: clear the CUDA tree so a deleted source leaves no stale copy.
        doFirst { delete(nativeRoot.get().dir("$id/share/euhedral_cuda")) }
        doFirst {
            val required = if (windows) listOf("cuda.lib", "cudart.lib", "nvrtc.lib")
                else listOf("libcuda.so", "libcudart.so", "libnvrtc.so")
            val driver = if (windows) imports.resolve("nvcuda.lib") else driverDirectory.resolve("libcuda.so")
            require(coherentCudaInputs(include, library, windows) &&
                required.drop(1).all { library.resolve(it).isFile } && driver.isFile &&
                (!windows || imports.resolve("cudart.lib").isFile) &&
                include.resolve("nvrtc.h").isFile) {
                "CUDA 13.1+ inputs missing for $id. Pass -P$prefix.include-dir and " +
                    "-P$prefix.library-dir, set matching CUDA_HOME/CUDA_PATH, or enable NVIDIA downloads."
            }
        }
    }
    id to task
}

host?.let {
    extra["euhedral.native.host.id"] = it["id"] as String
    extra["euhedral.native.host.filename"] = it["filename"] as String
} ?: run {
    extra["euhedral.native.host.id"] = "unsupported"
    extra["euhedral.native.host.filename"] = "unsupported"
    extra["euhedral.native.host.include"] = "unsupported"
    extra["euhedral.native.host.runtime"] = "unsupported"
}

tasks.register("nativeBuild") {
    group = "build"
    description = "Build every supported CUDA native product."
    dependsOn(nativeTasks.values)
}

tasks.register<Zip>("nativePackage") {
    group = "distribution"
    description = "Package selected target products and NVRTC sources."
    dependsOn(selected.map(nativeTasks::getValue))
    archiveFileName = "euhedral-cuda-native.zip"
    destinationDirectory = layout.buildDirectory.dir("distributions")
    selected.forEach { id ->
        val product = products.single { it["id"] == id }
        from(nativeRoot.map { it.dir(id) }) {
            into(id)
            include("lib/${product["filename"]}", "share/euhedral_cuda/**/*.cu", "share/euhedral_cuda/**/*.cuh")
        }
    }
}

tasks.register("nativeVerify") {
    group = "verification"
    dependsOn(tasks.named("nativeBuild"))
    doLast {
        val sourceRoot = file("native/src")
        val sources = fileTree(sourceRoot) { include("**/*.cu", "**/*.cuh") }
            .files.map { it.relativeTo(sourceRoot).invariantSeparatorsPath }.toSet()
        for (product in products) {
            val prefix = nativeRoot.get().dir(product["id"] as String)
            val binary = prefix.file("lib/${product["filename"]}").asFile
            check(binary.isFile) { "Missing native product $binary" }
            val installedRoot = prefix.dir("share/euhedral_cuda").asFile
            val installed = fileTree(installedRoot) { include("**/*.cu", "**/*.cuh") }
                .files.map { it.relativeTo(installedRoot).invariantSeparatorsPath }.toSet()
            check(sources == installed) { "Missing NVRTC sources for ${product["id"]}: ${sources - installed}" }
            val signature = binary.inputStream().use { it.readNBytes(2) }
            check(signature.contentEquals(if (product["hostOs"] == "windows") byteArrayOf(77, 90)
                else byteArrayOf(127, 69))) { "Wrong binary format for ${product["id"]}" }
            val names = binaryImports(binary.readBytes(), product["hostOs"] == "windows")
            val imports = if (product["hostOs"] == "windows") {
                listOf("cudart64_13.dll", "nvrtc64_130_0.dll", "nvcuda.dll")
            } else {
                listOf("libcudart.so.13", "libnvrtc.so.13", "libcuda.so.1")
            }
            check(imports.all(names::contains) &&
                (product["hostOs"] != "windows" || names.none { it.endsWith(".so") }) &&
                (product["hostOs"] != "linux" || names.none { it.endsWith(".dll") })) {
                "CUDA imports do not match target ${product["id"]}"
            }
        }
    }
}
