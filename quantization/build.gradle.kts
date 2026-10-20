import java.io.File
import java.net.URI
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

plugins {
    java
    application
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

// Calibration and rounding tools (GPTQ) that write pre-rounded codes for the converter: not a dependency of `core`
// or `api`, and not packaged in the API JAR.
application {
    applicationName = "euhedral-quantization"
    mainClass = "io.euhedral_execution.inference.quantization.QuantizationMain"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

dependencies {
    implementation(project(":core"))
    // The checkpoint's chat template (QwenChatTemplate), rendered as the server renders prompts.
    implementation(project(":api"))
    implementation(libs.euhedral.core)
    implementation(libs.jackson.databind)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}


// cuBLAS (Hessian accumulation) and cuSOLVER (Cholesky factors) for this module only, from NVIDIA's CUDA 13.1
// redistributables (redistrib_13.1.0.json, matching the engine's cudart 13.1.80); Linux x86-64.
val cudaMathArchives = mapOf(
    "libcublas/linux-x86_64/libcublas-linux-x86_64-13.2.0.9-archive.tar.xz" to
        "e23c85f0d80b0c8af5e76a87dc400a123f455e4a603524ed6219616c33283cda",
    "libcusolver/linux-x86_64/libcusolver-linux-x86_64-12.0.7.41-archive.tar.xz" to
        "593c9fa25ccb72b37de40873ee22135a55c441754841983fd43858157f1e3373",
    "libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-13.1.80-archive.tar.xz" to
        "850cba213192691046441b5334588daa796c3dbce71253131fbf3c77e29ca18d",
    // cuSOLVER links cuSPARSE.
    "libcusparse/linux-x86_64/libcusparse-linux-x86_64-12.7.2.19-archive.tar.xz" to
        "7366b71dd438e06ebc744aa3c272a0423ec4c156bd5254f6cb1a58d5f61c570a",
)
val cudaMathDirectory = layout.buildDirectory.dir("cuda-math/lib")

val cudaMathLibraries = tasks.register("cudaMathLibraries") {
    group = "build"
    description = "Download cuBLAS, cuSOLVER, cuSPARSE and nvJitLink shared libraries for the quantization tools."
    outputs.dir(cudaMathDirectory)
    outputs.upToDateWhen { cudaMathDirectory.get().file("libcusparse.so.12").asFile.isFile }
    doLast {
        val cache = gradle.gradleUserHomeDir.toPath().resolve("caches/euhedral-cuda/13.1.80")
        val destination = cudaMathDirectory.get().asFile.toPath()
        Files.createDirectories(destination)
        for ((relative, sha) in cudaMathArchives) {
            val archive = cache.resolve(relative.substringAfterLast('/'))
            fun digest(): String = MessageDigest.getInstance("SHA-256").let { md ->
                Files.newInputStream(archive).use { input ->
                    val buffer = ByteArray(1 shl 20)
                    while (true) { val n = input.read(buffer); if (n < 0) break; md.update(buffer, 0, n) }
                }
                md.digest().joinToString("") { "%02x".format(it) }
            }
            if (!Files.isRegularFile(archive) || digest() != sha) {
                Files.createDirectories(cache)
                val partial = archive.resolveSibling(archive.fileName.toString() + ".partial")
                URI("https://developer.download.nvidia.com/compute/cuda/redist/$relative").toURL().openStream().use {
                    Files.copy(it, partial, StandardCopyOption.REPLACE_EXISTING)
                }
                Files.move(partial, archive, StandardCopyOption.REPLACE_EXISTING)
                check(digest() == sha) { "checksum mismatch for $relative" }
            }
            val unpacked = temporaryDir.resolve(archive.fileName.toString().removeSuffix(".tar.xz"))
            unpacked.deleteRecursively()
            unpacked.mkdirs()
            val tar = ProcessBuilder("tar", "-xJf", archive.toString(), "-C", unpacked.absolutePath).inheritIO().start()
            check(tar.waitFor() == 0) { "could not unpack $archive" }
            unpacked.walkTopDown().filter { it.isFile && it.name.contains(".so") && it.parentFile.name == "lib" }.forEach {
                Files.copy(it.toPath(), destination.resolve(it.name), StandardCopyOption.REPLACE_EXISTING)
            }
            unpacked.walkTopDown().filter { Files.isSymbolicLink(it.toPath()) && it.parentFile.name == "lib" }.forEach {
                val link = destination.resolve(it.name)
                Files.deleteIfExists(link)
                Files.createSymbolicLink(link, Files.readSymbolicLink(it.toPath()))
            }
            unpacked.deleteRecursively()
        }
    }
}

val hostProductId = rootProject.extra["euhedral.native.host.id"] as String

tasks.named<JavaExec>("run") {
    dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
    workingDir = rootProject.projectDir
    val searchVariable = if (System.getProperty("os.name").startsWith("Windows")) "PATH" else "LD_LIBRARY_PATH"
    dependsOn(cudaMathLibraries)
    val mathDirectory = cudaMathDirectory.get().asFile.absolutePath
    argumentProviders.add(CommandLineArgumentProvider { listOf("--cuda-libraries", mathDirectory) })
    environment(
        searchVariable,
        listOf(cudaMathDirectory.get().asFile.absolutePath, System.getenv(searchVariable) ?: "")
            .joinToString(File.pathSeparator))
}

// The BF16 checkpoint for the tests that read it (they skip without one): -Peuhedral.quantization.checkpoint=DIR;
// -Peuhedral.quantization.forward-fixture=DIR moves the forward tests' fixture.
tasks.withType<Test>().configureEach {
    for (name in listOf("euhedral.quantization.checkpoint", "euhedral.quantization.forward-fixture")) {
        providers.gradleProperty(name).orNull?.let { systemProperty(name, it) }
    }
}

// The CUDA test tasks the root build registers also need this module's cuBLAS/cuSOLVER.
tasks.withType<Test>().matching { it.name.startsWith("cuda") }.configureEach {
    dependsOn(cudaMathLibraries)
    systemProperty("euhedral.quantization.cuda-libraries", cudaMathDirectory.get().asFile.absolutePath)
    environment(
        "LD_LIBRARY_PATH",
        listOf(cudaMathDirectory.get().asFile.absolutePath, System.getenv("LD_LIBRARY_PATH") ?: "")
            .joinToString(File.pathSeparator))
}

// The trellis decode header is shared with the engine's kernels: it is compiled from the module's resources, where
// Cuda.compileResource expands the include.
tasks.named<ProcessResources>("processResources") {
    from(rootProject.layout.projectDirectory.dir("native/src/trellis")) {
        include("*.cuh")
        into("cuda/trellis")
    }
}
