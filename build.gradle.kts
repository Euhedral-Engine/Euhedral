import org.gradle.api.plugins.JavaPluginExtension

import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test
import org.gradle.api.file.RegularFile
import org.gradle.api.provider.Provider
import org.gradle.api.tasks.TaskProvider
import org.gradle.jvm.toolchain.JavaLanguageVersion

plugins {
    alias(libs.plugins.spotless) apply false
    alias(libs.plugins.spring.boot) apply false
}

val nativeBuildDirectory = layout.buildDirectory.dir("native")
apply(from = "native.gradle.kts")
val hostProductId = extra["euhedral.native.host.id"] as String
val hostLibraryFilename = extra["euhedral.native.host.filename"] as String
val hostIncludeDirectory = extra["euhedral.native.host.include"] as String
val hostRuntimeDirectory = extra["euhedral.native.host.runtime"] as String
@Suppress("UNCHECKED_CAST")
val llguidanceTasks = extra["euhedral.llguidance.tasks"] as Map<String, TaskProvider<*>>
@Suppress("UNCHECKED_CAST")
val llguidanceFiles = extra["euhedral.llguidance.files"] as Map<String, Provider<RegularFile>>

// Common Java configuration applied to every subproject that applies the `java` plugin.
subprojects {
    plugins.withId("java") {
        the<JavaPluginExtension>().toolchain {
            languageVersion = JavaLanguageVersion.of(25)
        }
        tasks.named<Test>("test") {
            // Every *IntegrationTest needs a GPU (and usually an artifact); cudaIntegrationTest runs them.
            exclude("**/*IntegrationTest.class")
            useJUnitPlatform()
            // Host tests run concurrently, class by class; methods of a class stay on one thread. A class that
            // touches shared state opts out with @ResourceLock / @Isolated. Set here, not in
            // junit-platform.properties, so the CUDA tasks (which share GPU state) never run in parallel.
            systemProperty("junit.jupiter.execution.parallel.enabled", "true")
            systemProperty("junit.jupiter.execution.parallel.mode.default", "same_thread")
            systemProperty("junit.jupiter.execution.parallel.mode.classes.default", "concurrent")
            systemProperty("junit.jupiter.execution.parallel.config.strategy", "dynamic")
            systemProperty("junit.jupiter.execution.parallel.config.dynamic.factor", "0.5")
            // Shape goldens: `-Peuhedral.shapes.record=true` rewrites them instead of comparing.
            providers.gradleProperty("euhedral.shapes.record").orNull?.let { systemProperty("euhedral.shapes.record", it) }
            // Constrained decoding is CPU work; its tests load the host's llguidance build.
            llguidanceTasks[hostProductId]?.let { build ->
                dependsOn(build)
                systemProperty(
                        "euhedral.llguidance.library",
                        llguidanceFiles.getValue(hostProductId).get().asFile.absolutePath)
            }
        }
        val testSourceSet = the<SourceSetContainer>()["test"]
        // A CUDA class belongs to one of three kinds, told apart by JUnit tags (core's `ModelGroup` annotations):
        //  - light: no artifact and little device memory; several JVMs run these at once (cudaLightTest);
        //  - a model group: the class loads an artifact, directly or through an engine. The classes of all groups
        //    run one after another in ONE JVM, ordered by group, so each artifact is loaded once (cudaModelTest);
        //  - own JVM: the class cannot share a process (cudaOwnJvmTest, one JVM per class).
        // Flash-Next pins host memory sized from what its cgroup has left, so it runs in a Gradle run of its own, which
        // the gate gives a fresh cgroup, apart from the Qwen3.8 groups that fill the page cache.
        val qwen38GroupTags = listOf("model-q3", "engine-q3", "model-nvfp4", "engine-nvfp4")
        val flashNextTags = listOf("model-flash-next", "engine-flash-next")
        val modelGroupTags = qwen38GroupTags + flashNextTags
        val ownJvmTag = "own-jvm"
        val configureCuda: Test.() -> Unit = {
            group = "verification"
            dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
            testClassesDirs = testSourceSet.output.classesDirs
            classpath = testSourceSet.runtimeClasspath
            include("**/*IntegrationTest.class")
            systemProperty(
                    "euhedral.cuda.library",
                    nativeBuildDirectory.get().dir(hostProductId).file("lib/$hostLibraryFilename").asFile.absolutePath)
            systemProperty(
                    "euhedral.qwen.artifact",
                    providers.gradleProperty("euhedral.qwen.artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl")
                            .get())
            // Optional NVFP4 artifact (tools/convert_checkpoint.py --quantization nvfp4).
            systemProperty(
                    "euhedral.qwen.nvfp4-artifact",
                    providers.gradleProperty("euhedral.qwen.nvfp4-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl")
                            .get())
            // Optional Qwen3.8-Flash-Next artifact (qwen4_exp).
            systemProperty(
                    "euhedral.qwen4.artifact",
                    providers.gradleProperty("euhedral.qwen4.artifact")
                            .orElse("/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl")
                            .get())
            // Optional DFlash2 artifact (tools/convert_checkpoint.py --extend ... --dflash2 ...).
            systemProperty(
                    "euhedral.qwen.dflash2-artifact",
                    providers.gradleProperty("euhedral.qwen.dflash2-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4_compressed_dflash2_bf16.edrl")
                            .get())
            // Optional DFlash2 reference fixtures (tools/dflash2_reference.py) and the fixtures the engine writes.
            for (name in listOf("fixtures", "dump", "report", "verified", "prompt-tokens"))
                providers.gradleProperty("euhedral.dflash2.$name").orNull?.let { systemProperty("euhedral.dflash2.$name", it) }
            // Optional compressed Q3 artifact (tools/convert_checkpoint.py --quantization q3 --compressed).
            systemProperty(
                    "euhedral.qwen.q3-compressed-artifact",
                    providers.gradleProperty("euhedral.qwen.q3-compressed-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3_compressed.edrl")
                            .get())
            // Optional compressed NVFP4 artifact; the token-identity gate covers it.
            systemProperty(
                    "euhedral.qwen.nvfp4-compressed-artifact",
                    providers.gradleProperty("euhedral.qwen.nvfp4-compressed-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4_compressed.edrl")
                            .get())
            // Token-identity gate: `-Peuhedral.identity.record=true` rewrites the goldens instead of comparing.
            providers.gradleProperty("euhedral.identity.record").orNull?.let { systemProperty("euhedral.identity.record", it) }
            providers.gradleProperty("euhedral.identity.artifacts").orNull?.let { systemProperty("euhedral.identity.artifacts", it) }
            providers.gradleProperty("euhedral.shapes.record").orNull?.let { systemProperty("euhedral.shapes.record", it) }
            providers.gradleProperty("euhedral.shapes.artifacts").orNull?.let { systemProperty("euhedral.shapes.artifacts", it) }
            providers.gradleProperty("euhedral.twosessions.artifact").orNull?.let { systemProperty("euhedral.twosessions.artifact", it) }
            providers.gradleProperty("euhedral.twosessions.screen").orNull?.let { systemProperty("euhedral.twosessions.screen", it) }
            providers.gradleProperty("euhedral.scratch.artifact").orNull?.let { systemProperty("euhedral.scratch.artifact", it) }
            // Teacher-forced relaxed-numerics drift: decode length, prefill prefix, an optional per-step CSV
            // report, and `exact` to run the oracle on both sequences.
            for (name in listOf("steps", "prefix", "report", "candidate"))
                providers.gradleProperty("euhedral.numerics.drift.$name").orNull?.let { systemProperty("euhedral.numerics.drift.$name", it) }
            // Teacher-forced quality measurement (TeacherForcedQualityCudaIntegrationTest).
            for (name in listOf("report", "artifact", "prefix", "steps", "host-mib"))
                providers.gradleProperty("euhedral.quality.$name").orNull?.let { systemProperty("euhedral.quality.$name", it) }
            // MTP drafting-confidence report and draft-length screen (MtpDraftConfidence/MtpDraftLengthScreen tests).
            for (name in listOf("prompts", "report", "drafts", "context", "arms", "rounds"))
                providers.gradleProperty("euhedral.mtp.$name").orNull?.let { systemProperty("euhedral.mtp.$name", it) }
            // Speculative decoding tests: artifact (defaults to the NVFP4 artifact), prompt length, rows.
            for (name in listOf("artifact", "prefix", "rows", "tokens", "host-mib"))
                providers.gradleProperty("euhedral.speculative.$name").orNull?.let { systemProperty("euhedral.speculative.$name", it) }
            // Flash-Next fixture roots (tools/flash_next_reference.py) and test switches.
            for (name in listOf("fixtures", "model-fixtures", "verbose"))
                providers.gradleProperty("euhedral.qwen4.$name").orNull?.let { systemProperty("euhedral.qwen4.$name", it) }
            providers.gradleProperty("euhedral.test.heap").orNull?.let { maxHeapSize = it }
            jvmArgs("--enable-native-access=ALL-UNNAMED")
            environment("EUHEDRAL_CUDA_INCLUDE_DIR", hostIncludeDirectory)
            val searchVariable = if (System.getProperty("os.name").startsWith("Windows")) "PATH" else "LD_LIBRARY_PATH"
            environment(searchVariable, hostRuntimeDirectory + java.io.File.pathSeparator +
                    (System.getenv(searchVariable) ?: ""))
            useJUnitPlatform()
        }
        tasks.register<Test>("cudaIntegrationTest") {
            description = "Run the CUDA tests selected with --tests in one JVM; cudaTest runs the whole suite."
            configureCuda()
        }
        val cudaLight = tasks.register<Test>("cudaLightTest") {
            description = "Run the CUDA tests that need no artifact, several JVMs at once."
            configureCuda()
            useJUnitPlatform { excludeTags(*(modelGroupTags + ownJvmTag).toTypedArray()) }
            maxParallelForks = (providers.gradleProperty("cudaForks").orNull ?: "4").toInt()
        }
        val cudaModels = tasks.register<Test>("cudaModelTest") {
            description = "Run the CUDA tests of the Qwen3.8 model groups in one JVM, loading each artifact once."
            configureCuda()
            useJUnitPlatform { includeTags(*qwen38GroupTags.toTypedArray()) }
            mustRunAfter(cudaLight)
        }
        val cudaFlashNext = tasks.register<Test>("cudaFlashNextTest") {
            description = "Run the Flash-Next CUDA tests in one JVM; run it in a Gradle run of its own."
            configureCuda()
            useJUnitPlatform { includeTags(*flashNextTags.toTypedArray()) }
            mustRunAfter(cudaModels)
        }
        val cudaOwn = tasks.register<Test>("cudaOwnJvmTest") {
            description = "Run the CUDA tests that need a JVM of their own, one JVM each."
            configureCuda()
            useJUnitPlatform { includeTags(ownJvmTag) }
            forkEvery = 1
            mustRunAfter(cudaFlashNext)
        }
        tasks.register("cudaTest") {
            group = "verification"
            description = "Run every CUDA test: the light ones in parallel, then the model groups, then the own-JVM ones."
            dependsOn(cudaLight, cudaModels, cudaFlashNext, cudaOwn)
        }
    }
}

// Both modules' CUDA suites load models. Do not overlap them on one GPU.
gradle.projectsEvaluated {
    for (name in listOf("cudaLightTest", "cudaModelTest", "cudaFlashNextTest", "cudaOwnJvmTest")) {
        project(":api").tasks.named<Test>(name) {
            mustRunAfter(project(":core").tasks.matching { it.name.startsWith("cuda") })
        }
    }
}