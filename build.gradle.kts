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
            exclude("**/CudaGpuMemoryIntegrationTest.class")
            exclude("**/CudaGpuStreamIntegrationTest.class")
            exclude("**/QwenCompactCudaResidencyIntegrationTest.class")
            exclude("**/CudaGpuOperationsIntegrationTest.class")
            exclude("**/QwenInstructionCudaIntegrationTest.class")
            exclude("**/QwenCompactCudaExecutionIntegrationTest.class")
            exclude("**/QwenLayerGpuOperationsIntegrationTest.class")
            exclude("**/QwenFirstLayerCudaIntegrationTest.class")
            exclude("**/QwenAttentionCudaIntegrationTest.class")
            exclude("**/QwenFullModelCudaIntegrationTest.class")
            exclude("**/QwenGenerationSessionCudaIntegrationTest.class")
            exclude("**/InferenceEngineCudaIntegrationTest.class")
            exclude("**/StreamOrderedEngineCudaIntegrationTest.class")
            exclude("**/ChatCompletionsCudaIntegrationTest.class")
            exclude("**/RelaxedNumericsDriftCudaIntegrationTest.class")
            exclude("**/QwenP2e2CudaIntegrationTest.class")
            exclude("**/QwenNvfp4CudaLoadIntegrationTest.class")
            exclude("**/QwenNvfp4GenerationCudaIntegrationTest.class")
            exclude("**/SpeculativeVerifyCudaIntegrationTest.class")
            exclude("**/SpeculativeDecodeCudaIntegrationTest.class")
            exclude("**/CapturedQuantaCudaIntegrationTest.class")
            exclude("**/TeacherForcedQualityCudaIntegrationTest.class")
            exclude("**/DFlash2*CudaIntegrationTest.class")
            exclude("**/Qwen4*CudaIntegrationTest.class")
            useJUnitPlatform()
            // Constrained decoding is CPU work; its tests load the host's llguidance build.
            llguidanceTasks[hostProductId]?.let { build ->
                dependsOn(build)
                systemProperty(
                        "euhedral.llguidance.library",
                        llguidanceFiles.getValue(hostProductId).get().asFile.absolutePath)
            }
        }
        val testSourceSet = the<SourceSetContainer>()["test"]
        tasks.register<Test>("cudaIntegrationTest") {
            group = "verification"
            description = "Run dedicated CUDA 13.1.x memory, operator, residency, and execution integration tests."
            dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
            testClassesDirs = testSourceSet.output.classesDirs
            classpath = testSourceSet.runtimeClasspath
            include("**/CudaGpuMemoryIntegrationTest.class")
            include("**/CudaGpuStreamIntegrationTest.class")
            include("**/QwenCompactCudaResidencyIntegrationTest.class")
            include("**/CudaGpuOperationsIntegrationTest.class")
            include("**/QwenInstructionCudaIntegrationTest.class")
            include("**/QwenCompactCudaExecutionIntegrationTest.class")
            include("**/QwenLayerGpuOperationsIntegrationTest.class")
            include("**/QwenFirstLayerCudaIntegrationTest.class")
            include("**/QwenAttentionCudaIntegrationTest.class")
            include("**/QwenFullModelCudaIntegrationTest.class")
            include("**/QwenGenerationSessionCudaIntegrationTest.class")
            include("**/InferenceEngineCudaIntegrationTest.class")
            include("**/StreamOrderedEngineCudaIntegrationTest.class")
            include("**/ChatCompletionsCudaIntegrationTest.class")
            include("**/RelaxedNumericsDriftCudaIntegrationTest.class")
            include("**/QwenP2e2CudaIntegrationTest.class")
            include("**/QwenNvfp4CudaLoadIntegrationTest.class")
            include("**/QwenNvfp4GenerationCudaIntegrationTest.class")
            include("**/SpeculativeVerifyCudaIntegrationTest.class")
            include("**/SpeculativeDecodeCudaIntegrationTest.class")
            include("**/CapturedQuantaCudaIntegrationTest.class")
            include("**/PrefillPartitionCudaIntegrationTest.class")
            include("**/PrefixCacheCudaIntegrationTest.class")
            include("**/TeacherForcedQualityCudaIntegrationTest.class")
            include("**/DFlash2*CudaIntegrationTest.class")
            include("**/Qwen4*CudaIntegrationTest.class")
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
            // Teacher-forced relaxed-numerics drift: decode length, prefill prefix, an optional per-step CSV
            // report, and `exact` to run the oracle on both sequences.
            for (name in listOf("steps", "prefix", "report", "candidate"))
                providers.gradleProperty("euhedral.numerics.drift.$name").orNull?.let { systemProperty("euhedral.numerics.drift.$name", it) }
            // Teacher-forced quality measurement (TeacherForcedQualityCudaIntegrationTest).
            for (name in listOf("report", "artifact", "prefix", "steps", "host-mib"))
                providers.gradleProperty("euhedral.quality.$name").orNull?.let { systemProperty("euhedral.quality.$name", it) }
            // Speculative decoding tests: artifact (defaults to the NVFP4 artifact), prompt length, rows.
            for (name in listOf("artifact", "prefix", "rows", "tokens", "host-mib"))
                providers.gradleProperty("euhedral.speculative.$name").orNull?.let { systemProperty("euhedral.speculative.$name", it) }
            jvmArgs("--enable-native-access=ALL-UNNAMED")
            // Each class loads the model and may start the process-wide Euhedral lattice singleton.
            forkEvery = 1
            environment("EUHEDRAL_CUDA_INCLUDE_DIR", hostIncludeDirectory)
            val searchVariable = if (System.getProperty("os.name").startsWith("Windows")) "PATH" else "LD_LIBRARY_PATH"
            environment(searchVariable, hostRuntimeDirectory + java.io.File.pathSeparator +
                    (System.getenv(searchVariable) ?: ""))
            useJUnitPlatform()
        }
    }
}

// Both CUDA integration suites load the compact model. Do not overlap them on one GPU.
gradle.projectsEvaluated {
    project(":api").tasks.named<Test>("cudaIntegrationTest") {
        mustRunAfter(project(":core").tasks.named<Test>("cudaIntegrationTest"))
    }
}