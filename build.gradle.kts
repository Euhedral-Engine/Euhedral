import org.gradle.api.plugins.JavaPluginExtension

import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test
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
            exclude("**/QwenEmbeddingCudaIntegrationTest.class")
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
            useJUnitPlatform()
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
            include("**/QwenEmbeddingCudaIntegrationTest.class")
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
            systemProperty(
                    "euhedral.cuda.library",
                    nativeBuildDirectory.get().dir(hostProductId).file("lib/$hostLibraryFilename").asFile.absolutePath)
            systemProperty(
                    "euhedral.qwen.artifact",
                    providers.gradleProperty("euhedral.qwen.artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl")
                            .get())
            // Optional NVFP4 artifact (tools/convert_qwen_safetensors_to_compact_edrl.py --profile nvfp4).
            systemProperty(
                    "euhedral.qwen.nvfp4-artifact",
                    providers.gradleProperty("euhedral.qwen.nvfp4-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_nvfp4.edrl")
                            .get())
            // Optional P2E2 transcode of the compact artifact (tools/convert_compact_edrl_to_p2e2.py).
            systemProperty(
                    "euhedral.qwen.p2e2-artifact",
                    providers.gradleProperty("euhedral.qwen.p2e2-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3_p2e2.edrl")
                            .get())
            systemProperty(
                    "euhedral.qwen.reference-artifact",
                    providers.gradleProperty("euhedral.qwen.reference-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_bf16.edrl")
                            .get())
            // Teacher-forced relaxed-numerics drift: decode length, prefill prefix, an optional per-step CSV
            // report, and `exact` to run the oracle on both sequences.
            for (name in listOf("steps", "prefix", "report", "candidate"))
                providers.gradleProperty("euhedral.numerics.drift.$name").orNull?.let { systemProperty("euhedral.numerics.drift.$name", it) }
            // Speculative decoding tests: artifact (defaults to the NVFP4 artifact), prompt length, rows.
            for (name in listOf("artifact", "prefix", "rows", "tokens", "host-mib", "native"))
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