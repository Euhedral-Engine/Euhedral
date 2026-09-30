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
            systemProperty(
                    "euhedral.cuda.library",
                    nativeBuildDirectory.get().dir(hostProductId).file("lib/$hostLibraryFilename").asFile.absolutePath)
            systemProperty(
                    "euhedral.qwen.artifact",
                    providers.gradleProperty("euhedral.qwen.artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_compact_q3.edrl")
                            .get())
            systemProperty(
                    "euhedral.qwen.reference-artifact",
                    providers.gradleProperty("euhedral.qwen.reference-artifact")
                            .orElse("/mnt/shared/qwen38-quant/artifacts/qwen3_5_27b_bf16.edrl")
                            .get())
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