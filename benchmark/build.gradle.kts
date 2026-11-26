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

// Benchmark harness only: not a dependency of production `core` or `api`, and not packaged in the API JAR.
application {
    applicationName = "euhedral-inference-benchmark"
    mainClass = "io.euhedral_execution.inference.benchmark.BenchmarkMain"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

dependencies {
    implementation(project(":core"))
    implementation(libs.jackson.databind)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

val hostProductId = rootProject.extra["euhedral.native.host.id"] as String

// Matches cudaIntegrationTest: build the host native library, which carries its CUDA runtime, NVRTC and headers.
tasks.named<JavaExec>("run") {
    dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
    workingDir = rootProject.projectDir
}

// The prompt document the Rust bench also includes lives under rust/resources; it keeps its classpath path.
tasks.named<ProcessResources>("processResources") {
    from(rootProject.layout.projectDirectory.dir("rust/resources/bench")) {
        include("chat-corpus-v1-document.md")
        into("io/euhedral_execution/inference/benchmark/prompt")
    }
}
