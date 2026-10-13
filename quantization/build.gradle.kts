import java.io.File

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
    implementation(libs.euhedral.core)
    implementation(libs.jackson.databind)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

val hostProductId = rootProject.extra["euhedral.native.host.id"] as String
val hostIncludeDirectory = rootProject.extra["euhedral.native.host.include"] as String
val hostRuntimeDirectory = rootProject.extra["euhedral.native.host.runtime"] as String

tasks.named<JavaExec>("run") {
    dependsOn(rootProject.tasks.named("nativeBuild${hostProductId.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}"))
    workingDir = rootProject.projectDir
    environment("EUHEDRAL_CUDA_INCLUDE_DIR", hostIncludeDirectory)
    val searchVariable = if (System.getProperty("os.name").startsWith("Windows")) "PATH" else "LD_LIBRARY_PATH"
    environment(searchVariable, hostRuntimeDirectory + File.pathSeparator + (System.getenv(searchVariable) ?: ""))
}

// The BF16 checkpoint for the tests that read it (they skip without one): -Peuhedral.quantization.checkpoint=DIR
tasks.withType<Test>().configureEach {
    providers.gradleProperty("euhedral.quantization.checkpoint").orNull?.let { systemProperty("euhedral.quantization.checkpoint", it) }
}
