plugins {
    `java-library`
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

dependencies {
    implementation(libs.euhedral.core)
    implementation(libs.jackson.databind)
    api(libs.slf4j.api)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.junit.platform.launcher)
}

// Euhedral's ControlPlaneLattice is a process-wide singleton. Every test class that starts one runs in
// its own JVM, so no lattice, worker, or static edge state can leak between classes.
val latticeTestClasses = listOf(
    "**/runtime/EuhedralInferenceRuntimeLatticeTest.class",
    "**/model/qwen38/SessionTest.class",
    "**/model/qwen38/SessionPrefixCacheTest.class",
    "**/InferenceEngineTest.class",
    "**/model/qwen4/LatticeHostTest.class",
)

val latticeTest = tasks.register<Test>("latticeTest") {
    group = "verification"
    description = "Run CPU tests that start the process-wide Euhedral lattice, one JVM per test class."
    val testSourceSet = sourceSets["test"]
    testClassesDirs = testSourceSet.output.classesDirs
    classpath = testSourceSet.runtimeClasspath
    include(latticeTestClasses)
    forkEvery = 1
    // Each JVM starts its own lattice on its own worker threads (no ports, no shared files), so JVMs can overlap.
    maxParallelForks = minOf(4, maxOf(1, Runtime.getRuntime().availableProcessors() / 4))
    useJUnitPlatform()
}

tasks.named<Test>("test") {
    exclude(latticeTestClasses)
    finalizedBy(latticeTest)
    // The converted Flash-Next artifact (tools/convert_flash_next.py); tests that need it skip when it is absent.
    for (name in listOf("artifact", "verify"))
        providers.gradleProperty("euhedral.qwen4.$name").orNull?.let { systemProperty("euhedral.qwen4.$name", it) }
}

tasks.named("check") {
    dependsOn(latticeTest)
}

val tokenizerReferenceDirectory = providers.gradleProperty("euhedral.qwen.tokenizer-dir")
    .orElse("/mnt/shared/qwen38-quant/source/qwen")

tasks.register<Test>("tokenizerReferenceTest") {
    group = "verification"
    description = "Verify Qwen tokenizer behavior against the selected checkpoint assets."
    val testSourceSet = sourceSets["test"]
    testClassesDirs = testSourceSet.output.classesDirs
    classpath = testSourceSet.runtimeClasspath
    include("**/QwenTokenizerTest.class")
    systemProperty("euhedral.qwen.tokenizer-dir", tokenizerReferenceDirectory.get())
    doFirst {
        val checkpoint = file(tokenizerReferenceDirectory.get())
        require(listOf("tokenizer.json", "tokenizer_config.json", "generation_config.json")
            .all { checkpoint.resolve(it).isFile }) {
            "Qwen tokenizer reference assets are required in $checkpoint; pass -Peuhedral.qwen.tokenizer-dir=..."
        }
    }
    useJUnitPlatform()
}