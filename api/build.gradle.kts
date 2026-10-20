import org.springframework.boot.gradle.tasks.bundling.BootJar

plugins {
    java
    application
    alias(libs.plugins.spring.boot)
    alias(libs.plugins.spotless)
}

spotless {
    java {
        palantirJavaFormat("2.96.0")
    }
}

application {
    applicationName = "euhedral-inference-api"
    mainClass = "io.euhedral_execution.inference.api.EuhedralInferenceApplication"
    // The CUDA backend binds its native library through the FFM API.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

dependencies {
    implementation(platform(libs.spring.boot.dependencies))
    implementation(project(":core"))
    implementation(libs.spring.boot.starter.webmvc)
    implementation(libs.spring.boot.starter.actuator)
    implementation(libs.micrometer.registry.prometheus)
    testImplementation(platform(libs.spring.boot.dependencies))
    testImplementation(libs.spring.boot.starter.webmvc.test)
    testImplementation(libs.euhedral.core)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

// One server jar per platform. Each carries its platform's native product directory (the CUDA library, llguidance, the
// CUDA runtime and NVRTC, kernel sources and CUDA headers) under META-INF/euhedral-native/<platform>/, which the
// server unpacks into its native cache at startup. `bootJar` builds the host's.
@Suppress("UNCHECKED_CAST")
val nativeTasks = rootProject.extra["euhedral.native.tasks"] as Map<String, TaskProvider<*>>
val hostProductId = rootProject.extra["euhedral.native.host.id"] as String
val platformJars = nativeTasks.mapValues { (id, nativeTask) ->
    tasks.register<BootJar>("bootJar${id.split("-").joinToString("") { it.replaceFirstChar(Char::uppercase) }}") {
        group = "build"
        description = "Assemble the $id server jar with its native files."
        archiveFileName = "euhedral-inference-$id.jar"
        mainClass = application.mainClass
        targetJavaVersion = provider { java.targetCompatibility }
        classpath(sourceSets.main.map { it.runtimeClasspath })
        dependsOn(nativeTask)
        from(rootProject.layout.buildDirectory.dir("native/$id")) { into("META-INF/euhedral-native/$id") }
        manifest {
            attributes("Enable-Native-Access" to "ALL-UNNAMED")
        }
    }
}

tasks.bootJar {
    platformJars[hostProductId]?.let {
        enabled = false
        dependsOn(it)
    }
}

tasks.bootRun {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
