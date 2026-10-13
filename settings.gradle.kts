pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

rootProject.name = "euhedral-inference"

include(":core")
include(":api")
include(":benchmark")
include(":quantization")