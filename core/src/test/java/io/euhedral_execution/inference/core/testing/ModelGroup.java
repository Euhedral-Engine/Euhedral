package io.euhedral_execution.inference.core.testing;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.junit.jupiter.api.Tag;

/// The model artifact a CUDA test class loads. Classes of one group run one after another in one JVM, so the group's
/// model is loaded once ([SharedQwen38]); a class without a group needs no artifact, is cheap on the device, and
/// runs beside others. The group is also a JUnit tag, so a Gradle task selects a group's classes.
///
/// A class either works on the model directly (a group named for the artifact) or starts engines over it (the
/// `Engine` group of the artifact): an engine loads the artifact for itself, so the two never share the device.
public final class ModelGroup {
    public static final String Q3 = "model-q3";
    public static final String Q3_ENGINE = "engine-q3";
    public static final String NVFP4 = "model-nvfp4";
    public static final String NVFP4_ENGINE = "engine-nvfp4";
    public static final String FLASH_NEXT = "model-flash-next";
    public static final String FLASH_NEXT_ENGINE = "engine-flash-next";

    /// The compact Q3 Qwen3.8 artifact (`euhedral.qwen.artifact`), used through a shared [SharedQwen38.Loaded].
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(Q3)
    public @interface CompactQ3 {}

    /// Engines over the compact Q3 artifact.
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(Q3_ENGINE)
    public @interface CompactQ3Engine {}

    /// The NVFP4 Qwen3.8 artifact (`euhedral.qwen.nvfp4-artifact`), used through a shared [SharedQwen38.Loaded].
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(NVFP4)
    public @interface Nvfp4 {}

    /// Engines over the NVFP4 artifact.
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(NVFP4_ENGINE)
    public @interface Nvfp4Engine {}

    /// The Flash-Next artifact (`euhedral.qwen4.artifact`), used directly.
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(FLASH_NEXT)
    public @interface FlashNext {}

    /// Engines over the Flash-Next artifact.
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(FLASH_NEXT_ENGINE)
    public @interface FlashNextEngine {}

    private ModelGroup() {}
}
