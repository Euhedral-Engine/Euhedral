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
public final class ModelGroup {
    /// The compact Q3 Qwen3.8 artifact (`euhedral.qwen.artifact`).
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(Q3)
    public @interface CompactQ3 {}

    /// The NVFP4 Qwen3.8 artifact (`euhedral.qwen.nvfp4-artifact`).
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(NVFP4)
    public @interface Nvfp4 {}

    /// The Flash-Next artifact (`euhedral.qwen4.artifact`).
    @Target(ElementType.TYPE)
    @Retention(RetentionPolicy.RUNTIME)
    @Documented
    @Tag(FLASH_NEXT)
    public @interface FlashNext {}

    public static final String Q3 = "model-q3";
    public static final String NVFP4 = "model-nvfp4";
    public static final String FLASH_NEXT = "model-flash-next";

    private ModelGroup() {}
}
