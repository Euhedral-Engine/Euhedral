package io.euhedral_execution.inference.core.testing;

import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.ClassDescriptor;
import org.junit.jupiter.api.ClassOrderer;
import org.junit.jupiter.api.ClassOrdererContext;
import org.junit.jupiter.api.Tag;
import org.junit.platform.commons.support.AnnotationSupport;

/// Runs the classes of one [ModelGroup] together, so a JVM that runs several groups loads each model once; classes
/// without a group go first, then the groups in the order of [#GROUPS].
public final class ModelGroupOrderer implements ClassOrderer {
    private static final List<String> GROUPS = List.of(
            ModelGroup.Q3,
            ModelGroup.Q3_ENGINE,
            ModelGroup.NVFP4,
            ModelGroup.NVFP4_ENGINE,
            ModelGroup.FLASH_NEXT,
            ModelGroup.FLASH_NEXT_ENGINE);

    @Override
    public void orderClasses(ClassOrdererContext context) {
        context.getClassDescriptors()
                .sort(Comparator.comparingInt(ModelGroupOrderer::rank)
                        .thenComparing(ModelGroupOrderer::mtp)
                        .thenComparing(descriptor -> descriptor.getTestClass().getName()));
    }

    private static boolean mtp(ClassDescriptor descriptor) {
        return AnnotationSupport.findRepeatableAnnotations(descriptor.getTestClass(), Tag.class).stream()
                .anyMatch(tag -> tag.value().equals(ModelGroup.MTP));
    }

    private static int rank(ClassDescriptor descriptor) {
        int rank = Integer.MAX_VALUE;
        for (Tag tag : AnnotationSupport.findRepeatableAnnotations(descriptor.getTestClass(), Tag.class)) {
            int group = GROUPS.indexOf(tag.value());
            if (group >= 0) rank = Math.min(rank, group + 1);
        }
        return rank == Integer.MAX_VALUE ? 0 : rank;
    }
}
