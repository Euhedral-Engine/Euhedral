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
    private static final List<String> GROUPS = List.of(ModelGroup.Q3, ModelGroup.NVFP4, ModelGroup.FLASH_NEXT);

    @Override
    public void orderClasses(ClassOrdererContext context) {
        context.getClassDescriptors()
                .sort(Comparator.comparingInt(ModelGroupOrderer::rank)
                        .thenComparing(descriptor -> descriptor.getTestClass().getName()));
    }

    private static int rank(ClassDescriptor descriptor) {
        for (Tag tag : AnnotationSupport.findRepeatableAnnotations(descriptor.getTestClass(), Tag.class)) {
            int group = GROUPS.indexOf(tag.value());
            if (group >= 0) return group + 1;
        }
        return 0;
    }
}
