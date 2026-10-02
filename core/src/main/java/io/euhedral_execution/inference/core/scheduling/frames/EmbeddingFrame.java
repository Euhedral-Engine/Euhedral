package io.euhedral_execution.inference.core.scheduling.frames;

import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionContext;
import io.euhedral_execution.inference.core.scheduling.QwenExecutionPlan;
import io.euhedral_execution.inference.core.scheduling.graph.StageGraph;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// Runs the embedding lookup and owns its staged token IDs until the quantum retires.
public final class EmbeddingFrame extends QwenStageFrame {
    private ExecutionGpu.UploadBuffer pendingUpload;

    EmbeddingFrame(StageGraph graph, QwenExecutionPlan.Instruction instruction, ExecutionGpu gpu) {
        super(graph, instruction, gpu);
    }

    @Override
    protected void perform(QwenExecutionContext context, QwenExecutionPlan.Instruction instruction) {
        int[] ids = context.inputTokenIds();
        long bytes = (long) ids.length * Integer.BYTES;
        // Staged token IDs stay owned by this stage until the quantum's device work retires.
        ExecutionGpu.UploadBuffer upload = gpu().allocateUploadBuffer(bytes);
        this.pendingUpload = upload;
        MemorySegment host = upload.segment();
        for (int index = 0; index < ids.length; index++) {
            host.set(ValueLayout.JAVA_INT, (long) index * Integer.BYTES, ids[index]);
        }
        long tokenBuffer = context.workspace().tokenIdsAddress(bytes);
        gpu().copyUploadToDevice(tokenBuffer, upload);
        gpu().embedQ3(
                        tokenBuffer,
                        instruction.weightAddress(),
                        instruction.weightByteSize(),
                        context.workspace().hiddenStateAddress(),
                        ids.length,
                        context.plan().weights().config().vocabSize(),
                        instruction.outputWidth(),
                        instruction.weightLayout());
    }

    @Override
    protected void releaseTemporary(QwenExecutionContext context) {
        ExecutionGpu.UploadBuffer upload = this.pendingUpload;
        this.pendingUpload = null;
        // A poisoned GPU cannot prove that DMA has stopped reading pinned host memory.
        if (upload != null && gpu().completionProven()) upload.close();
    }
}
