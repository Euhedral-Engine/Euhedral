package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.runtime.graph.AbstractQuantum;

/// One kind of step a model's generation runs: a prompt chunk, a decode token, a verification, a draft. The model
/// provides its ports (each bound to the generation state it reads); the shared frames run them. [Admit] calls
/// [#admit]; the step's quantum throws `select` when it retires, and [Select] calls [#retired], which reads the result
/// and names the next port.
public interface StepPort {

    /// Starts the step: hands a quantum whose continuation is `select` to the runtime, or throws `select` into the
    /// lake itself when the step needs no quantum. After the hand-over it must not touch the generation's state:
    /// `select` may already run. Throws only when `select` will never be thrown.
    void admit(AbstractFrame select);

    /// The step concluded; `step` is its quantum, or null when the port threw `select` itself. Reads the result
    /// and returns the next port, or null once the generation is done (after [Generation#complete]). Throwing
    /// fails the generation.
    StepPort retired(AbstractQuantum step) throws Exception;
}
