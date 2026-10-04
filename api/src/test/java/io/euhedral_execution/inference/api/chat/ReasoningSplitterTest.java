package io.euhedral_execution.inference.api.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ReasoningSplitterTest {

    /// Every way of cutting the output into two chunks splits it the same way.
    @Test
    void theSplitDoesNotDependOnChunkBoundaries() {
        String output = " \nFirst <b>step</b>, then </thin k.\n\n</think>\n\n Answer </think> stays.";
        for (int cut = 0; cut <= output.length(); cut++) {
            Recorder recorder = new Recorder();
            var splitter = new ReasoningSplitter();
            splitter.accept(output.substring(0, cut), recorder);
            splitter.accept(output.substring(cut), recorder);
            splitter.finish(recorder);
            assertEquals("First <b>step</b>, then </thin k.", recorder.reasoning(), "cut " + cut);
            assertEquals("Answer </think> stays.", recorder.answer(), "cut " + cut);
            assertTrue(splitter.answering());
        }
    }

    @Test
    void reasoningStreamsAsItArrivesAndHoldsOnlyAPossibleTag() {
        Recorder recorder = new Recorder();
        var splitter = new ReasoningSplitter();
        splitter.accept("Thinking about it", recorder);
        assertEquals(List.of("Thinking about it"), recorder.parts);
        splitter.accept(" more\n</thi", recorder);
        assertEquals(List.of("Thinking about it", " more"), recorder.parts);
        splitter.accept("s is text", recorder);
        assertEquals("Thinking about it more\n</this is text", recorder.reasoning());
        assertFalse(splitter.answering());
    }

    @Test
    void reasoningThatNeverEndsIsTrimmedAtTheEnd() {
        Recorder recorder = new Recorder();
        var splitter = new ReasoningSplitter();
        splitter.accept("Unfinished  \n", recorder);
        splitter.finish(recorder);
        assertEquals("Unfinished", recorder.reasoning());
        assertEquals("", recorder.answer());
    }

    private static final class Recorder implements ReasoningSplitter.Output {
        final List<String> parts = new ArrayList<>();
        final StringBuilder reasoning = new StringBuilder();
        final StringBuilder answer = new StringBuilder();

        @Override
        public void reasoning(String text) {
            assertEquals(0, this.answer.length(), "reasoning after the answer began");
            this.parts.add(text);
            this.reasoning.append(text);
        }

        @Override
        public void answer(String text) {
            this.answer.append(text);
        }

        String reasoning() {
            return this.reasoning.toString();
        }

        String answer() {
            return this.answer.toString();
        }
    }
}
