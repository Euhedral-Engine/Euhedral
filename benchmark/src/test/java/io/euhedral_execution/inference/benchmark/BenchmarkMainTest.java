package io.euhedral_execution.inference.benchmark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.benchmark.run.BenchmarkRunner;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class BenchmarkMainTest {
    @Test
    void measuredIneligibleRowsDoNotMakeTheRunSuccessful() throws Exception {
        var scenario = new Scenario(Scenario.Kind.SUSTAINED_DECODE, 2, 8);
        var options = BenchmarkFixtures.options(List.of(scenario), 1, 1);
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                options,
                BenchmarkFixtures.prompts(List.of(scenario)),
                () -> {
                    var target = new BenchmarkFixtures.FakeTarget();
                    target.eosAfter = 3;
                    return target;
                },
                BenchmarkFixtures.context(),
                rows::add);
        assertEquals(BenchmarkResult.INELIGIBLE, rows.getLast().status());
        assertEquals(1, BenchmarkMain.measuredExitCode(rows));
    }

    @Test
    void failedWarmupDoesNotInvalidateASuccessfulMeasurement() throws Exception {
        var scenario = new Scenario(Scenario.Kind.PREFILL, 2, 0);
        var options = BenchmarkFixtures.options(List.of(scenario), 1, 1);
        List<BenchmarkResult> rows = new ArrayList<>();
        BenchmarkRunner.run(
                options,
                BenchmarkFixtures.prompts(List.of(scenario)),
                () -> {
                    var target = new BenchmarkFixtures.FakeTarget();
                    target.failOnCall = 0;
                    return target;
                },
                BenchmarkFixtures.context(),
                rows::add);
        assertEquals(BenchmarkResult.FAILED, rows.getFirst().status());
        assertEquals(BenchmarkResult.SUCCESS, rows.getLast().status());
        assertEquals(0, BenchmarkMain.measuredExitCode(rows));
    }
}
