package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.quantization.mixed.Cost.Key;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/// The choosable objects of a search in the order the fidelity model numbers them, the options that can be
/// chosen, and what each choice costs: the bytes of the object with its companions, and the prefill and
/// verification microseconds the cost table gives its launch.
final class Space {

    final Sources sources;
    final List<String> objects;
    final List<Option> options;
    final long[][] bytes;
    /// Microseconds of one prefill token (a launch at [CostTable#PREFILL_ROWS] rows over the rows) and of one
    /// verification step, by object and option.
    final double[][] prefill, decode;

    private final Map<String, Integer> index = new HashMap<>();
    private final Map<Option, Integer> optionIndex = new HashMap<>();

    /// @throws IllegalArgumentException naming the cost-table entry that is missing
    Space(Sources sources, List<Option> options, CostTable costs, int verifyRows) {
        this.sources = sources;
        this.objects = FidelityModel.sortedByLayer(sources.projections());
        this.options = List.copyOf(options);
        int n = objects.size(), k = options.size();
        bytes = new long[n][k];
        prefill = new double[n][k];
        decode = new double[n][k];
        for (int i = 0; i < n; i++) {
            index.put(objects.get(i), i);
            for (int j = 0; j < k; j++) {
                long total = 0;
                for (TensorDescriptor part : sources.objectWithCompanions(options.get(j), objects.get(i)))
                    total += part.byteSize();
                bytes[i][j] = total;
                var shape = Cost.shape(sources.descriptor(options.get(j), objects.get(i)));
                prefill[i][j] =
                        costs.micros(new Key(shape, options.get(j), CostTable.PREFILL_ROWS)) / CostTable.PREFILL_ROWS;
                decode[i][j] = costs.micros(new Key(shape, options.get(j), verifyRows));
            }
        }
        for (int j = 0; j < k; j++) optionIndex.put(options.get(j), j);
    }

    Allocation allocation(int[] choice) {
        Map<String, Option> choices = new TreeMap<>();
        for (int i = 0; i < choice.length; i++) choices.put(objects.get(i), options.get(choice[i]));
        return new Allocation(choices);
    }

    /// The option indices of a full allocation.
    int[] choice(Allocation allocation) {
        int[] choice = new int[objects.size()];
        for (int i = 0; i < choice.length; i++) {
            Option option = allocation.choices().get(objects.get(i));
            if (option == null)
                throw new IllegalArgumentException("the allocation has no option for " + objects.get(i));
            choice[i] = optionIndex.get(option);
        }
        return choice;
    }

    /// `allocation` with every object it does not name on the base's option.
    ///
    /// @throws IllegalArgumentException naming an object that is not a choosable projection or an option that
    /// this     search does not use
    Allocation normalize(Allocation allocation) {
        for (var entry : allocation.choices().entrySet()) {
            if (!index.containsKey(entry.getKey()))
                throw new IllegalArgumentException(entry.getKey() + " is not a projection that can be chosen");
            if (!optionIndex.containsKey(entry.getValue()))
                throw new IllegalArgumentException(
                        "no source and prior for " + entry.getValue().key() + " (chosen for " + entry.getKey() + ")");
        }
        Map<String, Option> choices = new TreeMap<>();
        for (String object : objects) choices.put(object, allocation.choices().getOrDefault(object, sources.base()));
        return new Allocation(choices);
    }

    long bytesOf(int[] choice) {
        long total = 0;
        for (int i = 0; i < choice.length; i++) total += bytes[i][choice[i]];
        return total;
    }

    int cheapest(int object) {
        int best = 0;
        for (int j = 1; j < options.size(); j++) if (bytes[object][j] < bytes[object][best]) best = j;
        return best;
    }

    int dearest(int object) {
        int best = 0;
        for (int j = 1; j < options.size(); j++) if (bytes[object][j] > bytes[object][best]) best = j;
        return best;
    }

    long minimumBytes() {
        long total = 0;
        for (int i = 0; i < objects.size(); i++) total += bytes[i][cheapest(i)];
        return total;
    }

    long maximumBytes() {
        long total = 0;
        for (int i = 0; i < objects.size(); i++) total += bytes[i][dearest(i)];
        return total;
    }

    /// The options of every object in a list of the order's length: `option` for all.
    int[] uniform(Option option) {
        int[] choice = new int[objects.size()];
        Arrays.fill(choice, optionIndex.get(option));
        return choice;
    }

    List<Allocation> pure() {
        List<Allocation> all = new ArrayList<>();
        for (Option option : options) all.add(allocation(uniform(option)));
        return all;
    }
}
