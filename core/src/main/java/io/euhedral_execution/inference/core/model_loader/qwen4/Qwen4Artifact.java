package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.util.Map;
import java.util.Optional;

/// A parsed version 3 artifact: the header, the configuration, the fixed objects and the expert banks. Holds no
/// payload bytes.
public record Qwen4Artifact(Qwen4Header header, Qwen4Config config, Qwen4Tensor[] tensors, ExpertBank[] banks) {

    public Optional<Qwen4Tensor> tensor(String name) {
        for (Qwen4Tensor tensor : this.tensors) if (tensor.name().equals(name)) return Optional.of(tensor);
        return Optional.empty();
    }

    /// Fixed objects by name, in file order.
    public Map<String, Qwen4Tensor> tensorsByName() {
        Map<String, Qwen4Tensor> byName = new java.util.LinkedHashMap<>();
        for (Qwen4Tensor tensor : this.tensors) byName.put(tensor.name(), tensor);
        return byName;
    }

    public Optional<ExpertBank> bank(String name) {
        for (ExpertBank bank : this.banks) if (bank.name().equals(name)) return Optional.of(bank);
        return Optional.empty();
    }
}
