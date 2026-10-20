package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Optional;

/// Whether the machine has what the jar cannot carry: the NVIDIA driver and a GPU it drives.
@FunctionalInterface
interface DriverProbe {
    /// Empty when the driver loads and sees a GPU; else what is missing.
    Optional<String> problem();

    /// Loads the driver library (`libcuda.so.1`, or `nvcuda.dll` on Windows) and counts its devices.
    static DriverProbe system() {
        return () -> {
            String name = NativeFiles.driverLibrary();
            SymbolLookup driver;
            try {
                driver = SymbolLookup.libraryLookup(name, Arena.global());
            } catch (IllegalArgumentException missing) {
                return Optional.of(name + " cannot be loaded");
            }
            Linker linker = Linker.nativeLinker();
            MethodHandle init = linker.downcallHandle(
                    driver.find("cuInit").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            MethodHandle count = linker.downcallHandle(
                    driver.find("cuDeviceGetCount").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
            try (Arena local = Arena.ofConfined()) {
                int status = (int) init.invokeExact(0);
                if (status != 0) return Optional.of("the driver finds no usable GPU (cuInit returned " + status + ")");
                MemorySegment devices = local.allocate(ValueLayout.JAVA_INT);
                status = (int) count.invokeExact(devices);
                if (status != 0 || devices.get(ValueLayout.JAVA_INT, 0) == 0)
                    return Optional.of("the driver finds no GPU");
                return Optional.empty();
            } catch (Throwable failure) {
                return Optional.of("the driver failed: " + failure);
            }
        };
    }
}
