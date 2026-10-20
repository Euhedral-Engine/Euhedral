package io.euhedral_execution.inference.api;

import io.euhedral_execution.inference.core.nativefiles.NativeFiles;
import io.euhedral_execution.inference.core.settings.Settings;
import io.euhedral_execution.inference.core.settings.SettingsException;
import io.euhedral_execution.inference.core.settings.SettingsLoader;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/// OpenAI-compatible HTTP entry point for Euhedral-Inference.
///
/// The settings file is read once, before Spring starts, by [SettingsLoader]; Spring receives the resulting records as
/// beans and adapts HTTP to the core engine. The single `InferenceEngine` bean owns the model, GPU, lattice, and
/// runtime; each request owns one generation session for its lifetime.
@SpringBootApplication
public class EuhedralInferenceApplication {

    /// The exit status when the settings cannot be used.
    static final int INVALID_SETTINGS = 2;

    static void main(String[] args) {
        int status = start(
                args,
                System.getenv(),
                System.err,
                (settings, nativeFiles) ->
                        ShutdownSignals.install(application(settings, nativeFiles, EuhedralInferenceApplication.class)
                                .run(springArguments(args))));
        if (status != 0) System.exit(status);
    }

    /// The command-line arguments without `--config` and its value, which only the settings loader reads.
    static String[] springArguments(String[] args) {
        List<String> kept = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--config")) i++;
            else if (!args[i].startsWith("--config=")) kept.add(args[i]);
        }
        return kept.toArray(String[]::new);
    }

    /// What starts the server once the settings are read and the native files are unpacked.
    interface Launcher {
        void launch(Settings settings, NativeFiles nativeFiles);
    }

    /// Reads the settings and unpacks the native files, then launches; the exit status is 0 after a launch, and
    /// [#INVALID_SETTINGS] after printing each problem on its own line.
    static int start(String[] args, Map<String, String> environment, PrintStream err, Launcher launcher) {
        Optional<Settings> settings = read(args, environment, err);
        Optional<NativeFiles> nativeFiles = settings.flatMap(found -> unpack(found, err));
        if (nativeFiles.isEmpty()) return INVALID_SETTINGS;
        launcher.launch(settings.get(), nativeFiles.get());
        return 0;
    }

    /// The settings the command line and environment name; empty after printing each problem on its own line.
    private static Optional<Settings> read(String[] args, Map<String, String> environment, PrintStream err) {
        try {
            return Optional.of(SettingsLoader.load(SettingsLoader.locate(args, environment), environment));
        } catch (SettingsException invalid) {
            invalid.problems().forEach(err::println);
            return Optional.empty();
        }
    }

    /// The jar's native files, unpacked into the native cache (`host.native-cache`, else the user's cache directory);
    /// empty after printing why not.
    private static Optional<NativeFiles> unpack(Settings settings, PrintStream err) {
        try {
            return Optional.of(NativeFiles.unpack(settings.host().nativeCache().orElseGet(NativeFiles::defaultCache)));
        } catch (NativeFiles.CacheException | IllegalStateException | UncheckedIOException failure) {
            err.println(failure.getMessage());
            return Optional.empty();
        }
    }

    /// The application with the settings registered as singletons, and the values Spring itself needs (the port and
    /// as default property.
    static SpringApplication application(Settings settings, NativeFiles nativeFiles, Class<?>... sources) {
        SpringApplication application = new SpringApplication(sources);
        application.addInitializers(context -> {
            var beans = context.getBeanFactory();
            beans.registerSingleton("settings", settings);
            beans.registerSingleton("requestSettings", settings.server().requests());
            beans.registerSingleton("nativeFiles", nativeFiles);
        });
        application.setDefaultProperties(
                Map.of("server.port", String.valueOf(settings.server().port())));
        return application;
    }
}
