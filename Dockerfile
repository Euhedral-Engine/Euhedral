# syntax=docker/dockerfile:1
FROM eclipse-temurin:25-jdk-noble AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
ARG ZIG_SHA256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
RUN curl -fsSL https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz -o /tmp/zig.tar.xz \
    && printf '%s  %s\n' "$ZIG_SHA256" /tmp/zig.tar.xz | sha256sum -c - \
    && mkdir -p /opt/zig \
    && tar -xJf /tmp/zig.tar.xz --strip-components=1 -C /opt/zig \
    && rm /tmp/zig.tar.xz
ENV ZIG=/opt/zig/zig
# llguidance (constrained decoding) builds with the pinned Rust toolchain and cargo-zigbuild (mise.toml).
ARG RUSTUP_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:/opt/zig:$PATH
RUN curl -fsSL https://static.rust-lang.org/rustup/archive/1.29.0/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init \
    && printf '%s  %s\n' "$RUSTUP_SHA256" /tmp/rustup-init | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.95.0 \
    && rm /tmp/rustup-init \
    && cargo install cargo-zigbuild --version 0.23.4 --locked
COPY . .
RUN ./gradlew :api:bootJar -Peuhedral.native.products=linux-x64 \
    -Peuhedral.cuda.force-download=true --no-daemon
RUN "$JAVA_HOME/bin/jlink" --add-modules \
    java.base,java.compiler,java.desktop,java.instrument,java.logging,java.management,java.naming,java.net.http,java.prefs,java.security.jgss,java.security.sasl,java.sql,java.xml,jdk.crypto.ec,jdk.jfr,jdk.management,jdk.unsupported \
    --strip-debug --no-man-pages --no-header-files --compress=zip-6 --output /opt/euhedral-jre \
    && "$JAVA_HOME/bin/javac" -d /opt/health container/HealthProbe.java

FROM gcr.io/distroless/cc-debian13:nonroot
WORKDIR /opt/euhedral
COPY --from=build /opt/euhedral-jre/ /opt/euhedral/jre/
COPY --from=build /opt/health/ /opt/euhedral/health/
COPY --from=build /src/api/build/libs/euhedral-inference-api.jar /opt/euhedral/app.jar
COPY --from=build /src/build/native/linux-x64/lib/libeuhedral_cuda.so /opt/euhedral/lib/libeuhedral_cuda.so
COPY --from=build /src/build/native/linux-x64/lib/libllguidance.so /opt/euhedral/lib/libllguidance.so
COPY --from=build /src/build/native/linux-x64/share/euhedral_cuda/ /opt/euhedral/share/euhedral_cuda/
COPY --from=build /src/build/cuda-dev/linux-x64/runtime/ /opt/euhedral/lib/
COPY --from=build /src/build/cuda-dev/linux-x64/include/ /opt/euhedral/cuda/include/
ENV LD_LIBRARY_PATH=/opt/euhedral/lib \
    EUHEDRAL_CUDA_INCLUDE_DIR=/opt/euhedral/cuda/include \
    EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH=/opt/euhedral/lib/libeuhedral_cuda.so \
    EUHEDRAL_INFERENCE_ARTIFACT_PATH=/models/model.edrl \
    EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY=/tokenizer \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
# The model ID, worker CPUs and the other settings come from the run's environment: docker run --env-file .env
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/opt/euhedral/jre/bin/java", "-cp", "/opt/euhedral/health", "HealthProbe"]
ENTRYPOINT ["/opt/euhedral/jre/bin/java", "--enable-native-access=ALL-UNNAMED", "-jar", "/opt/euhedral/app.jar"]
