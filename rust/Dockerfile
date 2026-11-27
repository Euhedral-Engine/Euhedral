# syntax=docker/dockerfile:1
# The Rust server image: the linux-x64 server bundle in the same distroless base as the Java image (../Dockerfile).
# Build from the repository root:  docker build -f rust/Dockerfile -t euhedral-rust .
FROM ubuntu:24.04 AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# The pinned Zig builds the native library; the pinned Rust (mise.toml, rust/rust-toolchain.toml) builds the server.
ARG ZIG_SHA256=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
RUN curl -fsSL https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz -o /tmp/zig.tar.xz \
    && printf '%s  %s\n' "$ZIG_SHA256" /tmp/zig.tar.xz | sha256sum -c - \
    && mkdir -p /opt/zig \
    && tar -xJf /tmp/zig.tar.xz --strip-components=1 -C /opt/zig \
    && rm /tmp/zig.tar.xz
ENV ZIG=/opt/zig/zig
ARG RUSTUP_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
# The Rust version and components come from rust/rust-toolchain.toml; the Zig pin must match mise.toml.
COPY rust/rust-toolchain.toml /tmp/rust-toolchain.toml
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:/opt/zig:$PATH
RUN curl -fsSL https://static.rust-lang.org/rustup/archive/1.29.0/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init \
    && printf '%s  %s\n' "$RUSTUP_SHA256" /tmp/rustup-init | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "$(awk -F'"' '/^channel/ {print $2}' /tmp/rust-toolchain.toml)" \
        --component rustfmt --component clippy \
    && rm /tmp/rustup-init /tmp/rust-toolchain.toml
COPY . .
# build/native/linux-x64 (the CUDA library, runtime, NVRTC, kernel sources and headers), then the bundle from it and
# the health probe.
RUN cd rust && cargo xtask native --product linux-x64 --force-download \
    && cargo xtask bundle --product linux-x64 \
    && cargo build --release --bin euhedral-health-probe

FROM gcr.io/distroless/cc-debian13:nonroot
WORKDIR /opt/euhedral
# bin/euhedral-server with lib/, share/ and cuda/ beside it: the server finds its CUDA library there. The NVIDIA
# Container Toolkit provides the driver.
COPY --from=build /src/build/bundle/euhedral-server-linux-x64/ /opt/euhedral/
COPY --from=build /src/rust/target/release/euhedral-health-probe /opt/euhedral/bin/euhedral-health-probe
# Default settings; override with --config <mounted-path> or EUHEDRAL_CONFIG.
COPY container/config.yaml /etc/euhedral/config.yaml
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    EUHEDRAL_CONFIG=/etc/euhedral/config.yaml
# Docker loads .env files with docker run --env-file .env (before the image name).
# The container needs io_uring allowed by seccomp: run it with --security-opt seccomp=unconfined or a profile that allows it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/opt/euhedral/bin/euhedral-health-probe"]
ENTRYPOINT ["/opt/euhedral/bin/euhedral-server"]
