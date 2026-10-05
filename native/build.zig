const std = @import("std");

pub fn build(b: *std.Build) void {
    const target_name = b.option([]const u8, "product-target", "Explicit supported CUDA target") orelse
        @panic("-Dproduct-target=x86_64-linux-gnu or x86_64-windows-gnu is required");
    if (!std.mem.eql(u8, target_name, "x86_64-linux-gnu") and
        !std.mem.eql(u8, target_name, "x86_64-windows-gnu"))
        @panic("unsupported CUDA target (only x86_64-linux-gnu and x86_64-windows-gnu)");
    const query = std.Target.Query.parse(.{ .arch_os_abi = target_name }) catch @panic("invalid CUDA target");
    const target = b.resolveTargetQuery(query);
    const optimize = b.standardOptimizeOption(.{});
    const cuda_include_dir = b.option(
        []const u8,
        "cuda-include-dir",
        "CUDA include directory",
    ) orelse @panic("-Dcuda-include-dir is required");
    const cuda_lib_dir = b.option(
        []const u8,
        "cuda-lib-dir",
        "CUDA library directory",
    ) orelse @panic("-Dcuda-lib-dir is required");
    const cuda_driver_lib_dir = b.option(
        []const u8,
        "cuda-driver-lib-dir",
        "CUDA target driver import/stub library directory",
    ) orelse @panic("-Dcuda-driver-lib-dir is required");
    if (!std.fs.path.isAbsolute(cuda_include_dir) or !std.fs.path.isAbsolute(cuda_lib_dir))
        @panic("CUDA include and library paths must be absolute");
    if (!std.fs.path.isAbsolute(cuda_driver_lib_dir))
        @panic("CUDA driver import/stub library path must be absolute");

    const library = b.addLibrary(.{
        .name = "euhedral_cuda",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
        .linkage = .dynamic,
    });
    library.root_module.addCSourceFiles(.{
        .root = b.path("src/host"),
        .files = &.{
            "euhedral_cuda.c", "q3_embedding.c", "cuda_kernel_loader.c", "submission.c",
            "rms_norm_bf16.c", "q3_linear_bf16.c", "q45_linear.c", "qwen_layer_ops.c", "nvfp4_linear.c", "q3_mx.c", "reference.c", "dflash.c", "qwen4_ops.c",
        },
        .flags = &.{"-std=c11", "-fvisibility=hidden"},
    });
    library.root_module.addIncludePath(b.path("include"));
    library.root_module.addIncludePath(.{.cwd_relative = cuda_include_dir});
    library.root_module.addLibraryPath(.{.cwd_relative = cuda_driver_lib_dir});
    library.root_module.addLibraryPath(.{.cwd_relative = cuda_lib_dir});
    if (target.result.os.tag == .windows) {
        library.root_module.linkSystemLibrary("cudart", .{});
        library.root_module.linkSystemLibrary("nvrtc", .{});
        library.root_module.linkSystemLibrary("nvcuda", .{});
    } else {
        library.root_module.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ cuda_lib_dir, "libcudart.so" }) });
        library.root_module.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ cuda_lib_dir, "libnvrtc.so" }) });
        library.root_module.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ cuda_driver_lib_dir, "libcuda.so" }) });
        library.root_module.linkSystemLibrary("pthread", .{});
        library.root_module.linkSystemLibrary("dl", .{});
    }
    if (target.result.os.tag == .windows) {
        const install = b.addInstallFile(library.getEmittedBin(), "lib/euhedral_cuda.dll");
        b.getInstallStep().dependOn(&install.step);
    } else {
        b.installArtifact(library);
    }
    // Every CUDA source is an NVRTC asset, installed at its source-relative path so module roots
    // (<domain>/kernels.cu) and their includes resolve against share/euhedral_cuda.
    b.installDirectory(.{
        .source_dir = b.path("src"),
        .install_dir = .prefix,
        .install_subdir = "share/euhedral_cuda",
        .include_extensions = &.{ ".cu", ".cuh" },
    });
}
