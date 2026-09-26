"""Temporal fragment ownership and complete-output differential CUDA tests."""
import ctypes as C
from contextlib import closing
import struct
import random
import unittest
from test_q3_primitives import Gpu, NVRTC, ROOT, P, _check
import test_q3_fragment_nodes as prior_nodes

SHAPES = prior_nodes.SHAPES


class PipelineGpu(Gpu):
    def launch(self, name, grid, arguments, synchronize=True):
        if not name.startswith('euhedral_q3_pipeline_'):
            return super().launch(name, grid, arguments, synchronize)
        function = P()
        _check(self.function(C.byref(function), self.module, name.encode()), name)
        params = (P * len(arguments))(*[C.cast(C.pointer(value), P) for value in arguments])
        _check(self.launch_kernel(function, grid[0], grid[1], 1, 256, 1, 1, 0, None, params, None), name)
        if synchronize:
            _check(self.sync(), name)


def pipeline_source():
    candidate = ROOT / 'native/src/q3/pipeline_kernels.cu'
    # Prior implementation is a compiling RED baseline, not an emulated pipeline.
    return b'#include "q3/pipeline_kernels.cu"\n' if candidate.exists() else b'#include "q3/fragment_kernels.cu"\n'


def controlled_pipeline_source():
    """A diagnostic schedule, not a shipped kernel or a performance measurement.

    Hold generation zero after acquire until both required branches complete real
    generation-one staging. Start that staging only after every local descendant
    has acquired zero. This makes useful overlap a deterministic capability test,
    rather than relying on a timestamp after work that may already have finished.
    """
    header = (ROOT / 'native/src/q3/strategies/pipelined_fragments.cuh').read_text()
    header = header.replace('#include "fragments.cuh"', '#include "q3/strategies/fragments.cuh"\n#include <cuda/atomic>')
    header = header.replace('    const Layout packed(weights, width, scale_offset);', '''
    using ProbeFlag = cuda::atomic_ref<unsigned int, cuda::thread_scope_block>;
    __shared__ unsigned int held[Tile::kWarps], staged[Node::kBranches];
    if (threadIdx.x < Tile::kWarps) held[threadIdx.x] = 0;
    if (threadIdx.x < Node::kBranches) staged[threadIdx.x] = 0;
    const Layout packed(weights, width, scale_offset);''')
    header = header.replace('            node.begin(branch, g, lane);', '''
            if (generation == 1) {
                for (unsigned int w = 0; w < Tile::kWarps; ++w) {
                    const bool borrows = branch < Node::kMBranches
                            ? w / Node::kNBranches == branch
                            : Node::kMBranches + w % Node::kNBranches == branch;
                    if (borrows)
                        while (!ProbeFlag(held[w]).load(cuda::memory_order_acquire)) {}
                }
            }
            node.begin(branch, g, lane);''')
    header = header.replace('            node.publish(branch, g, lane);', '''
            node.publish(branch, g, lane);
            if (generation == 1 && lane == 0)
                ProbeFlag(staged[branch]).store(1, cuda::memory_order_release);''')
    header = header.replace('            consume_fragment_node<Tile>(acc, lease);', '''
            if (generation == 0 && generations > 1) {
                if (lane == 0) ProbeFlag(held[consumer]).store(1, cuda::memory_order_release);
                while (!ProbeFlag(staged[a_branch]).load(cuda::memory_order_acquire)
                        || !ProbeFlag(staged[b_branch]).load(cuda::memory_order_acquire)) {}
            }
            consume_fragment_node<Tile>(acc, lease);''')
    kernels = (ROOT / 'native/src/q3/pipeline_kernels.cu').read_text()
    return (header + kernels.replace('#include "strategies/pipelined_fragments.cuh"', '')).encode()


def run_pipeline(gpu, tile, cm, cn, grid, x, w, rows, width, outputs, offset, observe=True):
    generations = (width + 63) // 64
    words = grid[0] * grid[1] * generations * 44
    y = gpu.zeros(rows * outputs * 2, 0xa5)
    trace = gpu.zeros(words * 8) if observe else 0
    try:
        gpu.launch(f'euhedral_q3_pipeline_{tile}_{cm}x{cn}', grid,
                   [C.c_uint64(x), C.c_uint64(w), C.c_uint64(y), C.c_uint(rows), C.c_uint(width),
                    C.c_uint(outputs), C.c_uint64(offset), C.c_uint64(trace)])
        return gpu.download(y, rows * outputs * 2), gpu.download(trace, words * 8) if observe else b''
    finally:
        if trace:
            gpu.free(trace)
        gpu.free(y)


@unittest.skipIf(NVRTC is None, 'pinned NVRTC unavailable')
class Q3PipelineTest(unittest.TestCase):
    def test_installed_pipeline_compiles_and_matches_source(self):
        product = ROOT / 'build/native/linux-x64/share/euhedral_cuda'
        for relative in ('q3/pipeline_kernels.cu', 'q3/strategies/pipelined_fragments.cuh'):
            self.assertEqual((ROOT / 'native/src' / relative).read_bytes(),
                             (product / relative).read_bytes())
        with closing(PipelineGpu(b'#include "q3/pipeline_kernels.cu"\n',
                                 include_dir=product, cpp_std=17)) as gpu:
            for tile in (32, 64):
                for cm, cn in SHAPES:
                    fn = P()
                    _check(gpu.function(C.byref(fn), gpu.module,
                           f'euhedral_q3_pipeline_{tile}_{cm}x{cn}'.encode()), 'installed symbol')

    def test_steady_state_has_no_broad_barriers(self):
        header = (ROOT / 'native/src/q3/strategies/pipelined_fragments.cuh').read_text()
        self.assertNotIn('__syncthreads', header)
        self.assertEqual(header.count('placement.cluster.sync();'), 2)
        body = header.split('if (warp < Node::kBranches) {', 1)[1].split('// Terminal DSM lifetime join:', 1)[0]
        self.assertNotIn('cluster.sync', body)
        prior = (ROOT / 'native/src/q3/strategies/fragments.cuh').read_text()
        lifecycle = prior.split('struct FragmentNode {', 1)[1].split('// The placement routes', 1)[0]
        self.assertEqual(lifecycle.count('__syncthreads();'), 5)
        loop = prior.split('for (unsigned int base = 0, generation = 0;', 1)[1].split('aggregate_output', 1)[0]
        self.assertEqual(loop.count('placement.cluster.sync();'), 2)
        hierarchy = (ROOT / 'native/src/q3/strategies/hierarchical.cuh').read_text()
        parent = hierarchy.split('void stage_owned_k(', 1)[1].split('// Keep the DSM', 1)[0]
        self.assertEqual(parent.count('__syncthreads();'), 1)  # sixth CTA barrier in PR #10

    def test_fragment_storage_has_independent_generation_slots(self):
        source = b'''#include "q3/strategies/pipelined_fragments.cuh"
extern "C" __global__ void probe_slots(unsigned int* out) {
    if (threadIdx.x == 0) *out = q3::PipelinedFragmentNode<q3::Prefill32>::kSlots;
}
'''
        with closing(Gpu(source, cpp_std=17)) as gpu:
            out = gpu.zeros(4)
            try:
                gpu.launch('probe_slots', 1, [C.c_uint64(out)])
                slots, = struct.unpack('<I', gpu.download(out, 4))
                self.assertGreaterEqual(slots, 2, 'a live generation still monopolizes fragment storage')
            finally:
                gpu.free(out)


    def test_held_borrower_prevents_reuse_but_not_other_slots_or_branches(self):
        probe = r'''
#include <cuda/atomic>
using Flag = cuda::atomic_ref<unsigned int, cuda::thread_scope_block>;
extern "C" __global__ void probe_held_borrower(unsigned int* out) {
    using Node = q3::PipelinedFragmentNode<q3::Prefill32>;
    __shared__ q3::FragmentSlot states[Node::kBranches * Node::kSlots];
    __shared__ unsigned int payload[2], flags[4];
    const Node node{states, nullptr, nullptr, nullptr};
    unsigned int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) { node.initialize(0, 2); node.initialize(1, 1); }
    if (threadIdx.x < 4) flags[threadIdx.x] = 0;
    __syncthreads();
    q3::KGeneration g0{0,0}, g1{1,1}, g2{2,0};
    if (warp == 0) {
        node.begin(0, g0, lane);
        if (lane == 0) payload[0] = 11;
        node.publish(0, g0, lane);
        while (!Flag(flags[0]).load(cuda::memory_order_acquire)
                || !Flag(flags[1]).load(cuda::memory_order_acquire)) {}
        node.begin(0, g1, lane);
        if (lane == 0) payload[1] = 22;
        node.publish(0, g1, lane);
        node.begin(0, g2, lane); // must not return until BOTH borrowers release g0
        if (lane == 0) { payload[0] = 33; out[8] = states[0].generation; }
    } else if (warp == 1 || warp == 2) {
        node.acquire(0, {0,0});
        if (lane == 0) Flag(flags[warp-1]).store(1, cuda::memory_order_release);
        node.acquire(0, {1,1}); // hold g0 while acquiring independently published g1
        if (warp == 1) {
            node.release(0, g0, lane);
            if (lane == 0) Flag(flags[2]).store(1, cuda::memory_order_release);
        } else {
            while (!Flag(flags[2]).load(cuda::memory_order_acquire)
                    || !Flag(flags[3]).load(cuda::memory_order_acquire)) {}
            if (lane == 0) {
                out[0] = payload[0]; out[1] = payload[1];
                out[2] = states[0].generation; out[3] = states[1].generation;
                out[4] = !cuda::ptx::mbarrier_test_wait_parity(cuda::ptx::sem_acquire,
                        cuda::ptx::scope_cluster, &states[0].released, 0);
            }
            node.release(0, g0, lane);
        }
        node.release(0, g1, lane);
    } else {
        while (!Flag(flags[0]).load(cuda::memory_order_acquire)
                || !Flag(flags[1]).load(cuda::memory_order_acquire)) {}
        // A sibling branch cycles repeatedly while branch 0 retains generation 0.
        for (unsigned int generation = 0; generation < 8; ++generation) {
            q3::KGeneration g{generation, generation % 2};
            node.begin(1, g, lane); node.publish(1, g, lane);
            node.acquire(1, {g.value,g.slot}); node.release(1, g, lane);
        }
        if (lane == 0) { out[5] = 8; Flag(flags[3]).store(1, cuda::memory_order_release); }
    }
    __syncthreads();
    if (threadIdx.x == 0) { out[6] = payload[0]; out[7] = payload[1]; }
}
'''
        header = (ROOT / 'native/src/q3/strategies/pipelined_fragments.cuh').read_text()
        header = header.replace('#include "fragments.cuh"', '#include "q3/strategies/fragments.cuh"')
        for bypass_reuse in (True, False):
            source = header.replace('if (g.value >= kSlots)', 'if (false)') if bypass_reuse else header
            with closing(Gpu((source+probe).encode(), cpp_std=17)) as gpu:
                out = gpu.zeros(9*4)
                try:
                    gpu.launch('probe_held_borrower', 1, [C.c_uint64(out)])
                    actual = struct.unpack('<9I', gpu.download(out, 9*4))
                    expected = (11,22,0,1,1,8,33,22,2)
                    if bypass_reuse:
                        self.assertNotEqual(actual, expected, 'reuse-bypass mutation escaped the regression')
                    else:
                        self.assertEqual(actual, expected)
                finally:
                    gpu.free(out)

    def test_pipeline_matches_all_predecessors_and_generations_overlap(self):
        self.compare_predecessors(pipeline_source())

    def test_useful_production_while_prior_acquired_for_every_composition(self):
        self.compare_predecessors(controlled_pipeline_source(), require_overlap=True)

    def compare_predecessors(self, source, require_overlap=False):
        with closing(PipelineGpu(source, cpp_std=17)) as gpu, \
             closing(Gpu(b'#include "q3/fragment_kernels.cu"\n', cpp_std=17)) as nodes, \
             closing(Gpu(b'#include "q3/hierarchical_kernels.cu"\n', cpp_std=17)) as direct, \
             closing(Gpu(b'#include "q3/cluster_kernels.cu"\n', cpp_std=17)) as replicated, \
             closing(Gpu(b'#include "q3/kernels.cu"\n')) as local:
            rng = random.Random(0x710E)
            for tile in (32, 64):
                for rows, width, outputs in ((1, 1, 1), (tile+1, 65, 35),
                                               (tile+3, 321, 67), (tile*4, 1024, 128)):
                    groups = ((width + 127)//128)*2
                    offset = (outputs * groups * 24 + 255) & ~255
                    packed = bytearray(rng.randbytes(offset + outputs*groups*2))
                    for i in range(outputs*groups):
                        struct.pack_into('<H', packed, offset+2*i, (0, 1, 0x03ff, 0x3555, 0xb555)[i%5])
                    with prior_nodes.Q3FragmentNodeTest.buffers(gpu, rows, width, packed) as (x, w):
                        reference = prior_nodes.Q3FragmentNodeTest.run_one(local,
                            'euhedral_q3_prefill' + ('_64' if tile == 64 else ''),
                            ((rows+tile-1)//tile)*((outputs+31)//32), x, w, rows, width, outputs, offset)
                        for cm, cn in SHAPES:
                            grid = (((outputs+32*cn-1)//(32*cn))*cn,
                                    ((rows+tile*cm-1)//(tile*cm))*cm)
                            with self.subTest(tile=tile, rows=rows, width=width, outputs=outputs, cm=cm, cn=cn):
                                actual, trace = run_pipeline(gpu, tile, cm, cn, grid, x, w, rows, width, outputs, offset)
                                self.assertEqual(actual, reference)
                                self.assertNotIn(b'\xa5\xa5', [actual[i:i+2] for i in range(0,len(actual),2)])
                                for prior, family in ((direct,'hierarchical'), (replicated,'cluster')):
                                    self.assertEqual(actual, prior_nodes.Q3FragmentNodeTest.run_one(prior,
                                        f'euhedral_q3_{family}_{tile}_{cm}x{cn}', grid,
                                        x, w, rows, width, outputs, offset))
                                old, _ = prior_nodes.Q3FragmentNodeTest.run_node(nodes, tile, cm, cn, grid,
                                                                    x, w, rows, width, outputs, offset)
                                self.assertEqual(actual, old)
                                plain, _ = run_pipeline(gpu, tile, cm, cn, grid, x, w, rows, width, outputs, offset, False)
                                self.assertEqual(plain, actual)
                                overlap_count = self.check_trace(trace, grid, tile, cm, cn, width)
                                if require_overlap and width > 64:
                                    self.assertGreater(overlap_count, 0,
                                        'no complete N+1 production interval inside N acquired lifetime')

    def check_trace(self, trace, grid, tile, cm, cn, width):
        values = struct.unpack(f'<{len(trace)//8}Q', trace)
        generations = (width+63)//64
        overlap = 0
        for cta in range(grid[0]*grid[1]):
            events = [values[(cta*generations+g)*44:(cta*generations+g+1)*44] for g in range(generations)]
            for g, event in enumerate(events):
                for branch in range(4):
                    begin, accepted, published, fetched, identity, slot = event[branch*6:branch*6+6]
                    self.assertTrue(0 < begin <= accepted <= published)
                    self.assertEqual((identity,slot), (g,g%2))
                    remote = (cta % grid[0]) % cn != 0 if branch < 2 else (cta // grid[0]) % cm != 0
                    self.assertEqual(fetched, (tile//32 if branch < 2 else 1) if remote else 0)
                    borrowers = [w for w in range(4) if (w//2 if branch < 2 else 2+w%2) == branch]
                    if g >= 2:
                        for w in borrowers:
                            self.assertGreaterEqual(begin, events[g-2][24+w*5+2])
                    if g:
                        for w in borrowers:
                            acquired, done, released = events[g-1][24+w*5:24+w*5+3]
                            if acquired < begin < accepted < done <= released:
                                overlap += 1
                for w in range(4):
                    acquired, done, released, a, b = event[24+w*5:24+w*5+5]
                    self.assertTrue(0 < acquired <= done <= released)
                    self.assertLessEqual(event[(w//2)*6+2], acquired)
                    self.assertLessEqual(event[(2+w%2)*6+2], acquired)
                    self.assertNotEqual(a, 0)
                    self.assertNotEqual(b, 0)
                    self.assertEqual(a, event[24+(w^1)*5+3])
                    self.assertEqual(b, event[24+(w^2)*5+4])
                    if g:
                        self.assertNotEqual(a, events[g-1][24+w*5+3])
                        self.assertNotEqual(b, events[g-1][24+w*5+4])
        return overlap


if __name__ == '__main__':
    unittest.main()
