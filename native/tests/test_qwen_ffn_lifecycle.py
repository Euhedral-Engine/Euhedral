"""Fault-inject the actual compound host function; no CUDA device is touched."""
import ctypes
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUBS = r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* CUstream;
#define EUHEDRAL_CUDA_SUCCESS 0
#define EUHEDRAL_CUDA_INVALID_ARGUMENT -1
#define EUHEDRAL_CUDA_FORMAT_MISMATCH -2
#define EUHEDRAL_CUDA_KERNEL_UNAVAILABLE -3
#define cudaStreamNonBlocking 1
#define cudaEventDisableTiming 2
static void *ffn_stream_gate=(void*)10, *ffn_stream_down=(void*)11;
static int fail_at, call_count, asynchronous, pending, fail_drains, next_event;
static int trace_kind[256], trace_handle[256], pending_before[256];
static unsigned char buffers[8 << 20] __attribute__((aligned(16)));
static int step(int kind, void* handle) {
    trace_kind[call_count]=kind;
    trace_handle[call_count]=(int)(uintptr_t)handle;
    pending_before[call_count]=pending;
    call_count++;
    return call_count==fail_at ? 42 : 0;
}
static int euhedral_cuda_bind_thread_context(void) {return 0;}
static int ensure_initialized(void) {return 0;}
static void* euhedral_cuda_submission_stream(void) {return asynchronous ? (void*)1 : NULL;}
static int cudaStreamCreateWithFlags(void** stream, unsigned flags) {
    int result=step(1,NULL); if(!result) *stream=(void*)2; return result;
}
static int cudaEventCreateWithFlags(void** event, unsigned flags) {
    int result=step(2,NULL); if(!result) *event=(void*)(uintptr_t)(++next_event); return result;
}
static int cudaStreamWaitEvent(void* stream, void* event, unsigned flags) {return step(3,stream);}
static int cudaEventRecord(void* event, void* stream) {return step(6,stream);}
static int cuLaunchKernel(void* fn, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, void* stream, void** args, void** extra) {
    int result=step(fn==ffn_stream_gate ? 4 : 5,stream);
    if(!result) pending |= 1 << (int)(uintptr_t)stream;
    return result;
}
static int cudaStreamSynchronize(void* stream) {
    int result=step(7,stream);
    if(!result && fail_drains) result=43;
    if(!result) {
        pending &= ~(1 << (int)(uintptr_t)stream);
        // The final default-stream join includes all consumer work.
        if(stream==NULL) pending=0;
    }
    return result;
}
static int cudaEventDestroy(void* event) {return step(8,event);}
static int cudaStreamDestroy(void* stream) {return step(9,stream);}
'''
WRAPPER = r'''
int probe(int injected_call, int async_mode) {
    fail_at=injected_call; asynchronous=async_mode&1; fail_drains=(async_mode&2)!=0;
    call_count=0; pending=0; next_event=10;
    memset(trace_kind,0,sizeof trace_kind);
    memset(trace_handle,0,sizeof trace_handle);
    memset(pending_before,0,sizeof pending_before);
    uint64_t gg=34816ull*80, dg=5120ull*272;
    uint64_t gb=((gg*24+255)&~255ull)+gg*2, db=((dg*24+255)&~255ull)+dg*2;
    return euhedral_cuda_q3_ffn_streamed_bf16(buffers,buffers,buffers,buffers,buffers,
            (float*)buffers,256,5120,17408,gb,db);
}
int calls(void) {return call_count;}
int kind(int i) {return trace_kind[i];}
int handle(int i) {return trace_handle[i];}
int borrowed(int i) {return pending_before[i];}
int pending_at_return(void) {return pending;}
'''


class StreamedFfnLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which('cc')
        if cc is None:
            raise unittest.SkipTest('a host C compiler is required')
        source = (ROOT / 'src/qwen_layer_ops.c').read_text()
        start = source.index('int euhedral_cuda_q3_ffn_streamed_bf16(')
        end = source.index('\nint euhedral_cuda_q3_gate_up_swiglu_bf16(', start)
        cls.directory = tempfile.TemporaryDirectory(prefix='ffn-lifecycle-')
        cls.addClassCleanup(cls.directory.cleanup)
        path = pathlib.Path(cls.directory.name)
        (path / 'probe.c').write_text(STUBS + source[start:end] + WRAPPER)
        subprocess.run([cc, '-std=c11', '-shared', '-fPIC', '-O0', str(path / 'probe.c'),
                        '-o', str(path / 'probe.so')], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(path / 'probe.so'))
        cls.lib.probe.argtypes = [ctypes.c_int, ctypes.c_int]
        for symbol in ['kind', 'handle', 'borrowed']:
            getattr(cls.lib, symbol).argtypes = [ctypes.c_int]

    def trace(self):
        return [(self.lib.kind(i), self.lib.handle(i), self.lib.borrowed(i))
                for i in range(self.lib.calls())]

    def test_success_joins_the_consumer_and_keeps_async_workspace_borrowed(self):
        for asynchronous in [0, 1]:
            with self.subTest(asynchronous=asynchronous):
                self.assertEqual(0, self.lib.probe(0, asynchronous))
                trace = self.trace()
                self.assertEqual(5, sum(kind == 4 for kind, _, _ in trace))
                self.assertEqual(5, sum(kind == 5 for kind, _, _ in trace))
                self.assertEqual(4, sum(kind == 8 for kind, _, _ in trace))
                self.assertEqual(1, sum(kind == 9 for kind, _, _ in trace))
                self.assertEqual(bool(asynchronous), bool(self.lib.pending_at_return()))
                last_submission = [row for row in trace if row[0] in (3, 4, 5, 6)][-1]
                self.assertEqual((3, asynchronous), last_submission[:2])

    def test_every_partial_submission_failure_drains_before_teardown(self):
        for asynchronous in [0, 1]:
            self.assertEqual(0, self.lib.probe(0, asynchronous))
            normal = self.trace()
            for index, (kind, _, _) in enumerate(normal):
                if kind in (8, 9):
                    continue
                with self.subTest(asynchronous=asynchronous, call=index + 1, kind=kind):
                    self.assertEqual(42, self.lib.probe(index + 1, asynchronous))
                    trace = self.trace()
                    self.assertEqual(0, self.lib.pending_at_return())
                    for event, _, pending in trace:
                        if event in (8, 9):
                            self.assertEqual(0, pending, 'teardown while workspace is still borrowed')
                    self.assertIn((7, asynchronous), [(k, h) for k, h, _ in trace])
                    if index > 0:
                        self.assertIn((7, 2), [(k, h) for k, h, _ in trace])

    def test_secondary_drain_failure_returns_error_with_unproven_workspace_ownership(self):
        for asynchronous in [0, 1]:
            self.assertEqual(0, self.lib.probe(0, asynchronous))
            first_down = next(i + 1 for i, row in enumerate(self.trace()) if row[0] == 5)
            self.assertEqual(42, self.lib.probe(first_down, asynchronous | 2))
            self.assertNotEqual(0, self.lib.pending_at_return())
            self.assertIn((7, 2), [(k, h) for k, h, _ in self.trace()])
            self.assertIn((7, asynchronous), [(k, h) for k, h, _ in self.trace()])

    def test_teardown_failures_are_returned_to_the_outer_completion_owner(self):
        for asynchronous in [0, 1]:
            self.assertEqual(0, self.lib.probe(0, asynchronous))
            normal = self.trace()
            for index, (kind, _, _) in enumerate(normal):
                if kind not in (8, 9):
                    continue
                with self.subTest(asynchronous=asynchronous, call=index + 1):
                    self.assertEqual(42, self.lib.probe(index + 1, asynchronous))
                    trace = self.trace()
                    self.assertEqual(4, sum(k == 8 for k, _, _ in trace))
                    self.assertEqual(1, sum(k == 9 for k, _, _ in trace))


if __name__ == '__main__':
    unittest.main()
