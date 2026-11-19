"""Every kernel that waits for its predecessor (euhedral_pdl_begin) does so before it returns and before it
touches anything a predecessor wrote: the rule and its limits are in pdl_wait_check.py. Host only."""
import pathlib
import tempfile
import unittest

import pdl_wait_check

ROW_KERNEL_BEFORE_THE_FIX = r'''
#define ROWS(name, Kv)                                                              \
    extern "C" __global__ __launch_bounds__(96) void name(const float* queryKey,    \
            const unsigned long long* position, float* partial, unsigned int from) { \
        const unsigned row = blockIdx.y;                                            \
        const unsigned length = (unsigned)*position + row + 1u;                     \
        if (length < from) return;                                                  \
        euhedral_pdl_begin();                                                       \
        partial[0] = queryKey[row];                                                 \
    }
ROWS(euhedral_rows_nvfp4, A)
ROWS(euhedral_rows_bf16, B)
'''

ROW_KERNEL_FIXED = ROW_KERNEL_BEFORE_THE_FIX.replace(
    "const unsigned row = blockIdx.y;  ", "euhedral_pdl_begin(); const unsigned row = blockIdx.y;  "
).replace("        euhedral_pdl_begin();                                                       \\\n", "")


def check(source):
    with tempfile.TemporaryDirectory() as directory:
        (pathlib.Path(directory) / "k.cuh").write_text(source)
        return pdl_wait_check.violations(directory)


class PdlWaitTest(unittest.TestCase):
    def test_the_installed_kernels_wait_first(self):
        self.assertEqual([], pdl_wait_check.violations(pdl_wait_check.SOURCES, pdl_wait_check.CATALOG))

    def test_a_row_kernel_that_returns_and_reads_the_position_before_its_wait_is_named(self):
        problems = check(ROW_KERNEL_BEFORE_THE_FIX)
        text = "\n".join(problems)
        self.assertIn("euhedral_rows_nvfp4, euhedral_rows_bf16: returns before", text)
        self.assertIn("uses pointer parameter(s) position before", text)

    def test_the_same_kernel_with_the_wait_first_passes(self):
        self.assertEqual([], check(ROW_KERNEL_FIXED))

    def test_a_wait_under_a_branch_is_refused(self):
        source = ('extern "C" __global__ void k(float* a) { if (threadIdx.x == 0) { euhedral_pdl_begin(); } a[0] = 1; }')
        self.assertIn("not a top-level statement", "\n".join(check(source)))

    def test_address_arithmetic_and_scalars_may_precede_the_wait(self):
        source = ('extern "C" __global__ void k(const float* input, float* out, unsigned int n) {'
                  ' const unsigned row = blockIdx.x; const float* x = input + (unsigned long long)row * n;'
                  ' euhedral_pdl_begin(); out[row] = x[0]; }')
        self.assertEqual([], check(source))

    def test_a_read_through_an_address_derived_before_the_wait_is_refused(self):
        source = ('extern "C" __global__ void k(const float* input, float* out) {'
                  ' const float* x = input + 4; float first = x[0]; euhedral_pdl_begin(); out[0] = first; }')
        self.assertIn("uses pointer parameter(s) x", "\n".join(check(source)))

    def test_a_wait_in_a_helper_is_followed_and_the_call_must_come_first(self):
        helper = 'static __device__ void body(float* a) { euhedral_pdl_begin(); a[0] = 1; }\n'
        self.assertEqual([], check(helper + 'extern "C" __global__ void k(float* a) { body(a); }'))
        late = helper + 'extern "C" __global__ void k(float* a) { float v = a[0]; body(a); }'
        self.assertIn("uses pointer parameter(s) a", "\n".join(check(late)))


if __name__ == "__main__":
    unittest.main()
