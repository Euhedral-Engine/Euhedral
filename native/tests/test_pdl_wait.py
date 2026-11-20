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

def check(source, catalog=None, **files):
    with tempfile.TemporaryDirectory() as directory:
        (pathlib.Path(directory) / "k.cuh").write_text(source, encoding="utf-8")
        for name, text in files.items():
            (pathlib.Path(directory) / name).write_text(text, encoding="utf-8")
        return pdl_wait_check.violations(directory, catalog, c_only=())

def kernel(before, params="const float* input, const unsigned long long* position, float* out"):
    return f'extern "C" __global__ void k({params}) {{ {before} euhedral_pdl_begin(); out[0] = 1.0f; }}'

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

    def refused(self, before, params=None):
        source = kernel(before) if params is None else kernel(before, params)
        self.assertIn("before euhedral_pdl_begin()", "\n".join(check(source)), before)

    def test_the_bypasses_before_the_wait_are_refused(self):
        for before in ("float a = *position;", "*out = 1.0f;", "float a = *(input + 1);",
                       "float a = *reinterpret_cast<const float*>(input + 2);",
                       "float a = __ldg(input + 1);", "float a = input[0];", "out[1] = 0.0f;",
                       "const View m{input}; float a = m.data[0];", "const View m(input); float a = m.data[0];",
                       "body(*position, 1);"):
            with self.subTest(before=before):
                self.refused(before)

    def test_an_unbraced_if_around_the_wait_is_refused(self):
        source = 'extern "C" __global__ void k(float* a, int c) { if (c) euhedral_pdl_begin(); a[0] = 1; }'
        self.assertIn("not a top-level statement", "\n".join(check(source)))

    def test_only_const_weight_parameters_may_be_read_early(self):
        const = 'extern "C" __global__ void k(const float* weights, float* out) { float w = weights[0]; euhedral_pdl_begin(); out[0] = w; }'
        writable = const.replace("const float* weights", "float* weights")
        self.assertEqual([], check(const))
        self.assertIn("uses pointer parameter(s) weights", "\n".join(check(writable)))

    def test_arguments_of_the_call_that_waits_are_checked(self):
        helper = "__device__ void body(unsigned long long n, const float* in, float* out) { euhedral_pdl_begin(); out[0] = in[0] + n; }\n"
        early = helper + 'extern "C" __global__ void k(const float* in, const unsigned long long* position, float* out) { body(*position, in, out); }'
        late = helper + 'extern "C" __global__ void k(const float* in, const unsigned long long* position, float* out) { body(1, in, out); }'
        self.assertIn("uses pointer parameter(s) position", "\n".join(check(early)))
        self.assertEqual([], check(late))

    def test_a_helper_is_resolved_inside_the_kernels_own_translation_unit(self):
        waiting = "__device__ void body(float* out) { euhedral_pdl_begin(); out[0] = 1.0f; }\n"
        silent = "__device__ void body(float* out) { out[0] = 1.0f; }\n"
        kernel_source = 'extern "C" __global__ void k(float* out) { body(out); }'
        # An unrelated file holding a waiting helper of the same name does not give the kernel a wait.
        problems = check(silent + kernel_source, **{"other.cuh": waiting})
        self.assertEqual([], problems)  # k never reaches a wait, and nothing registers it
        problems = check(silent + kernel_source, catalog=self.catalog("k"), **{"other.cuh": waiting})
        self.assertIn("never reaches euhedral_pdl_begin()", "\n".join(problems))

    def test_two_matching_helpers_in_one_unit_are_a_failure(self):
        waiting = "__device__ void body(float* out) { euhedral_pdl_begin(); out[0] = 1.0f; }\n"
        problems = check(waiting + waiting + 'extern "C" __global__ void k(float* out) { body(out); }')
        self.assertIn("2 definitions match", "\n".join(problems))

    def test_namespaces_pick_the_helper(self):
        source = ("namespace a { __device__ void body(float* out) { euhedral_pdl_begin(); out[0] = 1.0f; } }\n"
                  "namespace b { __device__ void body(float* out) { out[0] = 2.0f; } }\n"
                  'extern "C" __global__ void k(float* out) { a::body(out); }')
        self.assertEqual([], check(source))

    def catalog(self, *names, flags=None):
        directory = tempfile.mkdtemp(dir=self.scratch)
        path = pathlib.Path(directory) / "catalog.rs"
        entries = "".join(f'    spec(Family::Linear, "{n}", Module::Linear, 0, true),\n' for n in names)
        path.write_text(f"const ALL: &[Spec] = &[\n{entries}];\n", encoding="utf-8")
        return path

    @classmethod
    def setUpClass(cls):
        cls.holder = tempfile.TemporaryDirectory()
        cls.scratch = cls.holder.name

    @classmethod
    def tearDownClass(cls):
        cls.holder.cleanup()

    def test_a_registered_kernel_without_a_wait_is_named(self):
        source = 'extern "C" __global__ void k(float* out) { out[0] = 1.0f; }'
        self.assertIn("k: registered", "\n".join(check(source, catalog=self.catalog("k"))))

    NESTED = r'''
#define INNER(B, M)                                                       \
extern "C" __global__ void euhedral_q##B##_rows##M(float* out) {          \
    %s                                                                    \
    out[0] = 1.0f;                                                        \
}
#define OUTER(B)                                                          \
    INNER(B, 2) INNER(B, 3)
OUTER(4)
OUTER(5)
'''

    def test_a_two_level_two_parameter_macro_names_its_kernels(self):
        names = [f"euhedral_q{b}_rows{m}" for b in (4, 5) for m in (2, 3)]
        waits = self.NESTED % "euhedral_pdl_begin();"
        self.assertEqual([], check(waits, catalog=self.catalog(*names)))
        problems = "\n".join(check(self.NESTED % "", catalog=self.catalog(*names)))
        for name in names:
            self.assertIn(name, problems)

    def test_a_missing_catalog_and_a_miscounted_one_fail(self):
        source = kernel("")
        with self.assertRaises(FileNotFoundError):
            check(source, catalog=pathlib.Path(self.scratch) / "absent.rs")
        odd = pathlib.Path(self.scratch) / "odd.rs"
        odd.write_text('spec(Family::A, "k", Module::B, 0, true),\nspec(Family::A,\n "j", Module::B, 0, true),\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            check(source, catalog=odd)

    def test_dflash_kernels_registered_by_the_host_file_must_wait(self):
        directory = pathlib.Path(self.scratch) / "dflash"
        (directory / "host").mkdir(parents=True, exist_ok=True)
        (directory / "host" / "dflash.c").write_text('static const char* n = "euhedral_dflash_x_bf16";', encoding="utf-8")
        (directory / "k.cu").write_text('extern "C" __global__ void euhedral_dflash_x_bf16(float* o) { o[0] = 1.0f; }', encoding="utf-8")
        self.assertIn("euhedral_dflash_x_bf16: registered", "\n".join(pdl_wait_check.violations(directory, c_only=())))

    def test_non_ascii_sources_are_read_as_utf_8(self):
        self.assertEqual([], check("// caf\u00e9 \u2014 note\n" + ROW_KERNEL_FIXED))

    def test_constructor_reads_are_uses_only_for_types_with_a_device_constructor(self):
        head = "struct Head { const float* p; __device__ Head(const float* q) : p(q) { first = q[0]; } float first; };\n"
        plain = "struct Plain { const float* p; };\n"
        for form in ("const Head h(input);", "const Head h{input};"):
            with self.subTest(form=form):
                self.assertIn(
                    "uses pointer parameter(s) input", "\n".join(check(head + kernel(form))))
        self.assertEqual([], check(plain + kernel("const Plain h{input}; const float* q = input + 1;")))
        self.assertEqual([], check(head + kernel("const float* const q(input + 1);")))

    def test_a_weights_layout_with_a_constructor_stays_exempt(self):
        source = ("struct Layout { __device__ Layout(const unsigned char* w) { v = w[0]; } unsigned char v; };\n"
                  'extern "C" __global__ void k(const unsigned char* weights, float* out) {'
                  " const Layout l(weights); euhedral_pdl_begin(); out[0] = l.v; }")
        self.assertEqual([], check(source))

    def test_a_template_typed_helper_parameter_is_bound_by_position(self):
        helper = ("template <class P> __device__ void body(P a, float* o) { float v = a[0]; euhedral_pdl_begin(); o[0] = v; }\n")
        problems = check(helper + 'extern "C" __global__ void k(const float* input, float* out) { body(input, out); }')
        self.assertIn("uses pointer parameter(s) a", "\n".join(problems))

    def test_trap_and_exit_before_the_wait_are_early_exits(self):
        for before in ("__trap();", 'asm volatile("exit;");', 'asm("trap;");'):
            with self.subTest(before=before):
                self.refused(before)

    def test_array_parameters_are_pointers(self):
        source = ('extern "C" __global__ void k(const float input[], float* out) {'
                  " float a = input[0]; euhedral_pdl_begin(); out[0] = a; }")
        self.assertIn("uses pointer parameter(s) input", "\n".join(check(source)))

    def test_a_relative_root_gives_the_same_answer(self):
        import os
        previous = os.getcwd()
        os.chdir(pdl_wait_check.SOURCES.parent)
        try:
            self.assertEqual([], pdl_wait_check.violations("src", pdl_wait_check.CATALOG))
        finally:
            os.chdir(previous)

    def test_the_c_only_list_applies_only_with_the_qwen_layer_host_file(self):
        self.assertEqual([], check(kernel("")))  # no host/qwen_layer_ops.c under the root: c_only is not applied
        directory = pathlib.Path(self.scratch) / "qwen"
        (directory / "host").mkdir(parents=True, exist_ok=True)
        (directory / "host" / "qwen_layer_ops.c").write_text("", encoding="utf-8")
        (directory / "k.cuh").write_text(kernel(""), encoding="utf-8")
        self.assertIn("euhedral_gdn_control_fp32: registered", "\n".join(pdl_wait_check.violations(directory)))


if __name__ == "__main__":
    unittest.main()
