from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_reference  # noqa: E402

N_CTX, VOCAB, CHUNKS = 8, 6, 3


def write_reference(path: Path, tokens: np.ndarray, logp: np.ndarray) -> None:
    """A llama.cpp logits file: per record a float scale and minimum, then VOCAB 16-bit codes (VOCAB even)."""
    with path.open("wb") as out:
        out.write(b"_logits_")
        np.array([N_CTX, VOCAB, CHUNKS], dtype="<i4").tofile(out)
        tokens.astype("<i4").tofile(out)
        for chunk in range(CHUNKS):
            for row in logp[chunk]:
                minimum = row.min()
                scale = (row.max() - minimum) / 65535
                codes = np.round((row - minimum) / scale).astype("<u2")
                np.array([scale, minimum], dtype="<f4").view("<u2").tofile(out)
                codes.tofile(out)


def write_report(path: Path, forced: np.ndarray, logits: np.ndarray) -> None:
    with path.open("wb") as out:
        np.array([len(forced), VOCAB], dtype="<i4").tofile(out)
        for token, row in zip(forced, logits):
            np.array([token], dtype="<i4").tofile(out)
            (row.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tofile(out)


def bf16(values: np.ndarray) -> np.ndarray:
    return ((values.astype(np.float32).view(np.uint32) >> 16) << 16).view(np.float32).astype(np.float64)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class CompareReferenceTest(unittest.TestCase):
    def test_multi_chunk_reports_match_brute_force(self):
        rng = np.random.default_rng(5)
        first = N_CTX // 2
        records = N_CTX - 1 - first
        tokens = rng.integers(0, VOCAB, (CHUNKS, N_CTX))
        reference = log_softmax(rng.standard_normal((CHUNKS, records, VOCAB)) * 2)
        candidate = rng.standard_normal((CHUNKS, records, VOCAB)) * 2
        with tempfile.TemporaryDirectory() as tmp:
            write_reference(Path(tmp) / "ref.kld", tokens, reference)
            for chunk in range(CHUNKS):
                write_report(Path(tmp) / f"cand-chunk{chunk:03d}.bin", tokens[chunk, first + 1:first + 1 + records],
                             candidate[chunk])
            loaded = compare_reference.Reference(str(Path(tmp) / "ref.kld"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                compare_reference.main([str(Path(tmp) / "ref.kld"), str(Path(tmp) / "cand")])
        # The 16-bit reference codes and the BF16 candidate logits are what the tool sees.
        p = np.stack([loaded.log_probabilities(c, 0, records) for c in range(CHUNKS)])
        q = log_softmax(bf16(candidate))
        targets = tokens[:, first + 1:first + 1 + records]
        index = (np.arange(CHUNKS)[:, None], np.arange(records)[None, :], targets)
        nll, reference_nll = -q[index], -p[index]
        kl = np.where(p > -16, np.exp(p) * (p - q), 0).sum(axis=-1)
        chunk_difference = (nll - reference_nll).mean(axis=1)
        line = out.getvalue().strip().splitlines()[-1].split()
        self.assertAlmostEqual(float(line[1]), nll.mean(), places=4)
        self.assertAlmostEqual(float(line[3]), (nll - reference_nll).mean(), places=4)
        self.assertAlmostEqual(float(line[4]), chunk_difference.std(ddof=1) / np.sqrt(CHUNKS), places=4)
        self.assertAlmostEqual(float(line[5]), kl.mean(), places=5)
        self.assertAlmostEqual(float(line[7]), (p.argmax(-1) == q.argmax(-1)).mean(), places=4)

    def test_the_cuda_path_matches_the_numpy_path(self):
        torch = compare_reference.cuda_torch()
        if torch is None:
            self.skipTest("no CUDA PyTorch")
        rng = np.random.default_rng(7)
        first = N_CTX // 2
        records = N_CTX - 1 - first
        tokens = rng.integers(0, VOCAB, (CHUNKS, N_CTX))
        reference = log_softmax(rng.standard_normal((CHUNKS, records, VOCAB)) * 8)
        with tempfile.TemporaryDirectory() as tmp:
            write_reference(Path(tmp) / "ref.kld", tokens, reference)
            report = Path(tmp) / "cand.bin"
            write_report(report, tokens[0, first + 1:first + 1 + records], rng.standard_normal((records, VOCAB)) * 8)
            loaded = compare_reference.Reference(str(Path(tmp) / "ref.kld"))
            cpu = compare_reference.score(loaded, 0, str(report))
            gpu = compare_reference.score(loaded, 0, str(report), torch)
        for key in cpu:
            np.testing.assert_allclose(gpu[key], cpu[key], rtol=1e-12, atol=1e-12, err_msg=key)

    def test_the_reference_floor_is_left_out_of_the_kl(self):
        # The reference clamps entries 16 nats below its maximum; a candidate with the same head and a lower tail
        # differs from it only below the floor.
        head = np.array([0.0, -1.0, -2.0])
        p = np.concatenate([head, np.full(3, -16.0)])[None, :]
        p = p - np.log(np.exp(p).sum())
        q = np.concatenate([head, np.full(3, -30.0)])[None, :]
        q = q - np.log(np.exp(q).sum())
        kl = compare_reference.kl_divergence(p, q)
        above = p > -16
        expected = (np.exp(p[above]) * (p[above] - q[above])).sum()
        self.assertAlmostEqual(float(kl[0]), expected, places=12)
        self.assertLess(float(kl[0]), 1e-5)

    def test_a_prefix_with_missing_chunks_is_refused(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            write_reference(Path(tmp) / "ref.kld", rng.integers(0, VOCAB, (CHUNKS, N_CTX)),
                            log_softmax(rng.standard_normal((CHUNKS, 3, VOCAB))))
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                compare_reference.main([str(Path(tmp) / "ref.kld"), str(Path(tmp) / "missing")])


if __name__ == "__main__":
    unittest.main()
