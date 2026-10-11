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
        kl = (np.exp(p) * (p - q)).sum(axis=-1)
        chunk_difference = (nll - reference_nll).mean(axis=1)
        line = out.getvalue().strip().splitlines()[-1].split()
        self.assertAlmostEqual(float(line[1]), nll.mean(), places=4)
        self.assertAlmostEqual(float(line[3]), (nll - reference_nll).mean(), places=4)
        self.assertAlmostEqual(float(line[4]), chunk_difference.std(ddof=1) / np.sqrt(CHUNKS), places=4)
        self.assertAlmostEqual(float(line[5]), kl.mean(), places=5)
        self.assertAlmostEqual(float(line[7]), (p.argmax(-1) == q.argmax(-1)).mean(), places=4)

    def test_a_prefix_with_missing_chunks_is_refused(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            write_reference(Path(tmp) / "ref.kld", rng.integers(0, VOCAB, (CHUNKS, N_CTX)),
                            log_softmax(rng.standard_normal((CHUNKS, 3, VOCAB))))
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                compare_reference.main([str(Path(tmp) / "ref.kld"), str(Path(tmp) / "missing")])


if __name__ == "__main__":
    unittest.main()
