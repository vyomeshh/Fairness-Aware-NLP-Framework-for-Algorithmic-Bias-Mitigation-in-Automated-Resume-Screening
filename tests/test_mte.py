import math, unittest, warnings
import numpy as np

from fair_resume.config import MaskConfig
from fair_resume.graph_utils import DependencyGraph, synthetic_graph
from fair_resume.m1_extract import normalize_text
from fair_resume.m2_parse import Parser
from fair_resume.m3_mask import mask_text
from fair_resume.m4_encode import TfidfEncoder
from fair_resume.m5_rank import cosine_scores, rank_candidates
from fair_resume import m6_metrics as M
from fair_resume.pipeline import run_paired
from fair_resume.synthetic import build_dataset

warnings.simplefilter("ignore")
CFG = MaskConfig(top_k=8)
P = Parser("auto")


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.g = synthetic_graph()

    def test_bfs_depth_bounded(self):
        got = {self.g.tokens[n] for n, _ in self.g.bfs([0], max_depth=1)}
        self.assertEqual(got, {"Alex", "built"})            # only direct neighbour

    def test_bounded_neighborhood_excludes_seeds(self):
        ctx = [self.g.tokens[i] for i in self.g.bounded_neighborhood([5], 1)]
        self.assertEqual(ctx, ["in"])

    def test_bfs_dfs_cover_same_component(self):
        self.assertEqual({n for n, _ in self.g.bfs([0])}, set(self.g.dfs([0])))
        self.assertEqual(len(self.g.dfs([0])), len(self.g))

    def test_cycle_safe(self):
        g = DependencyGraph(list("abc"), [(0, 1, "x"), (1, 2, "x"), (2, 0, "x"), (0, 1, "dup")])
        self.assertEqual(len(g.dfs([0])), 3)
        self.assertEqual(len(g.bfs([0])), 3)

    def test_bad_edge_rejected(self):
        with self.assertRaises(ValueError):
            DependencyGraph(["a"], [(0, 5, "x")])


class MaskTests(unittest.TestCase):
    S = "Alex built Python APIs in Delhi and she led testing."

    def test_ipr_example(self):
        m, log = mask_text(P.parse(self.S), CFG)
        self.assertEqual(m, "[PERSON] built Python APIs in [LOCATION] and [PRONOUN] led testing.")
        self.assertEqual([e["label"] for e in log], ["PERSON", "LOCATION", "PRONOUN"])

    def test_job_terms_preserved(self):
        m, _ = mask_text(P.parse(self.S), CFG)
        for w in ("Python", "APIs", "testing", "built", "led"):
            self.assertIn(w, m)

    def test_audit_offsets_point_to_original(self):
        _, log = mask_text(P.parse(self.S), CFG)
        for e in log:
            self.assertEqual(self.S[e["start"]:e["end"]], e["text"])

    def test_category_selection(self):
        cfg = MaskConfig(categories=("PERSON",))
        m, _ = mask_text(P.parse(self.S), cfg)
        self.assertIn("Delhi", m); self.assertIn("she", m); self.assertTrue(m.startswith("[PERSON]"))

    def test_neutral_they_not_masked(self):
        m, _ = mask_text(P.parse("They built Python APIs."), CFG)
        self.assertEqual(m, "They built Python APIs.")

    def test_idempotent(self):
        m, _ = mask_text(P.parse(self.S), CFG)
        m2, log2 = mask_text(P.parse(m), CFG)
        self.assertEqual(m, m2); self.assertEqual(log2, [])

    def test_variants_identical_after_masking(self):
        if P.backend != "fallback":
            self.skipTest("statistical NER: invariance rate is reported by the demo, not asserted")
        resumes, *_ = build_dataset()
        seen = {}
        for rid, t in resumes.items():
            seen.setdefault(rid.rsplit("_", 1)[0], set()).add(mask_text(P.parse(t), CFG)[0])
        self.assertTrue(all(len(v) == 1 for v in seen.values()))


class RankTests(unittest.TestCase):
    def test_cosine_known_values(self):
        s = cosine_scores(np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), np.array([1, 0]))
        np.testing.assert_allclose(s, [1, 0, 1 / math.sqrt(2), 0])

    def test_descending_order(self):
        self.assertEqual(rank_candidates(["a", "b", "c"], [0.1, 0.9, 0.5]), ["b", "c", "a"])

    def test_tiebreak_seeded_reproducible_and_not_id_order(self):
        ids = [f"c{i}" for i in range(20)]
        r1 = rank_candidates(ids, [1.0] * 20, seed=42)
        self.assertEqual(r1, rank_candidates(ids, [1.0] * 20, seed=42))
        self.assertNotEqual(r1, ids)


class MetricTests(unittest.TestCase):
    sel = [1, 1, 0, 0, 1, 0, 0, 0]
    grp = [1, 1, 1, 1, 0, 0, 0, 0]       # Z=1 rate .5 ; Z=0 rate .25

    def test_di_dpd(self):
        self.assertAlmostEqual(M.disparate_impact(self.sel, self.grp), 0.5)
        self.assertAlmostEqual(M.demographic_parity_difference(self.sel, self.grp), -0.25)

    def test_di_undefined_when_reference_rate_zero(self):
        self.assertTrue(math.isnan(M.disparate_impact([0, 0, 1, 1], [1, 1, 0, 0])))

    def test_parity_gives_di_one_dpd_zero(self):
        self.assertAlmostEqual(M.disparate_impact([1, 0, 1, 0], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(M.demographic_parity_difference([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)

    def test_mrr(self):
        rk = {"q1": ["a", "b", "c", "d"], "q2": ["x", "y", "z", "r"]}
        rel = {"q1": {"a"}, "q2": {"r"}}                  # ranks 1 and 4
        self.assertAlmostEqual(M.mean_reciprocal_rank(rk, rel), (1 + 0.25) / 2)

    def test_mrr_no_relevant_retrieved_counts_zero(self):
        self.assertAlmostEqual(M.mean_reciprocal_rank({"q": ["a"]}, {"q": {"z"}}), 0.0)

    def test_recall_at_k(self):
        rk = {"q": ["a", "x", "c", "y", "b", "d"]}
        self.assertAlmostEqual(M.recall_at_k(rk, {"q": {"a", "b", "c", "d"}}, 3), 0.5)

    def test_bootstrap_ci_contains_mean_and_reproducible(self):
        d = [0.1, 0.2, 0.15, 0.05, 0.12]
        m, (lo, hi) = M.paired_bootstrap_ci(d, seed=1)
        self.assertLessEqual(lo, m); self.assertGreaterEqual(hi, m)
        self.assertEqual((m, (lo, hi)), M.paired_bootstrap_ci(d, seed=1))


class M1Tests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_text("Pyth-\non   dev\u00a0\n\n\nSQL"), "Python dev\nSQL")


class PipelineTests(unittest.TestCase):
    def test_paired_run_same_cases_and_deterministic(self):
        data = build_dataset()
        r1 = run_paired(*[data[0], data[1], data[2], data[3]], P, TfidfEncoder(), CFG)
        r2 = run_paired(*[data[0], data[1], data[2], data[3]], P, TfidfEncoder(), CFG)
        self.assertEqual(r1["rankings"], r2["rankings"])
        self.assertEqual(set(r1["rankings"]["A"]), set(r1["rankings"]["B"]))
        for jd in r1["rankings"]["A"]:
            self.assertEqual(sorted(r1["rankings"]["A"][jd]), sorted(r1["rankings"]["B"][jd]))
        for c in "AB":
            self.assertTrue(0 <= r1["metrics"][c]["MRR"] <= 1)

    def test_masking_removes_score_gap_between_variants(self):
        if P.backend != "fallback":
            self.skipTest("depends on NER coverage")
        resumes, jds, groups, relevant = build_dataset()
        r = run_paired(resumes, jds, groups, relevant, P, TfidfEncoder(), CFG)
        sc = r["scores"]["B"]["JD_software"]
        for pid in ("sw1", "sw2", "sw3"):
            vals = [v for k, v in sc.items() if k.startswith(pid + "_")]
            self.assertAlmostEqual(max(vals) - min(vals), 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
