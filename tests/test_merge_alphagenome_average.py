#!/usr/bin/env python3
"""
The two-score contract: an AlphaGenome-free ensemble for every variant class,
plus a post-hoc "MERGE + AlphaGenome average" for splicing and non-coding only.

What has to hold, and why each one is here rather than assumed:

1. No shipped ensemble panel contains an alphagenome_* column. This is the whole
   point of the change; if a bundle regressed, every downstream licence claim
   goes with it. load_bundles() refuses such a bundle outright, so the check is
   that the refusal works AND that the real bundles pass it.
2. merge_alphagenome_average == 0.5*ensemble + 0.5*AlphaGenome percentile, to the
   rounding, and that percentile comes from the label-free reference table --
   no model fitted on AlphaGenome output is involved.
3. Coding and non-SNV do NOT offer the average, and say so with a status of
   their own rather than an error or a silently missing key. Their best single
   AlphaGenome feature scores AUROC 0.571 / 0.540 held out — near enough to
   chance that averaging it in would move the score without adding evidence.
4. When AlphaGenome is absent on a task that does offer the average, the result
   is None with a status that says so — never the ensemble score wearing the
   average's name, and never a median-imputed AlphaGenome value dressed up as a
   measurement.
5. AlphaGenome extraction still works, for every task. Dropping it from the
   ensembles must not drop it from the output.
6. The average gets its own cut-points. Scoring it against the ensemble's bands
   is the specific mistake this test exists to catch.

Run: python tests/test_merge_alphagenome_average.py
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from merge_cli import local_engine as le  # noqa: E402

ep = le._load_ep_module()

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


TASKS = ["coding", "splice", "noncoding", "nonsnv"]
AVERAGE_TASKS = ["splice", "noncoding"]        # offer the average
ENSEMBLE_ONLY_TASKS = ["coding", "nonsnv"]     # ensemble score alone

# A variant with every panel feature present, so nothing is imputed and the
# arithmetic below is checkable. Values are plausible, not real.
TRANSCRIPTS = [{"AlphaMissense": {"score": "0.0424;0.0403"}, "ESM1b": {"score": "-2.4843"}}]
PRED_BASE = {
    "input": {"chr": "17", "pos": 43092919, "ref": "G", "alt": "A"},
    "carbon": {"score": 12.5},
    "gpn_msa": {"score": -1.2},
    "hyenadna": {"score": -0.42},
    "nt": {"score": 0.31},
    "evo2_7b": {"llr_score": -0.004},
    "evo2_7b_base": {"llr_score": -0.003},
    "evo2_1b_base": {"llr_score": -0.002},
    "generator": {"score": 0.19},
    "generator_v2": {"score": 0.21},
    "nt_v2": {"score": 0.28},
    "enformer": {f"enformer_{k}": v for k, v in {
        "absmean": 0.02, "mean": 0.001, "std": 0.011, "max_abs": 0.08,
        "top5_absmean": 0.031, "pos_sum": 0.4, "neg_sum": -0.38}.items()},
}
AG_STATS = {
    "raw_score_max": 0.21, "raw_score_mean": 0.05, "raw_score_min": -0.11,
    "quantile_score_max": 0.9971, "quantile_score_min": -0.9989,
    "quantile_score_mean": 0.10, "alphagenome_splicing": 2.4,
}
PRED_WITH_AG = dict(PRED_BASE, alphagenome={"statistics": AG_STATS})


print("=== 1. no shipped ensemble panel uses an AlphaGenome feature ===")
bundles = ep.load_bundles()
check("all four ensembles loaded", sorted(bundles) == sorted(TASKS), f"— {sorted(bundles)}")
for task in TASKS:
    feats = bundles[task]["features"]
    leaked = [f for f in feats if f.lower().startswith("alphagenome")]
    check(f"{task}: panel is AlphaGenome-free", not leaked, f"— {leaked or len(feats)} features")

print("\n=== 2. the guard actually refuses a leaked bundle ===")
# Loading is memoised, so drive the guard on a hand-built bundle rather than
# trusting that a real regression would be caught by the same code path.
import joblib  # noqa: E402
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    good = joblib.load(le.get_bundled_model_path("MERGE_noAlphaGenome_coding.pkl"))
    bad = dict(good, features=list(good["features"]) + ["alphagenome_raw_score_max"])
    joblib.dump(bad, os.path.join(td, "MERGE_noAlphaGenome_coding.pkl"), compress=3)
    saved_dir, saved_b, saved_e = ep.MODEL_DIR, dict(ep._bundles), dict(ep._load_errors)
    try:
        ep.MODEL_DIR = td
        ep._bundles.clear()
        ep._load_errors.clear()
        ep.load_bundles()
        rejected = "coding" not in ep._bundles
        reason = str(ep._load_errors.get("coding", ""))
        check("a panel with an alphagenome_* column is rejected", rejected)
        check("the rejection says why", "not AlphaGenome-free" in reason,
              f"— {reason.strip().splitlines()[-1][:80] if reason else 'no message'}")
    finally:
        ep.MODEL_DIR = saved_dir
        ep._bundles.clear()
        ep._bundles.update(saved_b)
        ep._load_errors.clear()
        ep._load_errors.update(saved_e)

print("\n=== 3. only splicing and non-coding offer the average ===")
check("AG_AVERAGE_TASKS is exactly splice + noncoding",
      sorted(ep.AG_AVERAGE_TASKS) == sorted(AVERAGE_TASKS), f"— {ep.AG_AVERAGE_TASKS}")
check("the AlphaGenome reference is loaded for exactly those two",
      sorted(ep.load_ag_bundles()) == sorted(AVERAGE_TASKS),
      f"— {sorted(ep.load_ag_bundles())}")

print("\n=== 4. the average is exactly the 1:1 average, where it is offered ===")
res, err = ep.run_task("splice", PRED_WITH_AG, TRANSCRIPTS, skip_shap=True)
r = res[next(iter(res))]
check("scoring succeeded", err is None, f"— {err}")
check("no feature was imputed", not r["missing_features"], f"— {r['missing_features']}")
check("the average is present", r["merge_alphagenome_average"] is not None)
check("status is ok", r["merge_alphagenome_average_status"] == "ok",
      f"— {r['merge_alphagenome_average_status']}")
expected = round(0.5 * r["score"] + 0.5 * r["alphagenome_component"], 4)
check("average == 0.5*score + 0.5*component",
      abs(r["merge_alphagenome_average"] - expected) <= 1e-4,
      f"— {r['merge_alphagenome_average']} vs {expected}")
check("the ensemble is flagged AlphaGenome-free in the result",
      r["ensemble_uses_alphagenome"] is False)
check("the AlphaGenome feature is named",
      r["alphagenome_feature"] == "alphagenome_splicing", f"— {r['alphagenome_feature']}")
check("no near-chance caveat on a task that kept the average",
      not r.get("merge_alphagenome_average_caveat"),
      f"— {r.get('merge_alphagenome_average_caveat')}")

direct = ep.compute_merge_alphagenome_average("splice", r["score"], {"alphagenome_splicing": 2.4})
ref = ep.load_ag_bundles()["splice"]
expect_comp = round(float(np.interp(2.4, ref["scores"], ref["percentiles"])), 4)
check("the component is a label-free percentile, not a fitted model",
      str(direct.get("alphagenome_component_kind", "")).startswith("label-free percentile"),
      f"— {direct.get('alphagenome_component_kind')}")
check("the component is the reference-table percentile of the AlphaGenome value",
      direct["alphagenome_component"] == expect_comp == r["alphagenome_component"],
      f"— {direct['alphagenome_component']} / {r['alphagenome_component']} vs {expect_comp}")

print("\n=== 5. coding and non-SNV report the ensemble score alone ===")
for task in ENSEMBLE_ONLY_TASKS:
    out, e = ep.run_task(task, PRED_WITH_AG, TRANSCRIPTS, skip_shap=True)
    row = out[next(iter(out))]
    check(f"{task}: the ensemble still scores", e is None and row.get("score") is not None,
          f"— {row.get('score')}")
    check(f"{task}: the average is None even with AlphaGenome present",
          row["merge_alphagenome_average"] is None, f"— {row['merge_alphagenome_average']}")
    check(f"{task}: the status is the deliberate one, not an error",
          row["merge_alphagenome_average_status"] == "not_offered_for_this_variant_type",
          f"— {row['merge_alphagenome_average_status']}")
    check(f"{task}: the key is present rather than absent",
          "merge_alphagenome_average" in row)
    check(f"{task}: no interpretation for a score that is not offered",
          row["merge_alphagenome_average_interpretation"] is None)
    check(f"{task}: the reason mentions splicing and non-coding",
          "splicing" in (row.get("merge_alphagenome_average_detail") or ""),
          f"— {row.get('merge_alphagenome_average_detail')}")

print("\n=== 6. without AlphaGenome: no average, and a reason ===")
res2, err2 = ep.run_task("splice", PRED_BASE, TRANSCRIPTS, skip_shap=True)
r2 = res2[next(iter(res2))]
check("the ensemble still scores", r2["score"] is not None, f"— {r2['score']}")
check("the ensemble score is unchanged by AlphaGenome's absence",
      r2["score"] == r["score"], f"— {r2['score']} vs {r['score']}")
check("the average is None", r2["merge_alphagenome_average"] is None,
      f"— {r2['merge_alphagenome_average']}")
check("status names the cause",
      r2["merge_alphagenome_average_status"] == "alphagenome_missing",
      f"— {r2['merge_alphagenome_average_status']}")
check("no interpretation is offered for a score that does not exist",
      r2["merge_alphagenome_average_interpretation"] is None)

print("\n=== 7. AlphaGenome extraction still runs ===")
feats = ep.extract_features(PRED_WITH_AG, TRANSCRIPTS)
ag_cols = [c for c in feats if c.startswith("alphagenome")]
check("all seven AlphaGenome summaries are extracted", len(ag_cols) == 7, f"— {len(ag_cols)}")
check("their values survive the trip",
      abs(feats["alphagenome_quantile_score_min"] - (-0.9989)) < 1e-9
      and abs(feats["alphagenome_splicing"] - 2.4) < 1e-9)

print("\n=== 8. the average has its own cut-points ===")
for task in AVERAGE_TASKS:
    cuts = ep.merge_alphagenome_average_cuts(task)
    check(f"{task}: cut-points load and descend",
          cuts is not None and len(cuts) == 4 and all(cuts[i] > cuts[i + 1] for i in range(3)),
          f"— {cuts}")
    check(f"{task}: they differ from the ensemble's", cuts != ep.ENSEMBLE_CUTS, f"— {cuts}")

# The concrete failure mode this guards: on splicing the average's top cut is
# 0.6727 and its second is 0.5339, so a number can land in a different band on
# each scale. Reading the average against the ensemble's bands is the mistake.
splice_cuts = ep.merge_alphagenome_average_cuts("splice")
probe = round((splice_cuts[0] + splice_cuts[1]) / 2, 4)
band_ens = ep.interpret_score(probe)["label"]
band_avg = ep.interpret_merge_alphagenome_average(probe, "splice")["label"]
check("the two scales are genuinely different cut-points",
      splice_cuts != ep.ENSEMBLE_CUTS, f"— {splice_cuts} vs {ep.ENSEMBLE_CUTS}")
check("the average's band says which calibration it used",
      "merge_alphagenome_average cut-points"
      in ep.interpret_merge_alphagenome_average(probe, "splice").get("calibration", ""),
      f"— probe={probe} ensemble={band_ens} average={band_avg}")

print("\n=== 9. every task scores end to end ===")
for task in TASKS:
    out, e = ep.run_task(task, PRED_WITH_AG, TRANSCRIPTS, skip_shap=True)
    row = out[next(iter(out))]
    wants_average = task in AVERAGE_TASKS
    has_average = row.get("merge_alphagenome_average") is not None
    ok = (e is None and row.get("score") is not None and has_average == wants_average)
    check(f"{task}: ensemble score, average {'offered' if wants_average else 'withheld'}", ok,
          f"— score={row.get('score')} average={row.get('merge_alphagenome_average')} "
          f"({row.get('merge_alphagenome_average_status')})")

print("\n" + "=" * 50)
if failures:
    print(f"FAIL {len(failures)} -> {failures}")
    sys.exit(1)
print(f"PASS all checks")
