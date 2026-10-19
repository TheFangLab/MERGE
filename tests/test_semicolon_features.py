"""Regression guard for the dbNSFP ';'-separated multi-transcript columns.

Offline (no network, no pkl): exercises ensemble_predict._f / build_feature_values
directly.

The bug: dbNSFP returns per-transcript columns as one ';'-joined string with '.'
for absent positions. `_f()` fed that to float(), got ValueError -> NaN, so the
pipeline median-imputed AlphaMissense=0.7536 and ESM1b=-10.2183 (both on the
pathogenic side of the training distribution). BRCA1 rs799917 — real
AlphaMissense 0.042, ClinVar Benign reviewed by expert panel — came out at
MERGE_Coding 0.7924 "Possibly Pathogenic". After the fix: 0.0079 "Likely Benign".

Run:  ./.venv/bin/python tests/test_semicolon_features.py
"""
import importlib.util
import os

import numpy as np

_EP = os.path.join(os.path.dirname(__file__), os.pardir,
                   "merge_cli", "data", "models", "ensemble_predict.py")
_spec = importlib.util.spec_from_file_location("ep_test", os.path.abspath(_EP))
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

# Verbatim from the merge.fanglab.cn response for chr17:43092919 G>A (17 transcripts).
AM = ".;.;0.0424;.;.;.;0.0403;.;.;.;.;.;.;.;.;.;0.042"
ESM = ("-2.6517136;-2.608526;-2.4174263;-2.4400437;-2.608526;-2.4174263;"
       "-2.4831018;-2.5583923;-2.4712148;-2.3819222;.;-2.4174263;-2.4174263;"
       "-2.4174263;-2.4172878;-2.4831114;-2.5583923")

# Training-set medians = what a NaN silently becomes. A parse failure must never
# leave the feature this far from the truth again.
MEDIAN_AM, MEDIAN_ESM = 0.7536, -10.2183

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def _explode(item):
    """The website's _parse_transcripts_backend (views.py:932), condensed:
    positional split on ';', transcript count taken from genename."""
    ann = item["annotations"]
    n = len(ann["genename"].split(";")) if ";" in ann["genename"] else 1
    out = []
    for i in range(n):
        t = {"annotations": {}, "dl_models": {}}
        for mn, md in item["dl_models"].items():
            t["dl_models"][mn] = {
                k: (v.split(";")[i] if isinstance(v, str) and ";" in v else v)
                for k, v in md.items()}
        out.append(t)
    return out


print("=== 1. _f parses ';'-joined dbNSFP columns as mean of valid positions ===")
check("AlphaMissense", np.isclose(ep._f(AM), np.mean([0.0424, 0.0403, 0.042])),
      f"-> {ep._f(AM):.7f}")
check("ESM1b", np.isclose(ep._f(ESM),
                          np.mean([float(x) for x in ESM.split(";") if x != "."])),
      f"-> {ep._f(ESM):.7f}")
check("not median-imputed (AM)", abs(ep._f(AM) - MEDIAN_AM) > 0.5)
check("not median-imputed (ESM1b)", abs(ep._f(ESM) - MEDIAN_ESM) > 5)

print("=== 2. scalars and sentinels behave exactly as before ===")
for v in (None, ".", "N/A", "", "NA", "-", "abc", ".;.;.", ";", [1, 2]):
    check(f"{v!r} -> NaN", np.isnan(ep._f(v)))
check("'0.5' -> 0.5", ep._f("0.5") == 0.5)
check("0.5 -> 0.5", ep._f(0.5) == 0.5)
check("3 -> 3.0", ep._f(3) == 3.0)
check("whitespace tolerated", ep._f("  0.75  ") == 0.75)

print("=== 3. raw dbNSFP row == website's exploded-transcript path ===")
row = {"annotations": {"genename": ";".join(["BRCA1"] * 17)},
       "dl_models": {"AlphaMissense": {"score": AM}, "ESM1b": {"score": ESM}}}
raw = ep.build_feature_values({}, [row])
exploded = ep.build_feature_values({}, _explode(row))
for k in ("AlphaMissense_score", "ESM1b_score"):
    check(f"{k} identical", np.isclose(raw[k], exploded[k]),
          f"raw={raw[k]:.10f} exploded={exploded[k]:.10f}")
    check(f"{k} not NaN", not np.isnan(raw[k]))

print("=== 4. genuinely absent features still yield NaN (median imputation) ===")
empty = ep.build_feature_values({}, [{"annotations": {}, "dl_models": {}}])
check("AlphaMissense_score NaN", np.isnan(empty["AlphaMissense_score"]))
check("ESM1b_score NaN", np.isnan(empty["ESM1b_score"]))

print("\n" + "=" * 50)
print(f"PASS {4 + 14 + 4 + 2 - len(failures)} / FAIL {len(failures)}"
      + (f" -> {failures}" if failures else ""))
raise SystemExit(1 if failures else 0)
