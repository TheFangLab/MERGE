"""Guard the scikit-learn floor required by the bundled MERGE pkl bundles.

Why this exists
---------------
The four MERGE_*.pkl bundles were pickled under scikit-learn 1.9.0. The coding
bundle is a LogisticRegression pipeline, and by 1.9.0 sklearn had removed
LogisticRegression's `multi_class` attribute — so the pickled estimator simply
does not carry it. predict_proba in sklearn 1.7.2 and earlier still reads
`self.multi_class`, so on those versions the LR panel dies with

    AttributeError: 'LogisticRegression' object has no attribute 'multi_class'

while the other panels keep answering normally. That asymmetry is the dangerous
part: a user on an old sklearn gets working answers for indels and splice
variants and a crash for coding variants, which reads like a data problem rather
than an environment one. (The splicing and non-SNV bundles also need CatBoost and
XGBoost, which this test exercises by loading them.)

Measured on 2026-07-31 against the then-current bundles, when the noncoding
panel was also a LogisticRegression:

    sklearn 1.4.2 / 1.5.2 / 1.6.1 / 1.7.2  ->  coding & noncoding FAIL
    sklearn 1.8.0 / 1.9.0                  ->  all four OK

Hence `scikit-learn>=1.8` in pyproject.toml. This test does not re-run that
matrix (it cannot install other versions); it asserts that whatever sklearn is
installed here can actually load and score all four bundles, which is what the
floor is there to guarantee. It fails loudly if someone lowers the pin, and it
fails on a future sklearn that breaks the pickles again.

Run:  ./.venv/bin/python tests/test_sklearn_compat.py
"""
import json
import warnings

import joblib
import pandas as pd
import sklearn

from merge_cli import local_engine

TASKS = ("coding", "splicing", "noncoding", "nonsnv")
MIN_SKLEARN = (1, 8)


def _version_tuple(v: str) -> tuple:
    """Leading numeric components of a version, e.g. '1.9.0rc1' -> (1, 9)."""
    parts = []
    for chunk in v.split(".")[:2]:
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


_floor = ".".join(str(p) for p in MIN_SKLEARN)
print(f"scikit-learn {sklearn.__version__}  (declared floor: >={_floor})")

check("installed sklearn meets the declared floor",
      _version_tuple(sklearn.__version__) >= MIN_SKLEARN,
      f"— found {sklearn.__version__}")

print("\nEach bundle must load AND score; loading alone does not prove"
      " compatibility,\nsince the LogisticRegression breakage only surfaces"
      " inside predict_proba.")

# The four ensembles are pickled sklearn pipelines and have to survive the
# installed sklearn. The AlphaGenome half of the MERGE + AlphaGenome average is
# not a pickle at all -- it is a JSON percentile table, checked separately below.
AVERAGE_TASKS = ["splicing", "noncoding"]
BUNDLES = [f"MERGE_noAlphaGenome_{t}.pkl" for t in TASKS]

for filename in BUNDLES:
    task = filename[:-4]
    path = local_engine.get_bundled_model_path(filename)
    try:
        with warnings.catch_warnings():
            # InconsistentVersionWarning is expected and harmless below 1.9.0;
            # it is not what this test is about.
            warnings.simplefilter("ignore")
            bundle = joblib.load(path)
    except Exception as e:
        check(f"{task}: load", False, f"— {type(e).__name__}: {e}")
        continue

    feats = bundle["features"]
    # A fixed synthetic row. The probability itself is meaningless; what matters
    # is that predict_proba completes and returns a probability at all.
    X = pd.DataFrame([{f: 0.5 for f in feats}])[feats]
    try:
        proba = bundle["pipeline"].predict_proba(X)[:, 1][0]
    except Exception as e:
        check(f"{task}: predict_proba", False, f"— {type(e).__name__}: {e}")
        continue

    check(f"{task}: load + predict_proba", 0.0 <= float(proba) <= 1.0,
          f"— algo={bundle.get('algorithm')}, proba={proba:.6f}")

    if filename.startswith("MERGE_"):
        leaked = [f for f in feats if f.lower().startswith("alphagenome")]
        check(f"{task}: panel is AlphaGenome-free", not leaked,
              f"— {leaked}" if leaked else "")

# The AlphaGenome reference table behind the average. A wheel that dropped it
# would turn the average into "unavailable" on every variant.
ref_path = local_engine.get_bundled_model_path("alphagenome_reference_quantiles.json")
try:
    with open(ref_path, encoding="utf-8") as fh:
        ref = json.load(fh)
except Exception as e:
    ref = {}
    check("AlphaGenome reference table loads", False, f"— {type(e).__name__}: {e}")
check("AlphaGenome reference table declares labels_used = false",
      ref.get("labels_used") is False, f"— {ref.get('labels_used')}")
for name in AVERAGE_TASKS:
    t = (ref.get("tasks") or {}).get(name) or {}
    sc, pc = t.get("scores") or [], t.get("percentiles") or []
    check(f"{name}: reference scores ascend and pair with percentiles",
          len(sc) > 1 and len(sc) == len(pc) and all(a < b for a, b in zip(sc, sc[1:])),
          f"— {t.get('feature')}, {len(sc)} scores")

# The MERGE + AlphaGenome average is only interpretable against its own
# cut-points, and those ship as a separate JSON. A wheel that dropped it would
# score every variant against the ensemble bands instead, silently. The file
# still carries all four tasks; only the two that offer the average are read.
thresholds_path = local_engine.get_bundled_model_path("merge_final_thresholds.json")
try:
    with open(thresholds_path, encoding="utf-8") as fh:
        thresholds = json.load(fh)
except Exception as e:
    thresholds = {}
    check("MERGE + AlphaGenome average thresholds load", False,
          f"— {type(e).__name__}: {e}")
for name in AVERAGE_TASKS:
    cuts = (thresholds.get(name) or {}).get("cuts")
    check(f"{name}: average cut-points present and descending",
          isinstance(cuts, list) and len(cuts) == 4
          and all(cuts[i] > cuts[i + 1] for i in range(3)),
          f"— {cuts}")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    if any(t in f for f in failures for t in ("coding", "noncoding")):
        print("\nBoth failing panels are LogisticRegression. If this is an "
              "AttributeError on\n'multi_class', the installed scikit-learn is "
              f"older than {MIN_SKLEARN} — upgrade it.")
else:
    print(f"All {len(BUNDLES)} bundles load and score under sklearn "
          f"{sklearn.__version__}.")

raise SystemExit(1 if failures else 0)
