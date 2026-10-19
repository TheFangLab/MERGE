"""The ensemble module must come from the package, never from model_dir.

The bug this guards against, reported from a fresh Windows install on
2026-07-31:

    Ensemble score failed (raw scores unaffected):
    module 'ep_local' has no attribute 'ENSEMBLE_TYPE_TO_TASK'

get_bundled_model_path() checks the user's configured model_dir before the
packaged copy, which is correct for the .pkl bundles — that override is how a
user points at their own models. It was also being applied to
ensemble_predict.py, which is *code*. Older merge-cli releases shipped a
different ensemble module (one with no ENSEMBLE_TYPE_TO_TASK, no
TASK_PKL, and a completely different feature pipeline), and any copy of it left
in ~/.merge-models would be found first and executed in place of the packaged
module. The prediction then failed with an AttributeError several frames
away from the cause, naming neither the file nor the reason.

Two guarantees are tested here:
  1. a stale ensemble_predict.py in model_dir is ignored outright;
  2. if a wrong module is ever loaded anyway, the error names the file and says
     what is missing, instead of surfacing as an AttributeError later.

Run:  ./.venv/bin/python tests/test_ep_module_isolation.py
"""
import os
import tempfile
from pathlib import Path

from merge_cli import local_engine as le

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


# A stand-in for the old module: valid Python, imports cleanly, but exposes none
# of the current API. Executing it is exactly the failure being guarded against.
STALE_MODULE = '''
"""ensemble_predict.py (stand-in for an older module)"""
TRAIN_CSV_PATHS = {}
FEATURE_ALIASES = {}
def extract_features(*a, **k): return {}
'''

with tempfile.TemporaryDirectory() as td:
    home = Path(td) / "home"
    (home / ".merge-models").mkdir(parents=True)
    stale = home / ".merge-models" / "ensemble_predict.py"
    stale.write_text(STALE_MODULE)

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    le._ep_module_cache.clear()
    try:
        # The data override still applies to model bundles...
        pkl_path = le.get_bundled_model_path("MERGE_noAlphaGenome_coding.pkl")
        check("pkl still resolves (the model_dir override is intact for data)",
              Path(pkl_path).is_file(), f"— {Path(pkl_path).name}")

        # ...but must not apply to the module.
        chosen = le.get_bundled_model_path("ensemble_predict.py", packaged_only=True)
        check("packaged_only ignores the stale copy in model_dir",
              Path(chosen).resolve() != stale.resolve(), f"— {chosen}")

        ep = le._load_ep_module()
        check("the loaded module is the packaged one",
              Path(ep.__file__).resolve() != stale.resolve())
        for attr in le._EP_REQUIRED_ATTRS:
            check(f"loaded module exposes {attr}", hasattr(ep, attr))
        check("it is the four-task module",
              sorted(ep.TASK_PKL) == ["coding", "noncoding", "nonsnv", "splice"],
              f"— {sorted(ep.TASK_PKL)}")

        # If a wrong module is somehow loaded, the failure must be legible.
        le._ep_module_cache.clear()
        real = le.get_bundled_model_path
        le.get_bundled_model_path = lambda fn, packaged_only=False: stale
        try:
            le._load_ep_module()
            check("loading a stale module raises", False)
        except RuntimeError as e:
            msg = str(e)
            check("loading a stale module raises RuntimeError", True)
            check("the error names the offending file", str(stale) in msg)
            check("the error names a missing attribute",
                  "ENSEMBLE_TYPE_TO_TASK" in msg)
            check("the error explains the likely cause",
                  "older merge-cli" in msg)
        except AttributeError:
            check("stale module raises RuntimeError, not a bare AttributeError", False)
        finally:
            le.get_bundled_model_path = real
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        le._ep_module_cache.clear()

print()
print(f"{len(failures)} FAILED: {', '.join(failures)}" if failures
      else "Ensemble module isolation holds.")
raise SystemExit(1 if failures else 0)
