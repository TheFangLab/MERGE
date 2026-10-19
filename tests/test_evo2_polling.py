"""Regression guard for the async Evo2 polling added to the MCP server.

Deterministic part: drives mcp_server._resolve_evo2 against a stubbed
api.get_evo2_status, covering every branch (done / error / timeout /
already-scored / no job / polling disabled). No network.

Live part (only with --live): one real call against merge.fanglab.cn.

Background: the single-variant endpoint answers {"status":"queued","job_id":...}
instead of a score when its Evo2 worker is busy or in degraded CPU mode. Nothing
followed up on that job_id, so evo2_*_score was silently median-imputed even
though Evo2 is one of the six selected features in the coding panel.

Run:  ./.venv/bin/python tests/test_evo2_polling.py [--live]
"""
import sys

from merge_cli import api, mcp_server as ms

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


class _Stub:
    """Returns the given responses in order; repeats the last one forever."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def __call__(self, job_id):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def run(pred, stub, timeout=180.0, no_sleep=True):
    orig_status, orig_sleep = api.get_evo2_status, ms.time.sleep
    api.get_evo2_status = stub
    if no_sleep:
        ms.time.sleep = lambda s: None
    try:
        return ms._resolve_evo2(pred, timeout)
    finally:
        api.get_evo2_status, ms.time.sleep = orig_status, orig_sleep


QUEUED = {"status": "queued", "job_id": "abc123"}
# Verbatim from the live server for chr17:43092919 G>A.
DONE = {"status": "done", "elapsed_seconds": 5.22,
        "result": {"llr_score": -0.002934, "model_name": "evo2_7b",
                   "ref_score": -0.925199, "var_score": -0.922265}}

print("=== 1. queued -> done: score folded into prediction ===")
pred = {"evo2": dict(QUEUED)}
stub = _Stub(dict(QUEUED), dict(QUEUED), DONE)
note = run(pred, stub)
check("no note on success", note is None, f"note={note!r}")
check("polled until terminal", stub.calls == 3, f"calls={stub.calls}")
check("llr_score present", pred["evo2"].get("llr_score") == -0.002934)
check("model_name kept (routes to evo2_7b_score)",
      pred["evo2"].get("model_name") == "evo2_7b")
check("status flipped to success", pred["evo2"].get("status") == "success")

print("=== 2. the folded result is what the ensemble actually reads ===")
import importlib.util
import os
_spec = importlib.util.spec_from_file_location("ep_ev", os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "merge_cli", "data", "models",
    "ensemble_predict.py")))
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)
feats = ep.build_feature_values(pred, [])
check("evo2_7b_score == llr_score", feats["evo2_7b_score"] == -0.002934,
      f"-> {feats['evo2_7b_score']}")
import numpy as np
check("unqueued evo2 still NaN",
      np.isnan(ep.build_feature_values({"evo2": dict(QUEUED)}, [])["evo2_7b_score"]))

print("=== 3. terminal error states are not polled forever ===")
for state, payload in (
    ("error", {"status": "error", "detail": "Reference mismatch at chr17:43092919"}),
    ("failed", {"status": "failed", "detail": "cuda oom"}),
):
    pred = {"evo2": dict(QUEUED)}
    stub = _Stub(payload)
    note = run(pred, stub)
    check(f"{state}: note returned", note and "median-imputed" in note)
    check(f"{state}: detail surfaced", note and payload["detail"][:12] in note)
    check(f"{state}: stopped after 1 poll", stub.calls == 1, f"calls={stub.calls}")

print("=== 4. timeout: gives up, says so, leaves feature imputed ===")
pred = {"evo2": dict(QUEUED)}
stub = _Stub(dict(QUEUED))
note = run(pred, stub, timeout=0.05, no_sleep=False)
check("note mentions degraded CPU mode", note and "degraded CPU mode" in note)
check("no score invented", not any(pred["evo2"].get(k) is not None
                                   for k in ms._EVO2_SCORE_KEYS))

print("=== 5. no polling when there is nothing to poll ===")
cached = {"evo2": {"status": "success", "llr_score": -0.5, "model_name": "evo2_7b"}}
stub = _Stub(DONE)
check("cache hit -> untouched", run(cached, stub) is None and stub.calls == 0,
      f"calls={stub.calls}")
stub = _Stub(DONE)
check("no evo2 key -> untouched", run({}, stub) is None and stub.calls == 0)
stub = _Stub(DONE)
check("evo2 not a dict -> untouched", run({"evo2": None}, stub) is None
      and stub.calls == 0)
stub = _Stub(DONE)
check("no job_id -> untouched",
      run({"evo2": {"status": "queued"}}, stub) is None and stub.calls == 0)

print("=== 6. evo2_timeout=0 disables waiting ===")
stub = _Stub(DONE)
note = run({"evo2": dict(QUEUED)}, stub, timeout=0)
check("note says disabled", note and "polling disabled" in note)
check("never called the API", stub.calls == 0, f"calls={stub.calls}")

print("=== 7. transport failure degrades, does not raise ===")
def boom(job_id):
    raise RuntimeError("connection reset")
pred = {"evo2": dict(QUEUED)}
note = run(pred, boom)
check("note reports the failure", note and "connection reset" in note)

print("\n" + "=" * 50)
total = 21
print(f"PASS {total - len(failures)} / FAIL {len(failures)}"
      + (f" -> {failures}" if failures else ""))

if "--live" in sys.argv:
    print("\n=== LIVE: real endpoint, chr17:43092919 G>A ===")
    resp = api.predict_single("chr17", 43092919, "G", "A", "hg38",
                              use_evo2=True, evo2_model="evo2_7b")
    pred = resp["prediction"]
    print(f"  server returned evo2: {str(pred.get('evo2'))[:150]}")
    note = ms._resolve_evo2(pred, 180.0)
    print(f"  after _resolve_evo2 : {str(pred.get('evo2'))[:150]}")
    print(f"  note                : {note}")
    live_ok = note is None and any(pred["evo2"].get(k) is not None
                                  for k in ms._EVO2_SCORE_KEYS)
    check("live evo2 resolved to a score", live_ok)

raise SystemExit(1 if failures else 0)
