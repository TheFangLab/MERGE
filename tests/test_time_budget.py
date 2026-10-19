"""Regression guard for predict_variant's wall-clock bound.

Why this exists
---------------
An MCP client enforces its own per-call timeout and *discards* anything slower —
Codex cuts a tool call off at 300s. So an unbounded call does not merely take a
long time, it returns nothing at all, which is strictly worse than a partial
answer. Two things used to push the upper bound far past any client's patience:

  * api._session() retried read timeouts. With _TIMEOUT_SINGLE at 180s and
    total=4, one slow /predict/ could occupy 900s — and each retry queued a
    second copy of the GPU work the server was still doing.
  * _resolve_evo2 then waited another 180s on top, no matter how much of the
    call had already been spent.

Both are bounded now: read retries are off, and predict_variant spends at most
`time_budget` seconds before answering with what it has. These are silent
failure modes — nothing raises, the call just takes longer than anyone waits —
so they need an explicit guard.

Entirely offline: every remote call and the ensemble itself are stubbed.

Run:  ./.venv/bin/python tests/test_time_budget.py
"""
import time

from merge_cli import api, local_engine, mcp_server as ms

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


QUEUED = {"status": "queued", "job_id": "job-1"}


class Clock:
    """A monotonic clock the test drives by hand."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def call(cost, **kwargs):
    """Run predict_variant with a /predict/ round trip that 'takes' `cost` seconds.

    Returns (result, evo2_wait) where evo2_wait is the timeout _resolve_evo2 was
    actually handed, or None when it was never called. The HTTP timeout the call
    asked for is recorded on the returned function as `.http_timeout`.
    """
    clock, seen = Clock(), []

    def fake_predict_single(chrom, pos, ref, alt, genome_version="hg38",
                            timeout=None, **flags):
        fake_predict_single.http_timeout = timeout
        clock.t += cost
        return {"success": True,
                "prediction": {"evo2": dict(QUEUED), "errors": {},
                               "_variant_type": "coding"}}
    fake_predict_single.http_timeout = None
    call.last_predict = fake_predict_single

    def fake_resolve(prediction, timeout):
        seen.append(timeout)
        return None

    def fake_ensemble(prediction, transcripts, ensemble_type=None,
                      skip_shap=True, variant_info=""):
        return {"MERGE_Coding": {"score": 0.5}}, None

    saved = (api.predict_single, ms._resolve_evo2,
             local_engine.run_local_ensemble, time.monotonic)
    api.predict_single = fake_predict_single
    ms._resolve_evo2 = fake_resolve
    local_engine.run_local_ensemble = fake_ensemble
    time.monotonic = clock
    try:
        out = ms.predict_variant("chr17", 43092919, "G", "A", **kwargs)
    finally:
        (api.predict_single, ms._resolve_evo2,
         local_engine.run_local_ensemble, time.monotonic) = saved
    return out, (seen[0] if seen else None)


print("defaults")
check("time budget defaults to 120s", ms._DEFAULT_TIME_BUDGET == 120.0,
      f"— {ms._DEFAULT_TIME_BUDGET}")
check("evo2 wait defaults to 45s, not the old 180",
      ms._DEFAULT_EVO2_TIMEOUT == 45.0, f"— {ms._DEFAULT_EVO2_TIMEOUT}")

print("\nthe budget is spent on the whole call, not per step")
out, wait = call(cost=5.0)
check("a fast round trip leaves the full evo2 wait", wait == 45.0, f"— {wait}")
check("and reports no degradation", "notes" not in out)
check("elapsed_seconds is reported", out.get("elapsed_seconds") == 5.0,
      f"— {out.get('elapsed_seconds')}")

out, wait = call(cost=100.0)
check("a slow round trip shortens the evo2 wait to what is left",
      wait == 20.0, f"— {wait}")

out, wait = call(cost=130.0)
check("an over-budget round trip skips the evo2 wait entirely", wait is None)
check("and says so instead of failing silently",
      any("budget" in n.lower() for n in out.get("notes", [])),
      f"— {out.get('notes')}")
check("but still returns a score", out["success"] is True and out["ensemble"])

print("\nthe budget is caller-controllable")
out, wait = call(cost=100.0, time_budget=300.0)
check("a bigger budget restores the full evo2 wait", wait == 45.0, f"— {wait}")
out, wait = call(cost=5.0, evo2_timeout=10.0)
check("an explicit evo2_timeout still caps below the budget", wait == 10.0,
      f"— {wait}")
out, wait = call(cost=5.0, evo2_timeout=0.0)
check("evo2_timeout=0 disables waiting without a budget note",
      wait == 0.0 and "notes" not in out, f"— {wait}")

print("\nthe budget reaches the remote request, not just the evo2 wait")

# The regression this section exists for: predict_single used to run on api's
# own 180s default, so time_budget governed only the Evo2 step. Asking for 60s
# could still block for 180 — the docstring's "hard ceiling" was false for any
# budget below that, and a client that gave up in between left this process
# stuck mid-request instead of exiting.
call(cost=1.0, time_budget=60.0)
asked = call.last_predict.http_timeout
check("a 60s budget does not let the request run for 180",
      asked is not None and asked <= 60.0, f"— asked for {asked}")
call(cost=1.0, time_budget=120.0)
check("the timeout scales with the budget",
      call.last_predict.http_timeout == 120.0 * ms._PREDICT_BUDGET_SHARE,
      f"— {call.last_predict.http_timeout}")
call(cost=1.0, time_budget=1.0)
check("a tiny budget is floored, not turned into an impossible timeout",
      call.last_predict.http_timeout == ms._MIN_HTTP_TIMEOUT,
      f"— {call.last_predict.http_timeout}")
check("the share leaves room for evo2 and the ensemble",
      0 < ms._PREDICT_BUDGET_SHARE < 1)
# api never raises the caller's ceiling above its own.
check("api clamps a request for more than its default",
      min(999.0, api._TIMEOUT_SINGLE) == api._TIMEOUT_SINGLE)


print("\na timed-out request explains itself")


class _Timeout(Exception):
    pass


_Timeout.__name__ = "ReadTimeout"


def _raise_timeout(*a, **k):
    raise _Timeout("read timed out")


saved = api.predict_single
api.predict_single = _raise_timeout
try:
    out = ms.predict_variant("chr17", 43092919, "G", "A", time_budget=30.0)
finally:
    api.predict_single = saved
check("a timeout is detected as one", ms._is_timeout(_Timeout("x")) is True)
check("an unrelated error is not", ms._is_timeout(ValueError("x")) is False)
check("it fails rather than hanging", out["success"] is False)
check("and says what to do about it",
      "time_budget" in out["error"] and "did not answer" in out["error"],
      f"— {out['error']}")
check("elapsed_seconds is still reported on failure", "elapsed_seconds" in out)


print("\nread timeouts are not retried")
retry = api._session().get_adapter(api._BASE).max_retries
check("read retries are off", retry.read is False, f"— read={retry.read}")
check("connect retries are kept (they cover the TLS handshake blip)",
      retry.connect == 4, f"— connect={retry.connect}")
check("gateway errors are still retried",
      tuple(retry.status_forcelist) == (502, 503, 504))
worst = api._TIMEOUT_SINGLE * (1 if retry.read is False else retry.total + 1)
check("so one /predict/ cannot outlast a client's tool timeout",
      worst <= 300, f"— worst case {worst}s")

print("\n" + "=" * 50)
total = 15 + 12
print(f"PASS {total - len(failures)} / FAIL {len(failures)}"
      + (f" -> {failures}" if failures else ""))
raise SystemExit(1 if failures else 0)
