"""Follow up on Evo2 jobs that the MERGE server queued instead of scoring inline.

The server answers with {"status": "queued", "job_id": ...} instead of a score
whenever its Evo2 worker is busy or in degraded CPU mode. Without a follow-up
the evo2_*_score feature is simply absent and gets median-imputed, silently,
even though Evo2 is in the splicing, non-coding and non-SNV panels. Used by
`merge predict` (remote mode) and by the MCP server.
"""
import time
from typing import Optional

from . import api

EVO2_SCORE_KEYS = ("llr_score", "score_mean", "score")
EVO2_POLL_MAX_INTERVAL = 5.0


def resolve_evo2(prediction: dict, timeout: float) -> Optional[str]:
    """Poll a queued async Evo2 job and fold its score into `prediction`.

    On success `prediction['evo2']` is replaced by the resolved result, whose
    'model_name' routes it to the right evo2_{7b,7b_base,1b_base}_score column
    and whose 'llr_score' is what ensemble_predict._evo2_value() reads.
    Returns a human-readable note when the score could not be obtained.
    """
    evo2 = prediction.get("evo2")
    if not isinstance(evo2, dict):
        return None
    job_id = evo2.get("job_id")
    if not job_id:
        return None
    if any(evo2.get(k) is not None for k in EVO2_SCORE_KEYS):
        return None  # already scored (server-side cache hit)
    if timeout <= 0:
        return (f"Evo2 left unresolved (job {job_id}, polling disabled); "
                "evo2 score median-imputed.")

    deadline = time.monotonic() + timeout
    interval, status = 1.0, str(evo2.get("status") or "queued").lower()
    while True:
        try:
            st = api.get_evo2_status(job_id)
        except Exception as e:
            return f"Evo2 polling failed ({e}); evo2 score median-imputed."
        status = str(st.get("status") or "").lower()

        if status == "done":
            result = st.get("result") or {}
            if not any(result.get(k) is not None for k in EVO2_SCORE_KEYS):
                return ("Evo2 job finished without a score; "
                        "evo2 score median-imputed.")
            prediction["evo2"] = {**result, "status": "success", "job_id": job_id,
                                  "elapsed_seconds": st.get("elapsed_seconds")}
            return None
        if status in ("error", "failed"):
            return (f"Evo2 failed ({st.get('detail') or status}); "
                    "evo2 score median-imputed.")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            prediction["evo2"] = {**evo2, "status": status or "queued"}
            return (f"Evo2 still '{status or 'queued'}' after {timeout:.0f}s — the "
                    "server runs Evo2 in degraded CPU mode; evo2 score "
                    f"median-imputed. Job {job_id} may finish later.")
        time.sleep(min(interval, remaining))
        interval = min(interval * 1.5, EVO2_POLL_MAX_INTERVAL)
