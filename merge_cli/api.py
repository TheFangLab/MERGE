"""
merge_cli/api.py
封装所有对 Django 后端的 HTTP 请求。
服务器地址固定为 https://merge.fanglab.cn/，无需 Token 认证。

模型开关与网站 views.py 的 use_* 参数对应（12 个开关覆盖 13 个模型：
evo2_7b 与 evo2_7b_base 共用 use_evo2，由 evo2_model 选择）。
popEVE 已移除：它不在任何 MERGE 集成面板的特征里。
"""
import sys
from typing import Optional

import requests

from .config import FIXED_API_URL, get_alphagenome_api_key

_TIMEOUT_SINGLE = 180
_TIMEOUT_BATCH  = 60

_BASE = FIXED_API_URL

# The model switches the server accepts. Kept in one place so predict/batch and
# the CLI cannot drift apart.
MODEL_FLAGS = (
    "use_alphagenome",
    "use_hyenadna",
    "use_nt",
    "use_alphamissense",
    "use_esm1b",
    "use_gpn_msa",
    "use_evo2",
    "use_enformer",
    "use_generator",
    "use_generator_v2",
    "use_nt_v2",
    "use_carbon",
)

# Evo2 weights the server can switch between (they are GPU-exclusive there).
EVO2_MODELS = ("evo2_7b", "evo2_7b_base")

# Every model runs by default: each one is part of at least one ensemble feature
# panel, and a model that is skipped has its feature median-imputed instead.
_DEFAULT_ON = set(MODEL_FLAGS)


def _headers() -> dict:
    return {"Accept": "application/json"}


def _session() -> requests.Session:
    session = requests.Session()
    # The MERGE endpoint is fixed and reachable directly. Some cluster nodes set
    # HTTP(S)_PROXY values that break Python TLS while curl still works.
    session.trust_env = False

    # The server occasionally drops the TLS connection mid-handshake — a cold
    # connection fails with "SSL: UNEXPECTED_EOF_WHILE_READING" and the immediate
    # retry succeeds. Retry that, and gateway errors, a few times with backoff so
    # a transient blip does not surface as a hard failure. allowed_methods=False
    # makes it retry POSTs too; the endpoints are idempotent (they compute a
    # score, they do not mutate anything).
    #
    # read=False is deliberate — it re-raises the read timeout as itself rather
    # than retrying. A read timeout is not a blip: it means the server
    # accepted the request and is still working on it, and _TIMEOUT_SINGLE is
    # already 180s. Retrying that multiplied the worst case by the retry count —
    # five attempts at 180s put predict_single's upper bound at 900s, which no
    # MCP client tolerates (Codex times a tool call out at 300s). Retrying also
    # queues a *second* copy of an expensive GPU job the server is still running.
    try:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        try:
            retry = Retry(total=4, connect=4, read=False, backoff_factor=0.6,
                          status_forcelist=(502, 503, 504), allowed_methods=False)
        except TypeError:  # urllib3 < 1.26 used method_whitelist
            retry = Retry(total=4, connect=4, read=False, backoff_factor=0.6,
                          status_forcelist=(502, 503, 504), method_whitelist=False)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception:
        pass  # retries are a bonus; a plain session still works
    return session


def _flag_payload(flags: dict) -> dict:
    """Render model switches as the lowercase strings the server expects."""
    out = {}
    for name in MODEL_FLAGS:
        val = flags.get(name, name in _DEFAULT_ON)
        out[name] = str(bool(val)).lower()
    return out


def _evo2_model(flags: dict) -> str:
    m = flags.get("evo2_model") or "evo2_7b"
    return m if m in EVO2_MODELS else "evo2_7b"


def _alphagenome_payload() -> dict:
    """The user's AlphaGenome key, if they have one, as a payload fragment.

    DeepMind licenses the AlphaGenome API for non-commercial use only, to the
    key holder. MERGE therefore never spends its own key on a user's request:
    a live AlphaGenome call is made under the caller's key or not at all.

    Omitted entirely when no key is set — that is not an error. The server still
    answers from its precomputed cache, which holds AlphaGenome values computed
    once by the MERGE team and published as research data; only a cache miss
    needs a key, and then the AlphaGenome features come back marked unavailable
    rather than failing the prediction.
    """
    key = get_alphagenome_api_key()
    return {"alphagenome_api_key": key} if key else {}


# ─────────────────────────────────────────────────────────────
# 1. 单变异预测  POST /predict/
# ─────────────────────────────────────────────────────────────
def predict_single(
    chrom: str, pos: int, ref: str, alt: str,
    genome_version: str = "hg38",
    timeout: Optional[float] = None,
    **flags,
) -> dict:
    """Score one variant remotely.

    `timeout` caps the read, defaulting to _TIMEOUT_SINGLE. Callers working to a
    deadline must pass it: this is by far the longest step, so a budget that
    does not reach it is not a budget at all. It is only ever lowered, never
    raised above the module default.
    """
    payload = {
        "chr": chrom, "pos": pos, "ref": ref, "alt": alt,
        "genome_version": genome_version,
        "evo2_model": _evo2_model(flags),
        **_flag_payload(flags),
        **_alphagenome_payload(),
    }
    resp = _session().post(
        f"{_BASE}/predict/",
        json=payload,
        headers=_headers(),
        timeout=min(timeout, _TIMEOUT_SINGLE) if timeout else _TIMEOUT_SINGLE,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# 2. 集成模型评分  POST /ensemble/
# ─────────────────────────────────────────────────────────────
def predict_ensemble(prediction: dict, all_transcripts: list,
                     ensemble_type: Optional[str] = None) -> dict:
    """Ask the server for the MERGE ensemble score.

    The server picks the task itself (non-SNV by allele length, else ANNOVAR) and
    currently ignores `ensemble_type`; it is forwarded for forward-compatibility
    only. Accepting the argument also keeps this callable from the CLI's shared
    code path, which passes the resolved type for both local and remote modes.
    """
    payload = {"prediction": prediction, "all_transcripts": all_transcripts}
    if ensemble_type:
        payload["ensemble_type"] = ensemble_type
        payload["variant_type"] = ensemble_type
    resp = _session().post(
        f"{_BASE}/ensemble/",
        json=payload,
        headers=_headers(),
        timeout=_TIMEOUT_SINGLE,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# 3. 批量提交  POST /api/submit_batch_job/
# ─────────────────────────────────────────────────────────────
def submit_batch(
    vcf_path: str,
    email: str,
    genome_version: str = "hg38",
    **flags,
) -> dict:
    data = {
        "notify_email":    email,
        "genome_version":  genome_version,
        "evo2_model":      _evo2_model(flags),
        **_flag_payload(flags),
        **_alphagenome_payload(),
    }
    with open(vcf_path, "rb") as f:
        resp = _session().post(
            f"{_BASE}/api/submit_batch_job/",
            data=data,
            files={"vcf_file": (vcf_path.split("/")[-1], f)},
            headers=_headers(),
            timeout=_TIMEOUT_BATCH,
        )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# 4. 查询批量任务状态  GET /api/batch_job/<job_id>/
# ─────────────────────────────────────────────────────────────
def get_batch_status(job_id: str) -> dict:
    resp = _session().get(
        f"{_BASE}/api/batch_job/{job_id}/",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# 5. 查询 Evo2 异步任务状态  GET /api/evo2/status/<job_id>/
# ─────────────────────────────────────────────────────────────
# The single-variant endpoint returns {"status": "queued", "job_id": ...} in
# place of a score whenever the server's Evo2 worker is busy or running in
# degraded CPU mode. Terminal states observed from the live server:
#   {"status": "done",  "elapsed_seconds": 5.22, "result": {"llr_score": ..., "model_name": ...}}
#   {"status": "error", "detail": "Reference mismatch at chr17:43092919: genome=G, provided=A"}
# "failed" is accepted as terminal too — the upstream GPU service uses it.
EVO2_TERMINAL_STATES = ("done", "error", "failed")


def get_evo2_status(job_id: str) -> dict:
    """Poll one async Evo2 job. Additive: nothing in the CLI calls this."""
    resp = _session().get(
        f"{_BASE}/api/evo2/status/{job_id}/",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
