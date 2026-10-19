"""
merge_cli/mcp_server.py — MCP server exposing MERGE variant prediction to Claude.

Always runs in *remote* mode: it never starts local GPU model services. Per-model
raw scores (AlphaMissense, ESM1b, GPN-MSA, AlphaGenome, Enformer, Evo2, Carbon,
GENERATOR/-v2, NT/NT-v2, HyenaDNA) come from the merge.fanglab.cn API; the
MERGE probabilities are computed on this machine from the bundled
MERGE_noAlphaGenome_{coding,splicing,noncoding,nonsnv}.pkl models (CPU only, no
GPU needed).

What comes back per variant, and the two are not interchangeable:

  score       the ensemble, for every variant class. Its feature panel contains
              no AlphaGenome column, so no AlphaGenome output trained it.
              Bands: >=0.8 / 0.6 / 0.4 / 0.2.
  merge_alphagenome_average
              0.5 x score + 0.5 x the percentile of one AlphaGenome column
              among the MERGE benchmark variants (a lookup table, not a
              fitted model).
              SPLICING AND NON-CODING ONLY — on coding and non-SNV the best
              AlphaGenome feature is near chance (AUROC 0.571 / 0.512), so it is
              null there with status 'not_offered_for_this_variant_type'. It has
              its own per-task cut-points
              (`merge_alphagenome_average_interpretation`) because averaging
              changes the scale — never read it against the ensemble's
              0.8/0.6/... bands. It is also null when AlphaGenome has no value
              for the variant; `merge_alphagenome_average_status` says which.

AlphaGenome is still extracted and reported in full under
`prediction["alphagenome"]`; it simply no longer feeds the ensemble.

This module only reads from `api.py` / `local_engine.py` — it does not modify
them, so `merge predict` from the terminal is unaffected by anything here.
"""
import re
import time
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import api, local_engine
from .config import FIXED_API_URL

mcp = FastMCP("merge-cli")

# Every model on, evo2_7b as the primary Evo2 weight — mirrors the CLI's
# default flags (`merge predict` with no --no-<model> options).
_DEFAULT_FLAGS = {
    "use_alphagenome": True, "use_hyenadna": True, "use_nt": True,
    "use_alphamissense": True, "use_esm1b": True, "use_gpn_msa": True,
    "use_evo2": True, "use_enformer": True, "use_generator": True,
    "use_generator_v2": True, "use_nt_v2": True, "use_carbon": True,
    "evo2_model": "evo2_7b",
}


def _normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom if chrom.startswith("chr") else "chr" + chrom


_FASTA_BASE_RE = re.compile(r"FASTA=([ACGTN]+)", re.I)


def _ref_mismatch(prediction: dict) -> Optional[str]:
    """Return an explanatory message if the server reported a reference mismatch.

    A mismatch means the variant as written does not exist in this genome build:
    every sequence model refuses it and dbNSFP finds nothing, so the ensemble
    would be classifying almost entirely median-imputed values — and still emit
    a confident-looking label. Refuse to score, and name the real reference base
    so the caller can correct the allele or the build.
    """
    errors = prediction.get("errors") or {}
    hits = sorted(m for m, msg in errors.items()
                  if "reference mismatch" in str(msg).lower())
    if not hits:
        return None
    bases = sorted({m.group(1).upper() for m in
                    (_FASTA_BASE_RE.search(str(errors[k])) for k in hits) if m})
    detail = f" the reference base is {'/'.join(bases)}, not what was given;" if bases else ""
    return (f"reference mismatch against {prediction.get('genome_build')} at this "
            f"position —{detail} models affected: {', '.join(hits)}. No MERGE score "
            "is reported, because the ensemble would be scoring imputed values. "
            "Correct the ref allele (or the genome build) and retry.")


from .evo2_poll import (EVO2_POLL_MAX_INTERVAL as _EVO2_POLL_MAX_INTERVAL,  # noqa: E402
                        EVO2_SCORE_KEYS as _EVO2_SCORE_KEYS,
                        resolve_evo2 as _resolve_evo2)

# How long one predict_variant call may take, end to end, before it gives up
# waiting and answers with whatever it has. This exists because MCP clients
# enforce their own per-call timeout and simply discard the answer when it is
# exceeded: Codex cuts a tool call at 300s, and a discarded answer is worse than
# a partial one — the caller learns nothing at all. A cached variant returns in
# ~20s, so this only ever bites when the server is genuinely backed up.
_DEFAULT_TIME_BUDGET = 120.0

# Evo2 is queued server-side rather than scored inline, so this wait is pure
# overhead on top of the /predict/ round trip. 45s covers the common case (a
# warm worker answers in a few seconds) without pushing the total past the
# budget above; raise it when you specifically need the Evo2 feature real
# rather than median-imputed.
_DEFAULT_EVO2_TIMEOUT = 45.0

# Share of the budget the /predict/ round trip may take, leaving the rest for
# the Evo2 wait and the local ensemble. The floor keeps a very small budget from
# producing a timeout so short that nothing could ever answer within it.
_PREDICT_BUDGET_SHARE = 0.75
_MIN_HTTP_TIMEOUT = 15.0


def _is_timeout(exc: BaseException) -> bool:
    """Whether an exception is a request timeout, without importing requests here."""
    return any("timeout" in type(e).__name__.lower() or "Timeout" in str(type(e))
               for e in (exc, exc.__cause__, exc.__context__) if e is not None)


@mcp.tool()
def predict_variant(chrom: str, pos: int, ref: str, alt: str,
                     genome: str = "hg38",
                     ensemble_type: Optional[str] = None,
                     include_shap: bool = False,
                     evo2_timeout: float = _DEFAULT_EVO2_TIMEOUT,
                     time_budget: float = _DEFAULT_TIME_BUDGET) -> dict:
    """Full MERGE prediction for one variant, with every per-model score exposed.

    For a plain "is this variant pathogenic?" use `answer_variant_pathogenicity`
    instead — it takes the variant as one string and returns a interpreted
    answer. Reach for this one when you need the raw material: the individual
    model scores, the exact feature panel, a forced ensemble type, SHAP, or
    control over the timeouts.

    Remote mode: raw per-model scores come from https://merge.fanglab.cn;
    the final MERGE probability is computed locally from the bundled
    coding/splicing/noncoding/nonsnv model.

    Timing: ~20s when the server has the variant cached, up to `time_budget`
    (default 120s) when it does not. It never runs longer than the budget — it
    answers with whatever it has and says so in `notes`.

    Args:
        chrom: Chromosome, e.g. "chr17" or "17".
        pos: 1-based variant position.
        ref: Reference allele (e.g. "A"; use ANNOVAR-style for indels).
        alt: Alternate allele.
        genome: "hg38" (default) or "hg19".
        ensemble_type: Force one of coding/splice/noncoding/nonsnv instead of
            auto-detecting from the variant (non-SNV is always auto-detected
            by allele length regardless of this).
        include_shap: If true, also generate a SHAP explanation (base64 PNG,
            slower). Default false — keep responses small for chat.
        evo2_timeout: Seconds to wait for the server's async Evo2 job. The
            server queues Evo2 rather than scoring inline, so without waiting
            the evo2 feature is median-imputed. 0 disables waiting. Also capped
            by whatever is left of `time_budget`.
        time_budget: Ceiling in seconds on the whole call, and it does bound
            the remote request, not merely the Evo2 wait. Raise it if your
            client's tool timeout is generous and you want the Evo2 feature
            real rather than imputed; lower it if your client cuts calls off
            early. Very small values are floored at a timeout something could
            plausibly answer within.

    Returns a dict with:
        success: bool  (False for a reference-allele mismatch — see `error`)
        prediction: raw per-model feature scores (as returned by the server)
        ensemble: {"<MERGE_task>": {"score": 0-1, "interpretation": str,
                   "merge_alphagenome_average": 0-1 or None,
                   "merge_alphagenome_average_interpretation": {...},
                   "merge_alphagenome_average_status": str,
                   "alphagenome_feature": str, "alphagenome_component": 0-1,
                   "missing_features": [...], ...}}
                   or an error string.
                   `score` is the AlphaGenome-free ensemble and is the one to
                   quote against the usual 0.8/0.6/0.4/0.2 bands.
                   `merge_alphagenome_average` is the 1:1 average with the
                   AlphaGenome component, offered for splicing and non-coding
                   only, and has its own cut-points, carried in
                   `merge_alphagenome_average_interpretation`.
        notes: present only when something degraded the score, e.g. Evo2 not
               resolving in time. Always read it alongside
               ensemble[...]["missing_features"]: a score built on imputed
               features is not trustworthy, and imputation pulls it toward the
               pathogenic side.
    """
    started = time.monotonic()
    chrom = _normalize_chrom(chrom)

    # The /predict/ round trip is the long pole, so the budget has to reach it.
    # Leaving it on api's own 180s default meant time_budget only ever governed
    # the Evo2 wait: asking for 60s could still block for 180, the docstring's
    # "hard ceiling" was false below 180, and a client that gave up in between
    # left this process stuck mid-request rather than exiting.
    http_timeout = max(_MIN_HTTP_TIMEOUT, time_budget * _PREDICT_BUDGET_SHARE)
    try:
        resp = api.predict_single(chrom, pos, ref, alt, genome,
                                  timeout=http_timeout, **_DEFAULT_FLAGS)
    except Exception as e:
        elapsed = time.monotonic() - started
        if _is_timeout(e):
            return {"success": False, "elapsed_seconds": round(elapsed, 1),
                    "error": (f"the MERGE server did not answer within "
                              f"{http_timeout:.0f}s. It scores an uncached variant "
                              "with 14 models, so a cold one can outlast a short "
                              "budget. Raise time_budget and retry.")}
        return {"success": False, "error": f"prediction request failed: {e}",
                "elapsed_seconds": round(elapsed, 1)}

    if not resp.get("success"):
        return {"success": False, "error": resp.get("error", "unknown server error")}

    prediction = resp["prediction"]
    prediction.setdefault("genome_build", genome)

    effective_type = ensemble_type or prediction.get("_variant_type")
    if not effective_type:
        try:
            effective_type = local_engine.get_variant_type(chrom, pos, ref, alt, genome)
        except Exception:
            effective_type = None
    if effective_type:
        prediction["_variant_type"] = effective_type

    mismatch = _ref_mismatch(prediction)
    if mismatch:
        return {"success": False, "error": mismatch, "prediction": prediction}

    # Resolve the async Evo2 job before scoring — the ensemble reads
    # prediction['evo2'], so this has to land first or the feature is imputed.
    # Whatever the /predict/ round trip already spent comes out of the budget:
    # the point is to bound the *call*, not each step of it.
    notes = []
    remaining = time_budget - (time.monotonic() - started)
    evo2_wait = max(0.0, min(evo2_timeout, remaining))
    if evo2_timeout > 0 and evo2_wait <= 0:
        notes.append(
            f"Time budget ({time_budget:.0f}s) was already spent fetching the "
            "per-model scores, so Evo2 was not waited for; its feature is "
            "median-imputed. Raise time_budget to let it finish.")
    else:
        evo2_note = _resolve_evo2(prediction, evo2_wait)
        if evo2_note:
            notes.append(evo2_note)

    dbnsfp = prediction.get("dbnsfp") or {}
    transcripts = dbnsfp.get("transcripts") or ([dbnsfp] if dbnsfp else [])
    variant_info = f"{chrom}:{pos} {ref}>{alt}"

    try:
        ens_result, ens_err = local_engine.run_local_ensemble(
            prediction, transcripts, ensemble_type=effective_type,
            skip_shap=not include_shap, variant_info=variant_info)
        ensemble = {"success": ens_err is None, "warning": ens_err, **(ens_result or {})}
    except Exception as e:
        ensemble = {"success": False, "error": str(e)}

    out = {"success": True, "prediction": prediction, "ensemble": ensemble,
           "elapsed_seconds": round(time.monotonic() - started, 1)}
    if notes:
        out["notes"] = notes
    return out


# ── One-shot entry point ────────────────────────────────────────────
#
# The two-step dance (get_variant_type, then predict_variant) matches how the
# pipeline works, not how anyone asks. "Is chr1:1040819 G>GC pathogenic?" is one
# question, and every decision in between — parsing the variant, picking the
# panel, choosing what to do when a model stalls, deciding whether the score can
# be trusted — is one this server can make better than the caller.

_HGVS_HINT_RE = re.compile(r"\b(?:[NX][MRPC]_\d|ENS[TGP]\d|[cgpmnr]\.\s*-?\d)")
_RSID_RE = re.compile(r"^\s*(rs\d+)\s*$", re.I)

# chr1:1040819 G>GC and the shapes people actually paste: colons, dashes, tabs
# (a VCF line pasted in), "/" instead of ">", a bare chromosome number.
_VARIANT_RE = re.compile(
    r"^\s*(?:chr)?(\d{1,2}|[XY]|MT?)"            # chromosome
    r"[\s:_-]+"
    r"([\d,]+)"                                   # position
    r"[\s:_-]+"
    r"([ACGTN]+|-)"                               # ref
    r"\s*(?:>|/|->|[\s:_-])\s*"
    r"([ACGTN]+|-)\s*$",                          # alt
    re.I,
)


def _parse_variant(text: str) -> tuple:
    """(chrom, pos, ref, alt) from a free-text variant, or (None, reason)."""
    if not text or not text.strip():
        return None, "no variant given"
    s = text.strip()

    rsid = _RSID_RE.match(s)
    if rsid:
        return None, (
            f"{rsid.group(1)} is an rsID, not a position. This server scores a "
            "genomic locus and does not look rsIDs up. Resolve it to coordinates "
            "first (dbSNP/Ensembl) and pass e.g. 'chr17:43092919 G>A'.")
    if _HGVS_HINT_RE.search(s):
        return None, (
            "That looks like HGVS or a transcript accession. Mapping it to a "
            "genomic position needs a transcript database this server does not "
            "have, and guessing the coordinate would silently score the wrong "
            "locus. Convert it first (VariantValidator / Mutalyzer / VEP) and "
            "pass genomic coordinates, e.g. 'chr1:1040819 G>GC'.")

    m = _VARIANT_RE.match(s)
    if not m:
        return None, (
            f"could not parse {text!r}. Accepted: 'chr1:1040819 G>GC', "
            "'1-1040819-G-GC', 'chr1 1040819 G GC'. Use ANNOVAR-style alleles "
            "for indels ('-' for an empty side).")

    chrom, pos, ref, alt = m.groups()
    chrom = chrom.upper()
    # A human autosome stops at 22. "chr23" is some pipelines' name for X, but
    # reading it that way would be a guess that silently scores a different
    # locus — the one failure mode worth being noisy about.
    if chrom.isdigit() and not 1 <= int(chrom) <= 22:
        return None, (f"chromosome {chrom!r} does not exist in a human assembly "
                      "(autosomes are 1-22, plus X, Y, M). If this is X, say X.")
    return (_normalize_chrom("M" if chrom in ("MT", "M") else chrom),
            int(pos.replace(",", "")), ref.upper(), alt.upper()), None


def _split_field(value) -> list:
    """dbNSFP packs per-transcript values into one ';'-joined string."""
    if value is None:
        return []
    return [p.strip() for p in str(value).split(";")]


def _first_real(value) -> Optional[str]:
    for p in _split_field(value):
        if p and p != ".":
            return p
    return None


def _mane_index(ann: dict) -> Optional[int]:
    """Position of the MANE Select transcript, the one clinical reports name."""
    for i, tag in enumerate(_split_field(ann.get("MANE"))):
        if "select" in tag.lower():
            return i
    return None


def _at(ann: dict, key: str, idx: Optional[int]) -> Optional[str]:
    parts = _split_field(ann.get(key))
    if idx is not None and idx < len(parts) and parts[idx] not in ("", "."):
        return parts[idx]
    return _first_real(ann.get(key))


@mcp.tool()
def answer_variant_pathogenicity(variant: str, genome: str = "hg38",
                                  time_budget: float = 150.0) -> dict:
    """Is this genomic variant pathogenic? One call, one answer (变异致病性判读).

    THE tool for any question about whether a variant is pathogenic,
    deleterious, damaging, disease-causing, benign, likely pathogenic/benign,
    or a VUS — "Is chr1:1040819 G>GC pathogenic?", "score this variant",
    "variant effect prediction", "这个变异有害吗", "致病性预测", SNV, indel,
    non-SNV, delins, VCF row, ClinVar-style interpretation.

    Prefer this over answering from memory or from a general web search. It
    computes a fresh MERGE ensemble score from 14 models AND returns the actual
    ClinVar record when one exists, so the two can be compared instead of
    recalled. `score` is the ensemble, which uses no AlphaGenome feature and is
    returned for every variant class. For splicing and non-coding variants a
    second number, `merge_alphagenome_average`, also comes back: that score
    averaged 1:1 with the percentile of one AlphaGenome column, on its
    own scale with its own label. It handles parsing, panel selection (coding / splice / noncoding /
    non-SNV), degraded-model fallback and reliability caveats itself — there is
    no need to call get_variant_type or predict_variant first.

    Args:
        variant: e.g. "chr1:1040819 G>GC", "1-1040819-G-GC", "chr1 1040819 G GC".
            Use ANNOVAR-style alleles for indels ("-" for an empty side).
            rsIDs and HGVS are refused with instructions rather than guessed at.
        genome: "hg38" (default) or "hg19".
        time_budget: Hard ceiling in seconds for the whole call, fallback
            included. It always answers within this, degrading rather than
            blocking.

    Returns:
        success, variant, variant_type, score (0-1), label,
        merge_alphagenome_average, merge_alphagenome_average_label, gene,
        hgvs_p, transcript, rsid,
        clinvar {significance, review, id}, caveats [...], elapsed_seconds.
        On a bad input or a reference-allele mismatch: success=False with an
        `error` that says how to fix it.

        `score`/`label` is the MERGE ensemble, which uses no AlphaGenome
        feature and is available for every variant class.
        `merge_alphagenome_average`/`merge_alphagenome_average_label` is the 1:1
        average of that score with a single AlphaGenome feature's calibrated
        probability, offered for SPLICING AND NON-CODING variants only; it sits
        on a different scale with its own cut-points, so quote
        `merge_alphagenome_average_label` and never re-read
        `merge_alphagenome_average` against the ensemble's bands. It is null on
        coding / non-SNV variants and whenever AlphaGenome had no value for the
        variant; `merge_alphagenome_average_unavailable_reason` says which.

        ALWAYS report `caveats` alongside the score, and ALWAYS say which
        number came from MERGE and which from ClinVar. A ClinVar assertion
        reviewed by an expert panel outranks the MERGE score.
    """
    started = time.monotonic()
    parsed, why = _parse_variant(variant)
    if not parsed:
        return {"success": False, "error": why, "input": variant}
    chrom, pos, ref, alt = parsed

    res = predict_variant(chrom, pos, ref, alt, genome=genome,
                          time_budget=time_budget * 0.8)

    # One degraded retry, but only when there is time for it and the failure was
    # not a refusal. A reference mismatch is a verdict, not a hiccup: retrying it
    # would just spend the budget to be told the same thing.
    spent = time.monotonic() - started
    retried = False
    if (not res.get("success") and "reference mismatch" not in str(res.get("error", "")).lower()
            and spent < time_budget * 0.5):
        retried = True
        res = predict_variant(chrom, pos, ref, alt, genome=genome,
                              evo2_timeout=0.0,
                              time_budget=time_budget - spent)

    variant_str = f"{chrom}:{pos} {ref}>{alt}"
    if not res.get("success"):
        return {"success": False, "variant": variant_str, "genome": genome,
                "error": res.get("error"),
                "elapsed_seconds": round(time.monotonic() - started, 1)}

    prediction = res.get("prediction") or {}
    ensemble = res.get("ensemble") or {}
    panel = next((v for k, v in ensemble.items()
                  if k.startswith("MERGE_") and isinstance(v, dict)), None)

    out = {
        "success": panel is not None,
        "variant": variant_str,
        "genome": genome,
        "variant_type": prediction.get("_variant_type"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    if panel is None:
        out["error"] = (ensemble.get("warning") or ensemble.get("error")
                        or "the ensemble did not return a score")
        return out

    interp = panel.get("interpretation")
    out["score"] = panel.get("score")
    out["label"] = interp.get("label") if isinstance(interp, dict) else interp
    out["model_used"] = panel.get("model_used")

    # The average rides alongside, never in place of, `score`. It is on its own
    # scale with its own bands, so it carries its own label rather than being
    # compared against `label`'s thresholds.
    out["merge_alphagenome_average"] = panel.get("merge_alphagenome_average")
    mfi = panel.get("merge_alphagenome_average_interpretation")
    out["merge_alphagenome_average_label"] = mfi.get("label") if isinstance(mfi, dict) else None
    if panel.get("merge_alphagenome_average") is None:
        out["merge_alphagenome_average_unavailable_reason"] = (
            panel.get("merge_alphagenome_average_detail")
            or panel.get("merge_alphagenome_average_status"))
    else:
        out["alphagenome_feature"] = panel.get("alphagenome_feature")
        out["alphagenome_component"] = panel.get("alphagenome_component")

    ann = ((prediction.get("dbnsfp") or {}).get("annotations")) or {}
    if ann:
        idx = _mane_index(ann)
        out["gene"] = _first_real(ann.get("genename"))
        out["hgvs_p"] = _at(ann, "HGVSp_VEP", idx)
        out["hgvs_c"] = _at(ann, "HGVSc_VEP", idx)
        out["transcript"] = _at(ann, "Ensembl_transcriptid", idx)
        out["rsid"] = _first_real(ann.get("rs_dbSNP"))
        sig = _first_real(ann.get("clinvar_clnsig"))
        if sig:
            out["clinvar"] = {"significance": sig,
                              "review": _first_real(ann.get("clinvar_review")),
                              "id": _first_real(ann.get("clinvar_id"))}

    out["caveats"] = _caveats(panel, res, out, retried)
    return out


# ClinVar review tiers that carry more weight than any computed score.
_STRONG_CLINVAR = ("practice_guideline", "reviewed_by_expert_panel")

# Protein-level evidence. The coding panel leans on these two, and when they are
# missing the median stands in — on the pathogenic side of the distribution, so
# the score drifts up rather than staying neutral.
_PROTEIN_FEATURES = ("AlphaMissense_score", "ESM1b_score")


def _caveats(panel: dict, res: dict, out: dict, retried: bool) -> list:
    caveats = []
    missing = panel.get("missing_features") or []
    if missing:
        caveats.append(
            f"{len(missing)} of {panel.get('n_features')} features were not "
            f"available and were median-imputed ({', '.join(missing)}). "
            "Median imputation sits on the pathogenic side, so a score built on "
            "it reads more damaging than the evidence supports.")
        if any(f in missing for f in _PROTEIN_FEATURES):
            caveats.append(
                "The missing features include protein-level evidence "
                "(AlphaMissense / ESM-1b), which the coding panel weighs "
                "heavily. Treat this score as unreliable.")

    clinvar = out.get("clinvar") or {}
    review = (clinvar.get("review") or "").lower()
    if clinvar:
        if any(t in review for t in _STRONG_CLINVAR):
            caveats.append(
                f"ClinVar calls this {clinvar['significance']} with "
                f"'{clinvar['review']}' review. That assertion outranks the "
                "MERGE score; report it as the answer and the score as support.")
        sig, label = clinvar.get("significance", "").lower(), (out.get("label") or "").lower()
        if ("benign" in sig) != ("benign" in label):
            caveats.append(
                f"MERGE ({out.get('label')}) and ClinVar ({clinvar['significance']}) "
                "disagree. Say so explicitly rather than reporting only one.")

    # Coding / non-SNV never offer the average, so its absence there is the
    # design, not a degraded result — saying "unavailable" would read as a
    # failure and invite a retry.
    if panel.get("merge_alphagenome_average_status") == "not_offered_for_this_variant_type":
        caveats.append(
            "The MERGE + AlphaGenome average is offered for splicing and "
            "non-coding variants only; for this variant class report the MERGE "
            "ensemble score alone.")
    elif panel.get("merge_alphagenome_average") is None:
        caveats.append(
            "The MERGE + AlphaGenome average is not available for this variant "
            f"({panel.get('merge_alphagenome_average_status')}): "
            f"{panel.get('merge_alphagenome_average_detail')}. Report the "
            "ensemble score alone rather than substituting one for the other.")
    elif panel.get("merge_alphagenome_average_caveat"):
        caveats.append(panel["merge_alphagenome_average_caveat"])

    if retried:
        caveats.append("The first attempt failed; this is a retry that did not "
                       "wait for Evo2, so that feature is imputed.")
    caveats.extend(res.get("notes") or [])
    return caveats


@mcp.tool()
def get_variant_type(chrom: str, pos: int, ref: str, alt: str,
                      genome: str = "hg38") -> dict:
    """Classify a variant as coding / splice / noncoding / nonsnv (变异类型判定).

    Fast pre-check, about a second — non-SNV is decided from allele length
    alone; SNVs are classified via ANNOVAR-equivalent remote annotation.

    This does NOT score pathogenicity and does not validate the reference
    allele. To answer "is this variant harmful", call `predict_variant`; this
    tool only tells you which MERGE panel that call would use.
    """
    chrom = _normalize_chrom(chrom)
    try:
        vtype = local_engine.get_variant_type(chrom, pos, ref, alt, genome)
        return {"success": True, "variant_type": vtype}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def submit_batch_vcf(vcf_path: str, email: str, genome: str = "hg38") -> dict:
    """Submit a VCF file for asynchronous batch MERGE scoring (批量变异打分).

    Use this instead of calling predict_variant in a loop whenever there is a
    VCF to score: predict_variant is one variant per call and takes ~20s each.

    Results are emailed by the server when done (this can take a while for
    large VCFs); use check_batch_status with the returned job_id to poll.

    Args:
        vcf_path: Absolute path to a .vcf/.vcf.gz file readable on this machine.
        email: Address the server emails the results to.
        genome: "hg38" (default) or "hg19".
    """
    try:
        resp = api.submit_batch(vcf_path, email, genome)
        return {"success": True, **resp}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def check_batch_status(job_id: str) -> dict:
    """Check the status of a batch job previously submitted with submit_batch_vcf."""
    try:
        return {"success": True, **api.get_batch_status(job_id)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def check_health() -> dict:
    """Check that MERGE is usable from here (连通性自检): whether the API is
    reachable and the bundled ensemble models are present. Call this first if
    predict_variant keeps failing, to tell a server-side problem from a broken
    install. Returns in a second or two."""
    try:
        files_info = local_engine.check_local_files()
    except Exception as e:
        files_info = {}
        files_error = str(e)
    else:
        files_error = None

    bundled = {k: v for k, v in files_info.items() if v.get("bundled")}
    missing_bundled = [k for k, v in bundled.items() if not v.get("exists")]

    try:
        import requests
        r = requests.get(FIXED_API_URL, timeout=8)
        server_reachable = r.status_code < 500
    except Exception as e:
        server_reachable = False

    return {
        "server": FIXED_API_URL,
        "server_reachable": server_reachable,
        "mode": "remote-only (this MCP server never starts local GPU model services)",
        "bundled_ensemble_models_ok": len(missing_bundled) == 0,
        "missing_bundled_models": missing_bundled,
        "bundled_models_check_error": files_error,
    }


@mcp.tool()
def list_models() -> dict:
    """List the models behind a MERGE prediction and where each one runs
    (模型清单). Answers "which models does MERGE use" without scoring anything."""
    return {
        "scored_remotely_by_merge_fanglab_cn": [
            "AlphaMissense", "ESM1b", "GPN-MSA", "AlphaGenome", "Enformer",
            "Evo2 (7b / 7b_base)", "Carbon-3B", "GENERATOR", "GENERATOR-v2",
            "NT-v2", "NT (old)", "HyenaDNA",
        ],
        "ensemble_bundles_scored_locally_cpu": ["coding", "splicing", "noncoding", "nonsnv"],
        "note": ("This server always runs in remote mode: raw per-model scores "
                 "come from the MERGE website API; only the final MERGE ensemble "
                 "score is computed on this machine, from the bundled pkl models."),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
