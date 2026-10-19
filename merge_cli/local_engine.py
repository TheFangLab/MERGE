"""
merge_cli/local_engine.py
Local mode: conda environments and model services for on-device scoring, the bundled
models and the local ensemble.
"""
from __future__ import annotations

import importlib.resources
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

from .config import FIXED_API_URL, get_local_config, set_local_config
from .api import _session as _remote_session

logger = logging.getLogger(__name__)

# ─── GPU Detection ────────────────────────────────────────────
def detect_gpu() -> dict:
    gpus, vram = [], []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                timeout=5, text=True,
            ).strip()
            for line in out.splitlines():
                if "," in line:
                    name, mem = line.rsplit(",", 1)
                    gpus.append(name.strip())
                    try:
                        vram.append(round(float(mem.strip()) / 1024, 1))
                    except ValueError:
                        vram.append(None)
        except Exception:
            pass
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append(torch.cuda.get_device_name(i))
                    props = torch.cuda.get_device_properties(i)
                    vram.append(round(props.total_memory / 1073741824, 1))
        except Exception:
            pass
    return {
        "has_gpu": bool(gpus),
        "gpus": gpus,
        "vram_gb": vram,
        "recommendation": "local" if gpus else "remote",
    }


# ─── Bundled Ensemble Model Configuration ─────────────────────
# The ensembles carry no alphagenome_* feature, so no AlphaGenome output went
# into training them. The AlphaGenome half of the MERGE + AlphaGenome average is
# a label-free percentile table (alphagenome_reference_quantiles.json), not a
# fitted model. Only splicing and non-coding use it -- coding and non-SNV do not
# offer the average. See MODEL_CARD.md.
_BUNDLED_MODELS = {
    "coding":                     "MERGE_noAlphaGenome_coding.pkl",
    "splicing":                   "MERGE_noAlphaGenome_splicing.pkl",
    "noncoding":                  "MERGE_noAlphaGenome_noncoding.pkl",
    "nonsnv":                     "MERGE_noAlphaGenome_nonsnv.pkl",
    "alphagenome_reference":      "alphagenome_reference_quantiles.json",
    "merge_final_thresholds":     "merge_final_thresholds.json",
    "ensemble_predict":           "ensemble_predict.py",
}

ENV_NAMES = {
    "evo2": "evo2",
    "hyenadna": "hyenadna",
    "nt": "nt",
    "alphagenome": "alphagenome",
    "enformer": "enformer",
    "carbon": "carbon",
    "generator": "generator",
    "generator_v2": "generator_v2",
    "nt_v2": "nt_v2",
}

# HuggingFace model ids for the DNA language models, matching the ones the
# MERGE feature tables were computed with. Changing these changes the score
# scale and silently invalidates the ensemble — keep them in sync with
# merge_feature_datasets/*/FEATURE_METHODS.md.
DNA_LM_MODELS = {
    "carbon":       "HuggingFaceBio/Carbon-3B",
    "generator":    "GenerTeam/GENERator-eukaryote-1.2b-base",
    "generator_v2": "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
    "nt_v2":        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
}

# Windowing / tokenisation per model, from FEATURE_METHODS.md. These MUST match
# training: carbon uses a 24576 bp window (divisible by 6 for its 6-mer
# tokenizer) and a SUM-delta score; generator/generator_v2/nt_v2 use 600 bp and
# max_length 128.
# `trust_model` mirrors how the training runs loaded each model: Carbon's
# tokenizer needs remote code but its model class does not (and asking for it
# breaks the load), while the others need it for both.
DNA_LM_PARAMS = {
    "carbon":       {"window_bp": 24576, "max_length": None, "kind": "carbon",  "multiple_of": 6,
                     "trust_model": False, "trust_tokenizer": True},
    "generator":    {"window_bp": 600,   "max_length": 128,  "kind": "causal",  "multiple_of": 1,
                     "trust_model": True,  "trust_tokenizer": True},
    "generator_v2": {"window_bp": 600,   "max_length": 128,  "kind": "causal",  "multiple_of": 1,
                     "trust_model": True,  "trust_tokenizer": True},
    "nt_v2":        {"window_bp": 600,   "max_length": 128,  "kind": "masked",  "multiple_of": 1,
                     "trust_model": True,  "trust_tokenizer": True},
}

# These models ship custom code written against transformers 4.x. On
# transformers 5.x, NT-v2 fails to load with
# "Could not import module 'EsmForMaskedLM'", so the env must be pinned.
# The MERGE feature tables were produced with transformers 4.57.
TRANSFORMERS_PIN = "transformers>=4.40,<5"

# AlphaGenome talks to a Google-hosted model, and the client version decides
# which scorers are requested — 0.7.0 exposes 19 RECOMMENDED scorers where
# 0.5.1 exposes fewer, which shifts the mean features by ~5-9%. Pinned to the
# version the MERGE server runs so local and remote mode agree.
#
# Note: this only aligns CLI with the server. Neither reproduces the values in
# the training tables, because the hosted model itself changed after they were
# computed (see TUTORIAL "AlphaGenome and reproducibility").
ALPHAGENOME_PIN = "alphagenome==0.5.1"

# Evo2 variants that local mode can deploy. evo2_1b_base needs FP8, i.e. a GPU
# with compute capability >= 8.9 (RTX 4000/5000 Ada, L40S, H100). On older cards
# the feature is left NaN and the ensemble pipeline median-imputes it.
EVO2_VARIANTS = {
    "evo2_7b":      {"repo": "arcinstitute/evo2_7b",      "needs_fp8": False},
    "evo2_7b_base": {"repo": "arcinstitute/evo2_7b_base", "needs_fp8": False},
    "evo2_1b_base": {"repo": "arcinstitute/evo2_1b_base", "needs_fp8": True},
}
EVO2_REPO_URL  = "https://github.com/arcinstitute/evo2"
EVO2_SRC_DIR   = str(Path.home() / ".merge-evo2-src")
EVO2_CACHE_DIR = str(Path.home() / ".merge-evo2-cache")

# Default model directory — override via set_local_config("evo2_model_dir", ...)
EVO2_DEFAULT_MODEL_DIR = str(Path.home() / ".merge-data" / "models" / "evo2_7b")


def get_bundled_model_path(filename: str, packaged_only: bool = False) -> Path:
    """Locate a bundled artefact, preferring the user's model_dir override.

    `packaged_only` disables that override. Use it for anything that is *code*
    rather than data: a stale ensemble_predict.py left in model_dir by an older
    install would otherwise shadow the packaged one and get executed, which is
    both a correctness problem — an older module has a different API entirely —
    and not something a data directory should be able to do.
    """
    if not packaged_only:
        cfg = get_local_config()
        user_path = Path(cfg["model_dir"]) / filename
        if user_path.exists():
            return user_path
    try:
        pkg_data = importlib.resources.files("merge_cli") / "data" / "models" / filename
        if pkg_data.is_file():
            return Path(str(pkg_data))
    except Exception:
        pass
    raise FileNotFoundError(f"Model file not found: {filename}")


def _cfg() -> dict:
    return get_local_config()


def _safe_float(val) -> Optional[float]:
    try:
        if val in (None, ".", "N/A", "", "NA", "-"):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _evo2_score_of(node) -> Optional[float]:
    """Evo2 services report the ref-minus-alt quantity as llr_score for SNVs and
    score_mean for non-SNVs; both mean the same thing."""
    if not isinstance(node, dict):
        return None
    for key in ("llr_score", "score_mean", "score"):
        if node.get(key) is not None:
            return node[key]
    return None


# ════════════════════════════════════════════════════════════════
#  Sequence Context Retrieval
# ════════════════════════════════════════════════════════════════

def get_sequence_context(chrom, pos, ref, alt, genome="hg38",
                         window=8192,
                         local_genome_path: Optional[str] = None):
    if local_genome_path:
        r, v = _fetch_local_fasta(chrom, pos, ref, alt, local_genome_path, window)
        if r:
            return r, v
    return _fetch_remote_seq(chrom, pos, ref, alt, genome, window)


def _fetch_remote_seq(chrom, pos, ref, alt, genome, window):
    try:
        r = _remote_session().post(
            f"{FIXED_API_URL}/api/sequence_context/",
            json={"chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                  "genome": genome, "window": window},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        return d.get("ref_seq"), d.get("var_seq")
    except Exception as exc:
        logger.warning(f"[remote/seq_ctx] {exc}")
        return None, None


# pysam is an optional dependency (the "local" extra). It is only needed to read
# a local genome FASTA, the dbNSFP / GPN-MSA tabix files and the precomputed VCF
# cache — remote mode and the MCP server never reach any of those. It also has no
# Windows wheels, which is why it is not in the base install. Every call site
# below either falls back to the remote equivalent or raises this hint.
_PYSAM_HINT = ("pysam is needed for local FASTA and tabix lookups but is not "
               "installed. Install it with:  pip install 'merge-cli[local]'")


def _import_pysam(required: bool = False):
    """Return the pysam module, or None when it is absent and optional."""
    try:
        import pysam
        return pysam
    except ImportError:
        if required:
            raise ImportError(_PYSAM_HINT) from None
        return None


def _fetch_local_fasta(chrom, pos, ref, alt, fasta_path, window):
    pysam = _import_pysam()
    if pysam is None:
        logger.warning(f"[local/fasta] {_PYSAM_HINT}")
        return None, None
    try:
        fa = pysam.FastaFile(fasta_path)
        chrom_q = str(chrom)
        if chrom_q not in fa.references:
            chrom_q = ("chr" + chrom_q if not chrom_q.startswith("chr")
                       else chrom_q.lstrip("chr"))
        if chrom_q not in fa.references:
            return None, None
        p = int(pos) - 1
        start = max(0, p - window // 2)
        end = min(fa.get_reference_length(chrom_q), p + window // 2)
        ref_seq = fa.fetch(chrom_q, start, end).upper()
        snv_pos = p - start
        if snv_pos >= len(ref_seq) or ref_seq[snv_pos] != ref.upper():
            return None, None
        var_seq = ref_seq[:snv_pos] + alt.upper() + ref_seq[snv_pos + 1:]
        return ref_seq, var_seq
    except Exception as exc:
        logger.warning(f"[local/fasta] {exc}")
        return None, None


# ─── dbNSFP / GPN-MSA Queries ──────────────────────────────────

def query_dbnsfp_local(chrom, pos, ref, alt, genome="hg38") -> dict:
    cfg = _cfg()
    build = "grch37" if genome == "hg19" else "grch38"
    fpath = os.path.join(cfg["data_dir"], f"dbNSFP5.3a_{build}.gz")
    # Check for the file before pysam: with no local dbNSFP there is nothing to
    # read anyway, and the remote fallback needs neither.
    if not os.path.exists(fpath):
        return _query_remote("dbnsfp", chrom=chrom, pos=pos, ref=ref, alt=alt, genome=genome)
    pysam = _import_pysam(required=True)

    _DBNSFP_COLS = {
        "chr": 0, "pos(1-based)": 1, "ref": 2, "alt": 3,
        "genename": 12, "Ensembl_transcriptid": 14,
        "ESM1b_score": 129, "AlphaMissense_score": 132,
    }
    chrom_q = chrom.lstrip("chr") if genome == "hg19" else chrom
    annotations, dl_models_list = [], []
    try:
        tbx = pysam.TabixFile(fpath)
        for row in tbx.fetch(chrom_q, pos - 1, pos):
            fields = row.split("\t")
            if len(fields) <= 3: continue
            if fields[2] != ref or fields[3] != alt: continue

            def _get(col):
                idx = _DBNSFP_COLS.get(col, -1)
                return fields[idx] if 0 <= idx < len(fields) else "."

            ann = {k: _get(k) for k in _DBNSFP_COLS if k not in ("ESM1b_score", "AlphaMissense_score")}
            annotations.append(ann)
            dl_models_list.append({
                "ESM1b":         {"score": _safe_float(_get("ESM1b_score"))},
                "AlphaMissense": {"score": _safe_float(_get("AlphaMissense_score"))},
                "Ensembl_transcriptid": _get("Ensembl_transcriptid"),
            })
    except Exception as exc:
        return {"error": str(exc), "annotations": [], "dl_models": []}

    if not annotations:
        return {"error": "No matching record found", "annotations": [], "dl_models": []}

    return {
        "annotations": annotations, "dl_models": dl_models_list,
        "transcripts": dl_models_list,
    }


def query_gpn_msa_local(chrom, pos, ref, alt) -> dict:
    fpath = os.path.join(_cfg()["data_dir"], "scores.tsv.bgz")
    if not os.path.exists(fpath):
        return _query_remote("gpn_msa", chrom=chrom, pos=pos, ref=ref, alt=alt)
    pysam = _import_pysam()
    if pysam is None:
        return _query_remote("gpn_msa", chrom=chrom, pos=pos, ref=ref, alt=alt)
    try:
        tbx = pysam.TabixFile(fpath)
        for row in tbx.fetch(chrom.lstrip("chr"), pos - 1, pos):
            fields = row.split("\t")
            if len(fields) >= 5 and fields[2] == ref and fields[3] == alt:
                return {"score": _safe_float(fields[4])}
    except Exception as exc:
        return {"score": None, "error": str(exc)}
    return {"score": None}


def _query_remote(endpoint: str, **kwargs) -> dict:
    try:
        r = _remote_session().post(f"{FIXED_API_URL}/api/{endpoint}/",
                                   json=kwargs, timeout=25)
        return r.json()
    except Exception as exc:
        return {"score": None, "error": f"Remote {endpoint} query failed: {exc}",
                "annotations": [], "dl_models": []}


# ─── ANNOVAR Variant Type Annotation ─────────────────────────

def annotate_variant_type_remote(chrom, pos, ref, alt, genome="hg38") -> str:
    # No annovar_path is sent: the server knows where its own ANNOVAR lives and
    # ignores the field entirely (verified against the live endpoint — a correct
    # path, an absent field and a nonexistent path all return the same answer).
    try:
        r = _remote_session().post(
            f"{FIXED_API_URL}/api/annotate_variant_type/",
            json={"chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                  "genome": genome},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("variant_type")
    except Exception as exc:
        logger.warning(f"[annovar/remote] Variant type annotation failed: {exc}")
        return None

def annotate_variant_type_local(chrom, pos, ref, alt, genome="hg38", annovar_dir: Optional[str] = None) -> str:
    cfg = _cfg()
    adir = annovar_dir or cfg.get("annovar_dir", "")
    if not adir or not os.path.exists(os.path.join(adir, "annotate_variation.pl")):
        return None
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            avinput = os.path.join(tmpdir, "query.avinput")
            with open(avinput, "w") as f:
                chrom_str = chrom if chrom.startswith("chr") else f"chr{chrom}"
                f.write(f"{chrom_str}\t{pos}\t{pos}\t{ref}\t{alt}\n")
            db = "hg19" if genome == "hg19" else "hg38"
            out_prefix = os.path.join(tmpdir, "out")
            cmd = [
                "perl", os.path.join(adir, "annotate_variation.pl"),
                "-geneanno", "-dbtype", "refGene",
                "-buildver", db,
                avinput, os.path.join(adir, "humandb"),
                "--outfile", out_prefix,
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            exonic_f = out_prefix + ".exonic_variant_function"
            variant_f = out_prefix + ".variant_function"
            if os.path.exists(exonic_f):
                with open(exonic_f) as f:
                    content = f.read().lower()
                    if "splicing" in content: return "splice"
                    if content.strip(): return "coding"
            if os.path.exists(variant_f):
                with open(variant_f) as f:
                    content = f.read().lower()
                    if "splicing" in content: return "splice"
                    if any(k in content for k in ("exonic", "utr", "ncrna")): return "noncoding"
            return "noncoding"
    except Exception as exc:
        logger.warning(f"[annovar/local] Failed: {exc}")
        return None

def is_non_snv(ref, alt) -> bool:
    """non-SNV = indel / MNV / delins (either allele not a single base, or '-')."""
    if not ref or not alt:
        return False
    ref, alt = str(ref).strip(), str(alt).strip()
    if ref == "-" or alt == "-":
        return True
    return len(ref) != 1 or len(alt) != 1


def get_variant_type(chrom, pos, ref, alt, genome="hg38", ensemble_type: Optional[str] = None) -> str:
    if ensemble_type: return ensemble_type
    # Allele length decides non-SNV before ANNOVAR is consulted: ANNOVAR would
    # happily label an indel 'exonic', but the MERGE ensemble scores every
    # non-SNV with its own dedicated model.
    if is_non_snv(ref, alt):
        return "nonsnv"
    vtype = annotate_variant_type_remote(chrom, pos, ref, alt, genome)
    if vtype: return vtype
    vtype = annotate_variant_type_local(chrom, pos, ref, alt, genome)
    return vtype or "coding"


# ════════════════════════════════════════════════════════════════
#  Local Inference Services and Conda Detection
# ════════════════════════════════════════════════════════════════

def check_disk_space(path: str, required_gb: float) -> tuple:
    try:
        st = shutil.disk_usage(path)
        free_gb = st.free / (1024 ** 3)
        return free_gb >= required_gb, round(free_gb, 1)
    except Exception:
        return True, -1

_ENV_DISK_GB = {
    "hyenadna": 8, "nt": 8, "evo2": 35, "alphagenome": 2, "enformer": 20,
    # Each DNA-LM env carries its own torch build plus the weights.
    "carbon": 24, "generator": 16, "generator_v2": 16, "nt_v2": 12,
}

# Rough peak VRAM per model at its fixed training window, measured on an
# RTX 5880 Ada. Carbon is the heavy one: 24576 bp is ~4096 six-mer tokens.
_MODEL_VRAM_GB = {
    "carbon": 20, "generator": 6, "generator_v2": 6, "nt_v2": 3,
    "nt": 3, "hyenadna": 2, "evo2": 20,
}

_SERVICE_PORTS = {
    "alphagenome":    5000,
    "hyenadna":       5001,
    "nt":             5002,
    "evo2":           5003,
    "enformer":       5004,
    "carbon":         5005,
    "generator":      5006,
    "generator_v2":   5007,
    "nt_v2":          5008,
}

_LOCAL_SERVICES = {
    "hyenadna":    {"url_key": "hyenadna_url",    "default": "http://localhost:5001"},
    "nt":          {"url_key": "nt_url",          "default": "http://localhost:5002"},
    "evo2":        {"url_key": "evo2_url",        "default": "http://localhost:5003"},
    "alphagenome": {"url_key": "alphagenome_url", "default": "http://localhost:5000"},
    "enformer":    {"url_key": "enformer_url",    "default": "http://localhost:5004"},
    "carbon":       {"url_key": "carbon_url",       "default": "http://localhost:5005"},
    "generator":    {"url_key": "generator_url",    "default": "http://localhost:5006"},
    "generator_v2": {"url_key": "generator_v2_url", "default": "http://localhost:5007"},
    "nt_v2":        {"url_key": "nt_v2_url",        "default": "http://localhost:5008"},
}

# Models the CLI can deploy and run entirely locally.
DEPLOYABLE_MODELS = ["evo2", "hyenadna", "nt", "alphagenome", "enformer",
                     "carbon", "generator", "generator_v2", "nt_v2"]

# Every service shown by `merge local env status` / `merge doctor`.
ALL_SERVICES = list(DEPLOYABLE_MODELS)

def check_service_health(model_name: str) -> dict:
    svc = _LOCAL_SERVICES.get(model_name)
    if not svc: return {"online": False, "error": f"Unknown model: {model_name}"}
    base_url = _cfg().get(svc["url_key"], svc["default"])
    try:
        t0 = time.time()
        session = requests.Session()
        session.trust_env = False
        r = session.get(f"{base_url}/health", timeout=5)
        lat = (time.time() - t0) * 1000
        return ({"online": True, "latency_ms": round(lat, 1), "url": base_url}
                if r.status_code == 200 else {"online": False, "error": f"HTTP {r.status_code}", "url": base_url})
    except requests.exceptions.ConnectionError:
        return {"online": False, "error": "Service not running", "url": base_url}
    except Exception as exc:
        return {"online": False, "error": str(exc), "url": base_url}

def check_docker_health(model_name: str) -> dict:
    return check_service_health(model_name)

def call_local_service(model_name: str, chrom, pos, ref, alt, genome="hg38",
                       timeout=180, extra: Optional[dict] = None) -> dict:
    svc = _LOCAL_SERVICES.get(model_name)
    base_url = _cfg().get(svc["url_key"], svc["default"])
    payload = {"chrom": chrom, "pos": pos, "ref": ref,
               "alt": alt, "genome": genome, "genome_version": genome}
    if extra:
        payload.update(extra)
    try:
        session = requests.Session()
        session.trust_env = False
        r = session.post(f"{base_url}/predict", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": (f"{model_name} service not running ({base_url}).\n"
                          f"Run first: merge local env start --model {model_name}")}
    except requests.exceptions.Timeout:
        return {"error": f"{model_name} timed out (>{timeout}s) — model may still be loading, try again shortly"}
    except Exception as exc:
        return {"error": f"{model_name} call failed: {exc}"}



# ─── Conda Environment Core Utilities ───────────────────────────

def _conda_roots() -> list:
    """Likely conda installation roots, most specific first.

    `conda` is frequently absent from PATH in non-interactive shells (ssh
    commands, cron, CI), which used to make every environment look missing.
    These roots let the environment lookup fall back to the filesystem.
    """
    roots = []
    for var in ("CONDA_ROOT", "CONDA_PREFIX_1", "MAMBA_ROOT_PREFIX"):
        v = os.environ.get(var)
        if v:
            roots.append(v)
    exe = os.environ.get("CONDA_EXE")  # <root>/bin/conda
    if exe:
        roots.append(str(Path(exe).parent.parent))
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        # Inside an activated env the root is two levels up (<root>/envs/<name>).
        roots.append(str(p.parent.parent) if p.parent.name == "envs" else prefix)
    home = Path.home()
    roots += [str(home / n) for n in
              ("miniconda3", "anaconda3", "miniforge3", "mambaforge", "micromamba")]
    roots += ["/opt/conda", "/usr/local/conda"]

    seen, out = set(), []
    for r in roots:
        if r and r not in seen and os.path.isdir(os.path.join(r, "envs")):
            seen.add(r)
            out.append(r)
    return out


def _conda_envs_from_disk() -> dict:
    """Environments discovered by scanning <root>/envs for a Python binary."""
    envs = {}
    for root in _conda_roots():
        envs_dir = os.path.join(root, "envs")
        try:
            names = os.listdir(envs_dir)
        except Exception:
            continue
        for name in names:
            prefix = os.path.join(envs_dir, name)
            for rel in ("bin/python", "python.exe", "Scripts/python.exe"):
                if os.path.exists(os.path.join(prefix, rel)):
                    envs.setdefault(name, prefix)
                    break
    return envs


def _conda_env_map() -> dict[str, str]:
    # Ask conda when it is reachable, but always merge in what is on disk so a
    # missing `conda` on PATH does not hide working environments.
    envs = dict(_conda_envs_from_disk())
    envs.update(_conda_env_map_via_cli())
    return envs


def _conda_env_map_via_cli() -> dict[str, str]:
    for cmd in (["conda", "env", "list", "--json"], ["conda", "info", "--envs", "--json"]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not result.stdout.strip(): continue
            data = json.loads(result.stdout)
            envs = {}
            for prefix in data.get("envs", []):
                env_name = Path(prefix).name
                if env_name: envs[env_name] = prefix
            if envs: return envs
        except Exception: continue

    try:
        out = subprocess.check_output(["conda", "env", "list"], text=True, timeout=15)
    except Exception: return {}

    envs = {}
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"): continue
        parts = [p for p in line.split() if p != "*"]
        if not parts: continue
        prefix = parts[-1]
        if os.path.sep in prefix or "/" in prefix or "\\" in prefix:
            env_name = Path(prefix).name
            if env_name: envs[env_name] = prefix
            continue
        envs[parts[0]] = parts[0]
    return envs

def _get_conda_env_prefix(name: str) -> Optional[str]:
    return _conda_env_map().get(name)

def _get_conda_env_python(name: str) -> Optional[str]:
    prefix = _get_conda_env_prefix(name)
    if not prefix: return None
    candidates = [
        Path(prefix) / "python.exe",
        Path(prefix) / "bin" / "python",
        Path(prefix) / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists(): return str(candidate)
    return None

def _run_python_in_conda_env(env_name: str, args: list[str], **kwargs):
    python_exe = _get_conda_env_python(env_name)
    if not python_exe: raise FileNotFoundError(f"Cannot find Python for conda environment '{env_name}'")
    return subprocess.run([python_exe] + list(args), **kwargs)

def _popen_python_in_conda_env(env_name: str, args: list[str], **kwargs):
    python_exe = _get_conda_env_python(env_name)
    if not python_exe: raise FileNotFoundError(f"Cannot find Python for conda environment '{env_name}'")
    return subprocess.Popen([python_exe] + list(args), **kwargs)

def _conda_env_exists(name: str) -> bool:
    return _get_conda_env_prefix(name) is not None


# ─── Conda Environment Deployment ───────────────────────────────

def setup_model_env(model_name: str, console=None, local_model_path: Optional[str] = None, model_size: str = "7b") -> bool:
    funcs = {
        "evo2":        _setup_evo2,
        "hyenadna":    _setup_hyenadna,
        "nt":          _setup_nt,
        "alphagenome": _setup_alphagenome,
        "enformer":    _setup_enformer,
        "carbon":       lambda **kw: _setup_dna_lm("carbon", **kw),
        "generator":    lambda **kw: _setup_dna_lm("generator", **kw),
        "generator_v2": lambda **kw: _setup_dna_lm("generator_v2", **kw),
        "nt_v2":        lambda **kw: _setup_dna_lm("nt_v2", **kw),
    }
    fn = funcs.get(model_name)
    if not fn:
        if console: console.print(f"[red]✗ Unknown model: {model_name}[/red]")
        return False
    return fn(console=console, local_model_path=local_model_path, model_size=model_size)


def _setup_enformer(console=None, local_model_path: Optional[str] = None, **_) -> bool:
    """Deploy the Enformer environment.

    Enformer is the one TensorFlow model in the set, so it gets its own env with
    tensorflow/tensorflow_hub/kipoiseq rather than torch. It also needs a
    normalisation transform pickle next to the weights — without it the raw
    track deltas are on a different scale than the ensemble expects.
    """
    def _log(m):
        if console: console.print(m)
        else: print(m)

    _log("\n[bold cyan]═══ Enformer Environment Setup ═══[/bold cyan]")
    _log("  Scoring: alt-minus-ref per track over a 393,216 bp window, "
         "normalised, reduced to 7 summary statistics.")

    if not _create_conda_env("enformer", "3.10", console):
        return False

    pkgs = ["tensorflow", "tensorflow_hub", "kipoiseq", "pyfaidx", "joblib",
            "numpy", "pandas", "fastapi", "uvicorn[standard]"]
    if not _pip_install("enformer", pkgs, console, timeout=2400):
        return False
    _log("  [green]✓[/green] Dependencies installed")

    cfg = _cfg()
    model_path = local_model_path or cfg.get("enformer_model_path")
    if model_path:
        set_local_config("enformer_model_path", model_path)
        _log(f"  [green]✓[/green] Model path: {model_path}")
    else:
        _log(
            "\n[yellow]⚠ Enformer weights are not bundled and are not on PyPI.[/yellow]\n"
            "  Point the CLI at a SavedModel directory (or a TF-Hub URL):\n"
            "    [bold]merge enformer configure --model-path /path/to/enformer_model \\\n"
            "        --transform-path /path/to/enformer.transform.pkl[/bold]\n"
            "  TF-Hub: https://tfhub.dev/deepmind/enformer/1\n"
            "  The transform pickle is the one used to normalise the variant\n"
            "  scores; the MERGE features were computed with it, so scores\n"
            "  produced without it will not match the ensemble."
        )

    _log("\n[bold green]✓ Enformer environment ready![/bold green]")
    _log(f"  Next: [bold]merge local env start --model enformer[/bold]  "
         f"(port {_SERVICE_PORTS['enformer']})")
    return True


def _setup_dna_lm(model_name: str, console=None, local_model_path: Optional[str] = None,
                  model_size: str = "7b", **_) -> bool:
    """Deploy one DNA language model (carbon / generator / generator_v2 / nt_v2)
    into its own conda environment, mirroring the Evo2 deployment style: one env,
    one generated FastAPI service, one port.

    Each model gets its own environment because their `trust_remote_code`
    implementations pin different transformers versions.
    """
    def _log(m):
        if console: console.print(m)
        else: print(m)

    repo = DNA_LM_MODELS[model_name]
    params = DNA_LM_PARAMS[model_name]
    env = ENV_NAMES[model_name]

    _log(f"\n[bold cyan]═══ {model_name} Environment Setup ═══[/bold cyan]")
    _log(f"  Model: [bold]{repo}[/bold]")
    _log(f"  Scoring: {params['kind']}, window {params['window_bp']} bp"
         + (f", max_length {params['max_length']}" if params["max_length"] else ""))
    need_vram = _MODEL_VRAM_GB.get(model_name)
    if need_vram:
        _log(f"  Needs about [bold]{need_vram} GB[/bold] of VRAM at this window.")
        vram = max(detect_gpu().get("vram_gb") or [0])
        if vram and vram < need_vram:
            _log(f"  [yellow]⚠ Largest GPU has {vram} GB. This model will likely OOM here.\n"
                 f"    The window is fixed by the training setup and cannot be reduced without\n"
                 f"    putting the feature on a different scale, so if it does not fit, leave\n"
                 f"    {model_name} off — the MERGE ensemble median-imputes missing features.[/yellow]")

    need_gb = _ENV_DISK_GB.get(model_name, 16)
    ok, free = check_disk_space(str(Path.home()), need_gb)
    if not ok:
        _log(f"  [yellow]⚠ Low disk space: {free} GB free, ~{need_gb} GB recommended. Continuing anyway.[/yellow]")

    if not _create_conda_env(env, "3.11", console):
        return False

    pkgs = ["torch", TRANSFORMERS_PIN, "accelerate", "sentencepiece", "protobuf",
            "pyfaidx", "numpy", "pandas", "fastapi", "uvicorn[standard]"]
    if not _pip_install(env, pkgs, console, timeout=1800):
        return False
    _log(f"  [green]✓[/green] Dependencies installed ({TRANSFORMERS_PIN})")

    # Pre-download weights so the first prediction is not a surprise 10 GB pull.
    if local_model_path:
        _record_local_model_path(model_name, local_model_path, console)
    else:
        _log(f"  Downloading weights for {repo} (this can take a while)…")
        py = _get_conda_env_python(env)
        subprocess.run([py, "-m", "pip", "install", "huggingface_hub", "--quiet"], timeout=300)
        script = (
            "from huggingface_hub import snapshot_download; "
            f"p = snapshot_download(repo_id='{repo}'); print(p)"
        )
        r = subprocess.run([py, "-c", script], timeout=7200, capture_output=False)
        if r.returncode != 0:
            _log(f"  [yellow]⚠ Weight download failed. The service will retry on first use, "
                 f"or fetch manually: huggingface-cli download {repo}[/yellow]")
        else:
            _log(f"  [green]✓[/green] Weights cached")

    _log(f"\n[bold green]✓ {model_name} environment ready![/bold green]")
    _log(f"  Next: [bold]merge local env start --model {model_name}[/bold]  "
         f"(port {_SERVICE_PORTS[model_name]})")
    return True


def _create_conda_env(name: str, python_ver: str, console=None) -> bool:
    def _log(m):
        if console: console.print(m)
    if _conda_env_exists(name):
        _log(f"  [yellow]⚠[/yellow] conda environment '{name}' already exists, skipping creation")
        return True
    _log(f"  Creating conda environment {name} (python={python_ver})…")
    try:
        result = subprocess.run(
            ["conda", "create", "-n", name, f"python={python_ver}", "-y", "--quiet"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            _log(f"  [red]✗ conda create failed: {result.stderr[-500:]}[/red]")
            return False
        if not _conda_env_exists(name):
            _log(f"  [red]✗ Environment verification failed after creation — run manually: conda create -n {name} python={python_ver} -y[/red]")
            return False
        _log(f"  [green]✓[/green] Environment created successfully")
        return True
    except Exception as e:
        _log(f"  [red]✗ Creation failed: {e}[/red]")
        return False


def _pip_install(env_name: str, pkgs: list, console=None, timeout=900, extra_index: str = "") -> bool:
    def _log(m):
        if console: console.print(m)
    base_pkgs = ["requests", "urllib3"]
    all_pkgs = base_pkgs + [p for p in pkgs if p not in base_pkgs]
    cmd = ["-m", "pip", "install", "--quiet"] + all_pkgs
    if extra_index:
        cmd += ["-i", extra_index]
    try:
        result = _run_python_in_conda_env(env_name, cmd, capture_output=False, timeout=timeout)
        if result.returncode != 0:
            _log(f"  [red]✗ pip install failed (exit {result.returncode})[/red]")
            return False
        return True
    except Exception as e:
        _log(f"  [red]✗ pip install failed: {e}[/red]")
        return False


def _record_local_model_path(model_name: str, local_model_path, console=None) -> None:
    """Remember a local weights directory for a model, so the generated service
    loads from disk instead of HuggingFace."""
    if not local_model_path:
        return
    path = os.path.abspath(os.path.expanduser(local_model_path))
    set_local_config(f"{model_name}_model_path", path)
    msg = f"  [green]✓[/green] {model_name} will load weights from: {path}"
    if console:
        console.print(msg)
        if not os.path.isdir(path):
            console.print(f"  [yellow]⚠ That directory does not exist yet.[/yellow]")
    else:
        print(msg)


def _setup_hyenadna(console=None, local_model_path: Optional[str] = None, **_) -> bool:
    def _log(m):
        if console: console.print(m)
    _log("\n[bold cyan]═══ HyenaDNA Environment Setup ═══[/bold cyan]")
    if not _create_conda_env("hyenadna", "3.10", console): return False
    pkgs = ["torch", TRANSFORMERS_PIN, "accelerate", "biopython", "pysam", "pyfaidx",
            "fastapi", "uvicorn[standard]", "numpy", "pandas", "scikit-learn"]
    if not _pip_install("hyenadna", pkgs, console): return False
    _record_local_model_path("hyenadna", local_model_path, console)
    _log("[bold green]✓ HyenaDNA environment ready![/bold green]")
    return True


def _setup_nt(console=None, local_model_path: Optional[str] = None, **_) -> bool:
    def _log(m):
        if console: console.print(m)
    _log("\n[bold cyan]═══ Nucleotide Transformer Environment Setup ═══[/bold cyan]")
    if not _create_conda_env("nt", "3.10", console): return False
    pkgs = ["torch", TRANSFORMERS_PIN, "accelerate", "biopython", "pysam", "pyfaidx",
            "fastapi", "uvicorn[standard]", "numpy", "pandas", "scikit-learn", "tqdm"]
    if not _pip_install("nt", pkgs, console): return False
    _record_local_model_path("nt", local_model_path, console)
    _log("[bold green]✓ NT environment ready![/bold green]")
    return True


def _setup_alphagenome(console=None, **_) -> bool:
    """Deploy the AlphaGenome environment.

    AlphaGenome is not a local model: the package is a thin client for Google
    DeepMind's hosted API, so this environment only needs the client library and
    the user's own API key. Nothing is downloaded and nothing runs on your GPU.
    """
    def _log(m):
        if console: console.print(m)
        else: print(m)

    _log("\n[bold cyan]═══ AlphaGenome Environment Setup ═══[/bold cyan]")
    _log("  AlphaGenome runs on Google DeepMind's servers, not on your machine.\n"
         "  You need your own API key: [bold]https://deepmind.google.com/science/alphagenome[/bold]")

    if not _create_conda_env("alphagenome", "3.10", console):
        return False

    pkgs = [ALPHAGENOME_PIN, "flask", "pandas", "numpy"]
    if not _pip_install("alphagenome", pkgs, console, timeout=1200):
        _log("  [yellow]⚠ Could not install the 'alphagenome' package automatically.\n"
             "    Install it by hand, then re-run this command:\n"
             "      conda activate alphagenome && pip install alphagenome[/yellow]")
        return False
    _log(f"  [green]✓[/green] Client library installed ({ALPHAGENOME_PIN})")

    if _cfg().get("alphagenome_api_key"):
        _log("  [green]✓[/green] An API key is already configured.")
    else:
        _log(
            "\n[yellow]No API key configured yet.[/yellow] Add one with:\n"
            "  [bold]merge alphagenome configure --api-key YOUR_KEY[/bold]\n"
            "  (or set the ALPHAGENOME_API_KEY environment variable)"
        )

    _log("\n[bold green]✓ AlphaGenome environment ready![/bold green]")
    _log(f"  Next: [bold]merge local env start --model alphagenome[/bold]  "
         f"(port {_SERVICE_PORTS['alphagenome']})")
    return True


# ════════════════════════════════════════════════════════════════
#  _setup_evo2 — Full Dependency Installation Chain
# ════════════════════════════════════════════════════════════════

def _setup_evo2(console=None, local_model_path: Optional[str] = None, model_size: str = "7b") -> bool:
    """
    Full Evo2 conda environment deployment. Steps:
      1. Create Python 3.12 environment
      2. conda install cuda-nvcc / cuda-cudart-dev
      3. If not a 7b model, conda install transformer-engine-torch==2.3.0
      4. pip install psutil
      5. pip install flash-attn==2.8.0.post2 --no-build-isolation
      6. pip install evo2 (PyPI) or from local source
      7. Optionally download model weights via HuggingFace
    """
    def _log(m):
        if console:
            console.print(m)
        else:
            print(m)

    _log("\n[bold cyan]═══ Evo2 Environment Setup ═══[/bold cyan]")

    # ── Step 1: Create Python 3.12 environment ─────────────────
    _log("  [Step 1/6] Creating conda environment evo2 (python=3.12)…")

    # Check existing environment Python version; recreate if wrong version
    if _conda_env_exists("evo2"):
        existing_py = _get_conda_env_python("evo2")
        version_ok = False
        if existing_py:
            try:
                out = subprocess.check_output(
                    [existing_py, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
                ).strip()
                # Expect "Python 3.12.x"
                version_ok = out.startswith("Python 3.12")
            except Exception:
                pass
        if not version_ok:
            _log("  [yellow]⚠ Existing evo2 environment has wrong Python version (not 3.12) — removing and recreating…[/yellow]")
            subprocess.run(["conda", "env", "remove", "-n", "evo2", "-y", "--quiet"],
                           capture_output=False, timeout=120)
        else:
            _log("  [yellow]⚠[/yellow] conda environment 'evo2' already exists with Python=3.12, skipping creation")

    if not _create_conda_env("evo2", "3.12", console):
        return False

    python_exe = _get_conda_env_python("evo2")
    if not python_exe:
        _log("[red]✗ Cannot find Python interpreter for evo2 environment — please inspect your conda setup[/red]")
        return False

    def _run_pip(*args, timeout=1200):
        """Run pip command inside the evo2 conda environment, printing output to terminal."""
        return subprocess.run(
            [python_exe, "-m", "pip"] + list(args),
            timeout=timeout,
            capture_output=False,
        )

    def _run_conda(*args, timeout=900):
        return subprocess.run(
            ["conda", "run", "-n", "evo2"] + list(args),
            timeout=timeout,
            capture_output=False,
        )

    # ── Step 2: Install CUDA components ────────────────────────
    _log("  [Step 2/6] Installing cuda-nvcc / cuda-cudart-dev (this may take a few minutes)…")
    result = subprocess.run(
        ["conda", "install", "-n", "evo2", "-c", "nvidia",
         "cuda-nvcc", "cuda-cudart-dev", "-y", "--quiet"],
        capture_output=False, timeout=900,
    )
    if result.returncode != 0:
        _log("  [yellow]⚠ CUDA component installation returned non-zero — safe to ignore if CUDA is already installed system-wide, continuing…[/yellow]")
    else:
        _log("  [green]✓ CUDA components installed[/green]")

    # ── Step 3: Transformer Engine (required for FP8 models) ────
    if model_size not in ("7b", "7b_262k", "7b_base"):
        _log(f"  [Step 3/6] Installing transformer-engine-torch=2.3.0 (required for {model_size})…")
        if not gpu_supports_fp8():
            _log("  [yellow]⚠ No GPU with compute capability >= 8.9 detected. "
                 f"{model_size} needs FP8 (Ada / Hopper: RTX 4000-5000 Ada, L40S, H100).\n"
                 "    Installing anyway, but the model will fail to run on this machine — "
                 "its feature is median-imputed by the MERGE ensemble instead.[/yellow]")
        result = subprocess.run(
            ["conda", "install", "-n", "evo2", "-c", "conda-forge",
             "transformer-engine-torch=2.3.0", "-y", "--quiet"],
            capture_output=False, timeout=1200,
        )
        if result.returncode != 0:
            _log("  [red]✗ transformer-engine installation failed[/red]")
            return False
        _log("  [green]✓ Transformer Engine installed[/green]")
    else:
        _log(f"  [Step 3/6] Skipping Transformer Engine ({model_size} does not require FP8)")

    # ── Step 4: psutil + torch (torch must precede flash-attn) ──
    _log("  [Step 4/6] Installing psutil and PyTorch (required before flash-attn compilation)…")
    result = _run_pip("install", "psutil")
    if result.returncode != 0:
        _log("  [red]✗ psutil installation failed[/red]")
        return False
    _log("  [green]✓ psutil installed[/green]")

    # ── Step 4b: Install service dependencies ──────────────────
    _log("  Installing service dependencies: requests / fastapi / uvicorn…")
    result = _run_pip("install", "requests", "fastapi", "uvicorn[standard]")
    if result.returncode != 0:
        _log("  [red]✗ Service dependency installation failed[/red]")
        return False
    _log("  [green]✓ Service dependencies installed[/green]")

    # torch must be installed before flash-attn to avoid "No module named torch"
    _log("  Installing torch==2.7.1 (cu128)…")
    result = _run_pip(
        "install", "torch==2.7.1",
        "--index-url", "https://download.pytorch.org/whl/cu128",
        timeout=1800,
    )
    if result.returncode != 0:
        # Fallback to cu121 (compatible with CUDA 12.1)
        _log("  [yellow]⚠ cu128 installation failed, trying cu121 build…[/yellow]")
        result = _run_pip(
            "install", "torch==2.6.0",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            timeout=1800,
        )
    if result.returncode != 0:
        _log("  [red]✗ PyTorch installation failed — please install manually and retry[/red]")
        return False
    _log("  [green]✓ PyTorch installed[/green]")

    # ── Step 5: flash-attn ──────────────────────────────────────
    _log("  [Step 5/6] Installing flash-attn==2.8.0.post2 (compilation may take 5–15 minutes)…")
    result = _run_pip(
        "install", "flash-attn==2.8.0.post2",
        "--no-build-isolation",
        timeout=1800,
    )
    if result.returncode != 0:
        _log("  [red]✗ flash-attn installation failed! Common causes:[/red]")
        _log("    [yellow]1. CUDA version mismatch → confirm nvcc --version reports 12.1+[/yellow]")
        _log("    [yellow]2. GCC version too old → confirm gcc --version reports 9+[/yellow]")
        _log("    [yellow]3. Insufficient memory → compilation requires ~4 GB RAM[/yellow]")
        return False
    _log("  [green]✓ flash-attn installed[/green]")

    # ── Step 6: Install evo2 ────────────────────────────────────
    _log("  [Step 6/6] Installing evo2…")
    evo2_src = Path(EVO2_SRC_DIR)
    if evo2_src.exists() and (evo2_src / "setup.py").exists():
        _log(f"  Detected local source directory {EVO2_SRC_DIR}, using pip install -e .")
        result = _run_pip("install", "-e", str(evo2_src))
    else:
        _log("  Installing evo2 from PyPI…")
        result = _run_pip("install", "evo2")
    if result.returncode != 0:
        _log("  [red]✗ evo2 installation failed![/red]")
        _log("  [yellow]Alternative: install from source:[/yellow]")
        _log(f"    [dim]conda activate evo2[/dim]")
        _log(f"    [dim]git clone {EVO2_REPO_URL} {EVO2_SRC_DIR}[/dim]")
        _log(f"    [dim]pip install -e {EVO2_SRC_DIR}[/dim]")
        return False
    _log("  [green]✓ evo2 installed[/green]")

    # ── Optional: download model weights via HuggingFace ────────
    _log("\n[bold]Model Weights Download (optional)[/bold]")
    _log(f"  Current model directory: {_get_evo2_model_dir()}")
    _log(f"  To automatically download evo2_{model_size} (~14 GB), answer y")
    answer = _yn_prompt(f"Download evo2_{model_size} model weights via HuggingFace?")
    if answer:
        _download_evo2_via_huggingface(console=console, model_size=model_size)

    _log("\n[bold green]✓ Evo2 environment setup complete![/bold green]")
    _log("  Next step: merge local env start --model evo2")
    return True


def _get_evo2_model_dir(model_name: Optional[str] = None) -> str:
    """Effective weight directory for an Evo2 variant.

    A per-variant path registered under `evo2_model_dirs` wins; otherwise fall
    back to the single-model config (`evo2_model_dir`) for the active model, and
    finally to <data_dir>/<model_name>.
    """
    cfg = _cfg()
    active = cfg.get("evo2_model_name", "evo2_7b")
    model_name = model_name or active

    per_model = (cfg.get("evo2_model_dirs") or {}).get(model_name)
    if per_model:
        return per_model
    if model_name == active and cfg.get("evo2_model_dir"):
        return cfg["evo2_model_dir"]
    data_dir = cfg.get("data_dir", "")
    if data_dir:
        return os.path.join(data_dir, "models", model_name)
    return EVO2_DEFAULT_MODEL_DIR


def gpu_supports_fp8() -> bool:
    """evo2_1b_base needs FP8, i.e. compute capability >= 8.9 (Ada/Hopper)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            timeout=5, text=True,
        ).strip()
        for line in out.splitlines():
            try:
                if float(line.strip()) >= 8.9:
                    return True
            except ValueError:
                continue
    except Exception:
        pass
    return False


def available_evo2_variants() -> list:
    """Evo2 variants this machine can actually run, given its GPUs."""
    fp8 = gpu_supports_fp8()
    return [name for name, spec in EVO2_VARIANTS.items()
            if fp8 or not spec["needs_fp8"]]


def _download_evo2_via_huggingface(console=None, model_size: str = "7b") -> bool:
    """Download evo2 weights to the configured model directory via HuggingFace."""
    def _log(m):
        if console: console.print(m)
        else: print(m)

    model_id_map = {
        "7b":     "arcinstitute/evo2_7b",
        "7b_262k": "arcinstitute/evo2_7b_262k",
        "7b_base": "arcinstitute/evo2_7b_base",
        "1b_base": "arcinstitute/evo2_1b_base",
        "20b":    "arcinstitute/evo2_20b",
        "40b":    "arcinstitute/evo2_40b",
    }
    model_id = model_id_map.get(model_size, f"arcinstitute/evo2_{model_size}")

    python_exe = _get_conda_env_python("evo2")
    if not python_exe:
        _log("[red]✗ Cannot find evo2 environment Python[/red]")
        return False

    _log("  Installing huggingface_hub…")
    subprocess.run(
        [python_exe, "-m", "pip", "install", "huggingface_hub", "--quiet"],
        timeout=300,
    )

    # Determine download directory
    cfg = _cfg()
    model_name = f"evo2_{model_size}"
    download_dir = (cfg.get("evo2_model_dirs") or {}).get(model_name) \
        or str(Path.home() / ".merge-data" / "models" / model_name)
    os.makedirs(download_dir, exist_ok=True)

    # Register this variant's directory so local mode can score it alongside the
    # others without reconfiguring.
    dirs = dict(cfg.get("evo2_model_dirs") or {})
    dirs[model_name] = download_dir
    set_local_config("evo2_model_dirs", dirs)

    # Write config for subsequent use
    set_local_config("evo2_model_dir", download_dir)
    set_local_config("evo2_model_name", f"evo2_{model_size}")

    _log(f"  Downloading {model_id} → {download_dir} (~14 GB, please wait)…")
    # Use HuggingFace snapshot_download via Python API
    py_script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='{model_id}', local_dir='{download_dir}')"
    )
    result = subprocess.run(
        [python_exe, "-c", py_script],
        timeout=7200, capture_output=False,
    )
    if result.returncode != 0:
        _log("[red]✗ Model download failed. To download manually:[/red]")
        _log(f"  huggingface-cli download {model_id} --local-dir {download_dir}")
        return False

    _log(f"[green]✓ Model downloaded to {download_dir}[/green]")
    return True


def _yn_prompt(msg: str) -> bool:
    try: return input(f"{msg} [y/N] ").strip().lower() in ("y", "yes")
    except Exception: return True


# ════════════════════════════════════════════════════════════════
#  Server Script Generation
# ════════════════════════════════════════════════════════════════

# Shared preamble for the generated DNA-LM services. It reproduces the window
# extraction used to build the MERGE training tables (see
# merge_feature_datasets/*/FEATURE_METHODS.md): a window centred on the variant,
# clamped at the contig edges, with the REF base verified against the reference.
# A local FASTA is preferred; without one it falls back to the MERGE sequence
# context endpoint.
_CONTEXT_HELPER = r'''
import os, requests

FASTA_PATH = __FASTA_PATH__
MERGE_API = "__MERGE_API__"
_genome = None


def _fasta():
    """Lazily open the reference FASTA; None when not configured/available."""
    global _genome
    if _genome is None:
        if not FASTA_PATH or not os.path.exists(FASTA_PATH):
            return None
        from pyfaidx import Fasta
        _genome = Fasta(FASTA_PATH, sequence_always_upper=True, as_raw=True)
    return _genome


def _chrom_key(genome, chrom):
    direct = chrom if str(chrom).startswith("chr") else "chr" + str(chrom)
    bare = direct[3:]
    for cand in (direct, bare):
        if cand in genome.keys():
            return cand
    return None


def _is_snv(ref, alt):
    return len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT"


def _trim_common_prefix(ref, alt):
    shift = 0
    while ref and alt and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        shift += 1
    return shift, ref, alt


def build_haplotype(chrom, pos, ref, alt, window_bp, multiple_of=1):
    """REF/ALT haplotype pair for a non-SNV (indel / MNV / delins).

    Ported from the reference pipeline's non_snv_sequences.make_sequence_pair:
    the common left prefix is trimmed only to locate the true edit coordinate,
    the window is centred on that edit, and ALT is the same window with
    edit_ref -> edit_alt (so ALT may be a different length than REF). Matching
    this exactly is what makes the score comparable to the training features.
    """
    ref, alt = str(ref).upper().strip(), str(alt).upper().strip()
    if not ref or not alt or any(b not in "ACGT" for b in ref + alt):
        return None, None, "REF/ALT must be non-empty A/C/G/T strings"

    genome = _fasta()
    if genome is None:
        return None, None, ("non-SNV scoring needs a local reference FASTA; "
                            "set one with: merge local predict --genome-ref /path/to/hg38.fa")
    key = _chrom_key(genome, chrom)
    if key is None:
        return None, None, "chromosome not found in FASTA: %s" % chrom

    pos0 = int(pos) - 1
    chrom_len = len(genome[key])
    if pos0 < 0 or pos0 + len(ref) > chrom_len:
        return None, None, "variant outside chromosome bounds"

    observed = str(genome[key][pos0:pos0 + len(ref)]).upper()
    if observed != ref:
        return None, None, "reference mismatch: FASTA=%s, input=%s" % (observed, ref)

    prefix_len, edit_ref, edit_alt = _trim_common_prefix(ref, alt)
    edit_pos0 = pos0 + prefix_len
    if not edit_ref and not edit_alt:
        return None, None, "REF and ALT are identical after normalization"

    start0 = max(0, edit_pos0 - window_bp // 2)
    end0 = min(chrom_len, start0 + window_bp)
    start0 = max(0, end0 - window_bp)
    ref_sequence = str(genome[key][start0:end0]).upper()
    offset = edit_pos0 - start0

    if multiple_of > 1:
        trim = len(ref_sequence) % multiple_of
        if trim:
            ref_sequence = ref_sequence[trim:]
            start0 += trim
            offset -= trim

    if any(b not in "ACGT" for b in ref_sequence):
        return None, None, "sequence window contains non-ACGT bases"
    if offset < 0 or offset + len(edit_ref) > len(ref_sequence):
        return None, None, "edit falls outside sequence window"
    if edit_ref and ref_sequence[offset:offset + len(edit_ref)] != edit_ref:
        return None, None, "trimmed REF does not match FASTA window"

    alt_sequence = ref_sequence[:offset] + edit_alt + ref_sequence[offset + len(edit_ref):]
    return ref_sequence, alt_sequence, None


def extract_context(chrom, pos, ref, alt, window_bp, multiple_of=1, trim_end=False):
    """Return (ref_seq, alt_seq) for an SNV, or (None, None) with a reason.

    Mirrors the training-time extraction exactly: this is what makes the score
    comparable to the values the ensemble pkl was fitted on.
    """
    ref, alt = str(ref).upper().strip(), str(alt).upper().strip()
    if not _is_snv(ref, alt):
        return None, None, "only SNVs are supported"

    genome = _fasta()
    if genome is not None:
        key = _chrom_key(genome, chrom)
        if key is None:
            return None, None, "chromosome not found in FASTA: %s" % chrom
        pos0 = int(pos) - 1
        chrom_len = len(genome[key])
        start = max(0, pos0 - window_bp // 2)
        end = min(chrom_len, start + window_bp)
        start = max(0, end - window_bp)
        offset = pos0 - start
        if offset < 0 or offset >= end - start:
            return None, None, "variant outside window"
        seq = str(genome[key][start:end]).upper()
        if multiple_of > 1:
            trim = len(seq) % multiple_of
            if trim:
                if trim_end:
                    seq = seq[:-trim]          # carbon trims the tail
                else:
                    seq = seq[trim:]           # generator/nt_v2 trim the head
                    offset -= trim
        if offset < 0 or offset >= len(seq):
            return None, None, "variant outside window after trim"
        if seq[offset] != ref:
            return None, None, "reference mismatch: FASTA has %s, input has %s" % (seq[offset], ref)
        if any(b not in "ACGT" for b in seq):
            return None, None, "window contains non-ACGT base"
        return seq, seq[:offset] + alt + seq[offset + 1:], None

    # No local FASTA: fall back to the MERGE sequence-context endpoint.
    try:
        d = requests.post(MERGE_API + "/api/sequence_context/",
                          json={"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                                "genome": "hg38", "window": window_bp},
                          timeout=30).json()
    except Exception as e:
        return None, None, "sequence context fetch failed: %s" % e
    rs, vs = d.get("ref_seq"), d.get("var_seq")
    if not rs or not vs:
        return None, None, "sequence context unavailable: %s" % d.get("error", "unknown")
    rs, vs = rs.upper(), vs.upper()
    if multiple_of > 1:
        trim = len(rs) % multiple_of
        if trim:
            if trim_end:
                rs, vs = rs[:-trim], vs[:-trim]
            else:
                rs, vs = rs[trim:], vs[trim:]
    return rs, vs, None
'''


def _context_helper_code() -> str:
    cfg = _cfg()
    fasta = cfg.get("genome_ref") or ""
    return (_CONTEXT_HELPER
            .replace("__FASTA_PATH__", repr(fasta))
            .replace("__MERGE_API__", FIXED_API_URL))


def _dna_lm_server_code(model_name: str) -> str:
    """Generate the FastAPI service for one DNA language model.

    The scoring conventions are fixed by the MERGE training tables:
      carbon  -> logP_sum(REF) - logP_sum(ALT)   (SUM-delta), 24576 bp, <dna> wrapped
      causal  -> NLL(ALT) - NLL(REF)             600 bp, max_length 128
      masked  -> ||emb(ALT) - emb(REF)||_2       600 bp, max_length 128
    """
    repo = DNA_LM_MODELS[model_name]
    params = DNA_LM_PARAMS[model_name]
    port = _SERVICE_PORTS[model_name]
    cfg = _cfg()
    local_path = cfg.get(f"{model_name}_model_path") or ""
    model_ref = local_path or repo
    kind = params["kind"]
    # nt_v2's ESM-style attention breaks under bf16 matmul, and the training run
    # used fp32 for every LM, so only carbon takes bfloat16 here.
    dtype = "torch.bfloat16" if kind == "carbon" else "None"

    header = f'''#!/usr/bin/env python3
"""MERGE local {model_name} inference service ({repo}).

Scoring matches the MERGE training tables — do not change the window or the
score formula without recomputing the ensemble models.
"""
import torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
'''

    body = f'''
MODEL_REF = {model_ref!r}
MODEL_NAME = {model_name!r}
WINDOW_BP = {params["window_bp"]}
MAX_LENGTH = {params["max_length"]!r}
MULTIPLE_OF = {params["multiple_of"]}
KIND = {kind!r}
TRIM_END = {kind == "carbon"}
TRUST_MODEL = {params["trust_model"]}
TRUST_TOKENIZER = {params["trust_tokenizer"]}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = {dtype}

app = FastAPI()
_model = None
_tokenizer = None
_load_error = None


def _load():
    global _model, _tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        print("[%s] loading %s ..." % (MODEL_NAME, MODEL_REF), flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REF, trust_remote_code=TRUST_TOKENIZER)
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token or _tokenizer.unk_token
        cls = AutoModelForMaskedLM if KIND == "masked" else AutoModelForCausalLM
        kwargs = {{"trust_remote_code": TRUST_MODEL, "low_cpu_mem_usage": True}}
        if DTYPE is not None:
            kwargs["torch_dtype"] = DTYPE
        _model = cls.from_pretrained(MODEL_REF, **kwargs).to(DEVICE).eval()
        print("[%s] ready on %s" % (MODEL_NAME, DEVICE), flush=True)
    except Exception as e:
        _load_error = str(e)
        print("[%s] load failed: %s" % (MODEL_NAME, e), flush=True)


_load()


class V(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    genome_version: str = "hg38"


@app.get("/health")
def health():
    if _model is not None:
        return {{"status": "ok", "model": MODEL_NAME, "model_ref": MODEL_REF, "device": DEVICE}}
    return {{"status": "error" if _load_error else "loading", "error": _load_error}}


def _mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp_min(1)


@torch.inference_mode()
def _seq_logp_sum(seqs):
    """Total log-probability of each sequence (carbon's SUM convention)."""
    texts = ["<dna>" + s + "</dna>" for s in seqs]
    enc = _tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
    ids = enc["input_ids"].to(DEVICE)
    am = enc["attention_mask"].to(DEVICE)
    out = _model(input_ids=ids, attention_mask=am)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    tmask = am[:, 1:].bool()
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp = lp.masked_fill(~tmask, 0.0)
    return lp.sum(dim=1)


@torch.inference_mode()
def _seq_nll(seqs):
    """Length-normalised negative log-likelihood (causal convention).

    REF and ALT are scored in separate forward passes, exactly as the reference
    pipeline does. Batching them together would pad the shorter haplotype of an
    indel, and this model's NLL comes out measurably different when padded —
    matching the reference here is what keeps non-SNV scores on the training scale.
    """
    out = []
    for seq in seqs:
        enc = _tokenizer([seq], return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_LENGTH)
        ids = enc["input_ids"].to(DEVICE)
        am = enc["attention_mask"].to(DEVICE)
        res = _model(input_ids=ids, attention_mask=am)
        logits = res.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        mask = am[:, 1:].bool()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        lengths = mask.sum(1).clamp_min(1)
        out.append(-(lp.masked_fill(~mask, 0).sum(1) / lengths))
    return torch.cat(out)


@torch.inference_mode()
def _emb_l2(ref_seq, alt_seq):
    """L2 distance between mean-pooled embeddings (masked convention)."""
    enc = _tokenizer([ref_seq, alt_seq], return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_LENGTH)
    enc = {{k: v.to(DEVICE) for k, v in enc.items()}}
    out = _model(**enc, output_hidden_states=True)
    emb = _mean_pool(out.hidden_states[-1], enc["attention_mask"])
    return torch.linalg.vector_norm(emb[1:] - emb[:1], dim=1).float().cpu().numpy()[0]


@app.post("/predict")
def predict(v: V):
    if _model is None:
        return {{"score": None, "error": "model not loaded: %s" % (_load_error or "initializing")}}
    try:
        ref, alt = str(v.ref).upper().strip(), str(v.alt).upper().strip()
        # SNVs get the single-base swap; indels/MNVs get the haplotype pair, in
        # which ALT may differ in length from REF. Both use the same window and
        # score formula for this model.
        if _is_snv(ref, alt):
            variant_type = "snv"
            ref_seq, alt_seq, err = extract_context(
                v.chrom, v.pos, ref, alt, WINDOW_BP, MULTIPLE_OF, TRIM_END)
        else:
            variant_type = "non_snv"
            ref_seq, alt_seq, err = build_haplotype(
                v.chrom, v.pos, ref, alt, WINDOW_BP, MULTIPLE_OF)
        if err:
            return {{"score": None, "error": err}}

        if KIND == "carbon":
            totals = _seq_logp_sum([ref_seq, alt_seq])
            score = float(totals[0] - totals[1])          # logP_sum(ref) - logP_sum(alt)
        elif KIND == "causal":
            nll = _seq_nll([ref_seq, alt_seq])
            score = float(nll[1] - nll[0])                # NLL(alt) - NLL(ref)
        else:
            score = float(_emb_l2(ref_seq, alt_seq))      # ||emb(alt) - emb(ref)||_2

        return {{"score": score, "model": MODEL_NAME, "model_ref": MODEL_REF,
                 "window_bp": WINDOW_BP, "score_type": KIND,
                 "variant_type": variant_type}}
    except Exception as e:
        return {{"score": None, "error": str(e)}}


@app.post("/batch_predict")
def batch_predict(payload: dict):
    variants = payload.get("variants") or []
    results = []
    for item in variants:
        results.append(predict(V(**item)))
    return {{"success": True, "results": results}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
    return header + _context_helper_code() + body


def _model_ref(model_name: str, default_repo: str) -> str:
    """Weights location for a model: a configured local directory wins over the
    HuggingFace repo id. Set one with `merge local env setup --model <name>
    --local-model-path DIR`, or `<name>_model_path` in the config."""
    path = _cfg().get(f"{model_name}_model_path")
    return path if path else default_repo


def _write_server_script(model_name: str) -> str:
    d = Path.home() / ".merge-local-servers"
    d.mkdir(exist_ok=True)
    p = d / f"{model_name}_server.py"
    cfg = _cfg()

    if model_name in DNA_LM_MODELS:
        p.write_text(_dna_lm_server_code(model_name), encoding="utf-8")
        return str(p)

    # ── 1. NT Script (old NT, masked embedding L2) ───────────────
    # Scoring口径 per FEATURE_METHODS.md: ||emb(alt) - emb(ref)||_2 over an
    # 8192 bp window centred on the variant, mean-pooled last hidden state,
    # max_length = tokenizer.model_max_length. The previous version used a
    # 6000 bp window and a 10000-token cap, which did not match training.
    if model_name == "nt":
        code = '''#!/usr/bin/env python3
"""MERGE local NT (old) inference service.

Scoring matches the MERGE training tables — do not change the window or the
score formula without recomputing the ensemble models.
"""
import torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
''' + _context_helper_code() + '''
MODEL = __NT_MODEL__
# The two MERGE datasets scored this same model differently, so the service has
# to follow the variant: the SNV tables used an 8192 bp window at the tokenizer's
# own max length, while the non-SNV tables used a 600 bp haplotype window at
# max_length 128 (feature nt_old_score).
SNV_WINDOW_BP = 8192
NONSNV_WINDOW_BP = 600
NONSNV_MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
_model = None
_tokenizer = None
_load_error = None


def _load():
    global _model, _tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        print("[NT] loading %s ..." % MODEL, flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model = AutoModelForMaskedLM.from_pretrained(MODEL).to(DEVICE).eval()
        print("[NT] ready on %s" % DEVICE, flush=True)
    except Exception as e:
        _load_error = str(e)
        print("[NT] load failed: %s" % e, flush=True)


_load()


class V(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    genome_version: str = "hg38"


@app.get("/health")
def health():
    if _model is not None:
        return {"status": "ok", "model": MODEL, "device": DEVICE}
    return {"status": "error" if _load_error else "loading", "error": _load_error}


@torch.inference_mode()
def _score(ref_seq, alt_seq, max_length):
    enc = _tokenizer([ref_seq, alt_seq], return_tensors="pt", padding=True,
                     truncation=True, max_length=max_length)
    ids = enc["input_ids"].to(DEVICE)
    mask = (ids != _tokenizer.pad_token_id).to(DEVICE)
    out = _model(ids, attention_mask=mask, output_hidden_states=True)
    last = out.hidden_states[-1]
    emb = (last * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    return float(torch.norm(emb[1] - emb[0]).item())


@app.post("/predict")
def predict(v: V):
    if _model is None:
        return {"score": None, "error": "model not loaded: %s" % (_load_error or "initializing")}
    try:
        ref, alt = str(v.ref).upper().strip(), str(v.alt).upper().strip()
        if _is_snv(ref, alt):
            variant_type = "snv"
            window, max_length = SNV_WINDOW_BP, _tokenizer.model_max_length
            ref_seq, alt_seq, err = extract_context(v.chrom, v.pos, ref, alt, window)
        else:
            variant_type = "non_snv"
            window, max_length = NONSNV_WINDOW_BP, NONSNV_MAX_LENGTH
            ref_seq, alt_seq, err = build_haplotype(v.chrom, v.pos, ref, alt, window)
        if err:
            return {"score": None, "error": err}
        return {"score": _score(ref_seq, alt_seq, max_length), "model": MODEL,
                "window_bp": window, "score_type": "masked_emb_l2",
                "variant_type": variant_type}
    except Exception as e:
        return {"score": None, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
'''

    # ── 2. HyenaDNA Script (causal next-token log-ratio) ─────────
    # Scoring口径 per FEATURE_METHODS.md: next-token log p(alt) - log p(ref)
    # over a 1000 bp *upstream* context ending just before the variant. The
    # previous version scored the last token of a variant-centred window, which
    # is a different quantity entirely and did not match training.
    elif model_name == "hyenadna":
        code = '''#!/usr/bin/env python3
"""MERGE local HyenaDNA inference service.

Scoring matches the MERGE training tables — do not change the context or the
score formula without recomputing the ensemble models.
"""
import os, numpy as np, torch, uvicorn, requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
''' + _context_helper_code() + '''
MODEL = __HYENA_MODEL__
# Like NT, this model was scored two different ways by the MERGE datasets:
#   SNV     -> next-token log p(ALT) - log p(REF) over 1000 bp of upstream
#              context (feature heyenadna_score)
#   non-SNV -> causal NLL(ALT) - NLL(REF) over a 600 bp haplotype window at
#              max_length 1024 (feature hyenadna_score)
# max_length 1024 matters: at 128 the character-level tokenizer truncates away
# the central edit and every score collapses to zero.
CONTEXT_LEN = 1000
NONSNV_WINDOW_BP = 600
NONSNV_MAX_LENGTH = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
_model = None
_tokenizer = None
_load_error = None


def _load():
    global _model, _tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        print("[HyenaDNA] loading %s ..." % MODEL, flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).to(DEVICE).eval()
        print("[HyenaDNA] ready on %s" % DEVICE, flush=True)
    except Exception as e:
        _load_error = str(e)
        print("[HyenaDNA] load failed: %s" % e, flush=True)


_load()


class V(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    genome_version: str = "hg38"


@app.get("/health")
def health():
    if _model is not None:
        return {"status": "ok", "model": MODEL, "device": DEVICE}
    return {"status": "error" if _load_error else "loading", "error": _load_error}


def _upstream_context(chrom, pos, ref=None):
    """The CONTEXT_LEN bases immediately BEFORE the variant position.

    `ref` is verified against the reference when a local FASTA is available.
    The scoring itself only needs the upstream bases, but skipping the check
    would let a wrong REF through silently: every other model rejects such a
    variant, so this one must not answer with a confident-looking number.
    """
    genome = _fasta()
    if genome is not None:
        key = _chrom_key(genome, chrom)
        if key is None:
            return None, "chromosome not found in FASTA: %s" % chrom
        pos0 = int(pos) - 1
        if ref:
            observed = str(genome[key][pos0:pos0 + 1]).upper()
            if observed and observed != ref:
                return None, "reference mismatch: FASTA has %s, input has %s" % (observed, ref)
        start = max(0, pos0 - CONTEXT_LEN)
        end = pos0
        seq = str(genome[key][start:end]).upper()
        return "".join(b if b in "ACGT" else "N" for b in seq), None
    # Fallback: take a centred window from the API and keep only the upstream half.
    try:
        d = requests.post(MERGE_API + "/api/sequence_context/",
                          json={"chrom": chrom, "pos": int(pos), "ref": "N", "alt": "N",
                                "genome": "hg38", "window": CONTEXT_LEN * 2},
                          timeout=30).json()
    except Exception as e:
        return None, "sequence context fetch failed: %s" % e
    rs = d.get("ref_seq")
    if not rs:
        return None, "sequence context unavailable"
    mid = len(rs) // 2
    return rs[max(0, mid - CONTEXT_LEN):mid].upper(), None


@torch.inference_mode()
def _score(context, ref, alt):
    ids = _tokenizer(context, return_tensors="pt")["input_ids"].to(DEVICE)
    probs = torch.softmax(_model(ids).logits[0, -1, :], dim=0)
    p_ref = float(probs[_tokenizer.convert_tokens_to_ids(ref)])
    p_alt = float(probs[_tokenizer.convert_tokens_to_ids(alt)])
    eps = 1e-12
    return float(np.log(p_alt + eps) - np.log(p_ref + eps))


@torch.inference_mode()
def _nll(seqs, max_length):
    """One forward pass per sequence — see the note in the DNA-LM services: an
    indel's REF and ALT differ in length, and padding them into one batch shifts
    the causal NLL away from the training scale."""
    out = []
    for seq in seqs:
        enc = _tokenizer([seq], return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(DEVICE)
        am = enc.get("attention_mask")
        am = am.to(DEVICE) if am is not None else torch.ones_like(ids)
        logits = _model(ids).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        mask = am[:, 1:].bool()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        lengths = mask.sum(1).clamp_min(1)
        out.append(-(lp.masked_fill(~mask, 0).sum(1) / lengths))
    return torch.cat(out)


@app.post("/predict")
def predict(v: V):
    if _model is None:
        return {"score": None, "error": "model not loaded: %s" % (_load_error or "initializing")}
    try:
        ref, alt = str(v.ref).upper().strip(), str(v.alt).upper().strip()
        if _is_snv(ref, alt):
            context, err = _upstream_context(v.chrom, v.pos, ref)
            if err:
                return {"score": None, "error": err}
            if not context:
                return {"score": None, "error": "empty upstream context"}
            return {"score": _score(context, ref, alt), "model": MODEL,
                    "context_length": CONTEXT_LEN, "score_type": "next_token_logratio",
                    "variant_type": "snv"}

        ref_seq, alt_seq, err = build_haplotype(v.chrom, v.pos, ref, alt, NONSNV_WINDOW_BP)
        if err:
            return {"score": None, "error": err}
        nll = _nll([ref_seq, alt_seq], NONSNV_MAX_LENGTH)
        return {"score": float(nll[1] - nll[0]), "model": MODEL,
                "context_length": NONSNV_WINDOW_BP, "score_type": "alt_nll_minus_ref_nll",
                "variant_type": "non_snv"}
    except Exception as e:
        return {"score": None, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
'''

    # ── 3. AlphaGenome Script ────────────────────────────────────
    elif model_name == "alphagenome":
        api_key = cfg.get("alphagenome_api_key", "")
        code = f'''#!/usr/bin/env python3
"""MERGE local AlphaGenome client service.

AlphaGenome runs on Google DeepMind's servers; this process is only a client.
It needs your own API key — get one at
https://deepmind.google.com/science/alphagenome — supplied either via
`merge alphagenome configure --api-key ...` or the ALPHAGENOME_API_KEY
environment variable.
"""
from flask import Flask, request, jsonify
import os, numpy as np, pandas as pd

DNA_MODEL = None
SUPPORTED_LENGTHS = {{}}
INIT_ERROR = None

# The configured key is baked in at generation time; the environment variable
# wins so a key can be rotated without regenerating this script.
API_KEY = os.environ.get("ALPHAGENOME_API_KEY") or {api_key!r}

try:
    if not API_KEY:
        raise RuntimeError(
            "no AlphaGenome API key. Set one with "
            "'merge alphagenome configure --api-key YOUR_KEY' or export "
            "ALPHAGENOME_API_KEY. Keys: https://deepmind.google.com/science/alphagenome")
    from alphagenome.models import dna_client, variant_scorers
    from alphagenome.data import genome
    DNA_MODEL = dna_client.create(API_KEY)
    SUPPORTED_LENGTHS = dna_client.SUPPORTED_SEQUENCE_LENGTHS
    print("[AlphaGenome] client ready", flush=True)
except Exception as e:
    INIT_ERROR = str(e)
    print("[AlphaGenome] init failed: %s" % e, flush=True)

app = Flask(__name__)

@app.route('/health')
def health():
    if DNA_MODEL:
        return jsonify({{"status": "ok", "model": "alphagenome", "remote_api": True}})
    return jsonify({{"status": "error", "error": INIT_ERROR}})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        chrom, pos, ref, alt = data['chrom'], int(data['pos']), data['ref'], data['alt']
        if not chrom.startswith('chr'): chrom = 'chr' + chrom

        variant = genome.Variant(chromosome=chrom, position=pos, reference_bases=ref, alternate_bases=alt)
        interval = variant.reference_interval.resize(SUPPORTED_LENGTHS['SEQUENCE_LENGTH_1MB'])

        # Use every RECOMMENDED scorer the organism supports, exactly as the
        # MERGE feature tables were built. Scoring only a subset changes the
        # population that max/min/mean are taken over, which silently shifts
        # every AlphaGenome feature.
        scorer_selections = {{
            'rna_seq': True, 'cage': True, 'procap': True, 'atac': True,
            'dnase': True, 'chip_histone': True, 'chip_tf': True,
            'polyadenylation': True, 'splice_sites': True,
            'splice_site_usage': True, 'splice_junctions': True,
        }}
        all_s = variant_scorers.RECOMMENDED_VARIANT_SCORERS
        sel = [all_s[k] for k in all_s if scorer_selections.get(k.lower(), False)]
        organism = dna_client.Organism.HOMO_SAPIENS
        unsupported = [
            sc for sc in sel
            if (organism.value not in variant_scorers.SUPPORTED_ORGANISMS[sc.base_variant_scorer])
            or ((sc.requested_output == dna_client.OutputType.PROCAP)
                and (organism == dna_client.Organism.MUS_MUSCULUS))
        ]
        for sc in unsupported:
            sel.remove(sc)

        raw_scores = DNA_MODEL.score_variant(interval=interval, variant=variant,
                                              variant_scorers=sel, organism=organism)
        df = variant_scorers.tidy_scores([raw_scores])
        if df.empty: return jsonify({{"error": "No scores returned"}})

        def _sf(val): return float(val) if pd.notna(val) else None

        df['_ot'] = df['output_type'].astype(str).str.upper()
        # Per-scorer MAX of raw_score across that scorer's tracks. This mirrors how the
        # training tables were built (groupby(variant_id, output_type).raw_score.max(),
        # then the three terms summed with the division applied to SPLICE_JUNCTIONS
        # alone). It was .mean() until 2026-08-13, which made the served feature a
        # different quantity from the one the splicing ensemble was fitted on.
        def _gx(k):
            sub = df[df['_ot'] == k]
            if sub.empty:
                sub = df[df['_ot'].str.endswith(k, na=False)]
            return _sf(sub['raw_score'].max()) if not sub.empty else None

        splice_sites_max      = _gx('SPLICE_SITES')
        splice_site_usage_max = _gx('SPLICE_SITE_USAGE')
        splice_junctions_max  = _gx('SPLICE_JUNCTIONS')

        alphagenome_splicing = None
        vals = [splice_sites_max, splice_site_usage_max, splice_junctions_max]
        if any(v is not None for v in vals):
            alphagenome_splicing = (
                (splice_sites_max      or 0.0) +
                (splice_site_usage_max or 0.0) +
                ((splice_junctions_max or 0.0) / 5.0)
            )

        # tidy_scores yields one row per (scorer, track), so aggregating over
        # rows gives the global min/max and an n-weighted mean — the same
        # cross-scorer aggregation the training tables used. The quantile comes
        # from the API's own quantile_score column; deriving it from the raw
        # scores would produce an empirical rank in [0,1] and lose the sign.
        stats = {{
            "alphagenome_raw_score_max":       _sf(df['raw_score'].max()),
            "alphagenome_quantile_score_max":  _sf(df['quantile_score'].max()),
            "alphagenome_raw_score_min":       _sf(df['raw_score'].min()),
            "alphagenome_quantile_score_min":  _sf(df['quantile_score'].min()),
            "alphagenome_raw_score_mean":      _sf(df['raw_score'].mean()),
            "alphagenome_quantile_score_mean": _sf(df['quantile_score'].mean()),
            "alphagenome_splicing": _sf(alphagenome_splicing),
        }}
        return jsonify({{"success": True, "result": {{"statistics": stats}}}})
    except Exception as e:
        return jsonify({{"success": False, "error": str(e)}}), 500

if __name__ == "__main__": app.run(host="0.0.0.0", port=5000)
'''

    # ── 4. Evo2 Script (multi-variant: evo2_7b / 7b_base / 1b_base) ──
    # ── 3b. Enformer (TensorFlow SavedModel) ─────────────────────
    # Unlike every other model here this one is TF, not PyTorch, and it needs a
    # normalisation transform pickle alongside the weights. Scoring口径 per
    # FEATURE_METHODS.md: alt-minus-ref per track (mean over bins), passed
    # through the transform, then reduced to the 7 summary statistics the
    # ensemble uses. kipoiseq's VariantSeqExtractor handles indels/MNVs, so this
    # service covers non-SNVs without a separate haplotype path.
    elif model_name == "enformer":
        model_path = cfg.get("enformer_model_path") or str(Path.home() / "enformer_model")
        transform_path = (cfg.get("enformer_transform_path")
                          or str(Path.home() / "genome_data" / "enformer.transform.pkl"))
        fasta = cfg.get("genome_ref") or ""
        code = f'''#!/usr/bin/env python3
"""MERGE local Enformer inference service.

Scoring matches the MERGE training tables — do not change the sequence length or
the summary statistics without recomputing the ensemble models.
"""
import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = {model_path!r}
TRANSFORM_PATH = {transform_path!r}
FASTA_PATH = {fasta!r}
SEQUENCE_LENGTH = 393216
ORGANISM = "human"

app = FastAPI()
_scorer = None
_extractor = None
_load_error = None


class _FastaStringExtractor:
    def __init__(self, fasta_file):
        import pyfaidx
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._sizes = {{k: len(v) for k, v in self.fasta.items()}}

    def extract(self, interval, **kwargs):
        from kipoiseq import Interval
        chrom_len = self._sizes[interval.chrom]
        trimmed = Interval(interval.chrom, max(interval.start, 0), min(interval.end, chrom_len))
        seq = str(self.fasta.get_seq(trimmed.chrom, trimmed.start + 1, trimmed.stop).seq).upper()
        return "N" * max(-interval.start, 0) + seq + "N" * max(interval.end - chrom_len, 0)


def _load():
    global _scorer, _extractor, _load_error
    if _scorer is not None or _load_error is not None:
        return
    try:
        import joblib, kipoiseq, tensorflow as tf, tensorflow_hub as hub
        print("[Enformer] loading %s ..." % MODEL_PATH, flush=True)
        model = hub.load(MODEL_PATH).model
        with tf.io.gfile.GFile(TRANSFORM_PATH, "rb") as fh:
            transform = joblib.load(fh).steps[0][1]

        class _Scorer:
            def score(self, ref_1h, alt_1h):
                ref = model.predict_on_batch(ref_1h)[ORGANISM].numpy()
                alt = model.predict_on_batch(alt_1h)[ORGANISM].numpy()
                raw = alt.mean(axis=1) - ref.mean(axis=1)
                return transform.transform(raw)

        _scorer = _Scorer()
        _extractor = kipoiseq.extractors.VariantSeqExtractor(
            reference_sequence=_FastaStringExtractor(FASTA_PATH))
        print("[Enformer] ready", flush=True)
    except Exception as e:
        import traceback
        _load_error = str(e)
        print("[Enformer] load failed: %s" % e, flush=True)
        traceback.print_exc()


_load()


class V(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    genome_version: str = "hg38"


@app.get("/health")
def health():
    if _scorer is not None:
        return {{"status": "ok", "model": "enformer", "sequence_length": SEQUENCE_LENGTH}}
    return {{"status": "error" if _load_error else "loading", "error": _load_error}}


def _summarize(scores):
    scores = np.asarray(scores).ravel()
    abs_scores = np.abs(scores)
    top = min(5, abs_scores.size)
    top_abs = np.partition(abs_scores, -top)[-top:]
    return {{
        "enformer_absmean": float(abs_scores.mean()),
        "enformer_mean": float(scores.mean()),
        "enformer_std": float(scores.std()),
        "enformer_max_abs": float(abs_scores.max()),
        "enformer_top5_absmean": float(top_abs.mean()),
        "enformer_pos_sum": float(scores[scores > 0].sum()),
        "enformer_neg_sum": float(scores[scores < 0].sum()),
    }}


@app.post("/predict")
def predict(v: V):
    if _scorer is None:
        return {{"error": "model not loaded: %s" % (_load_error or "initializing")}}
    try:
        import kipoiseq
        from kipoiseq import Interval
        chrom = v.chrom if str(v.chrom).startswith("chr") else "chr" + str(v.chrom)
        variant = kipoiseq.dataclasses.Variant(
            chrom=chrom, pos=int(v.pos),
            ref=str(v.ref).upper(), alt=str(v.alt).upper(), id="q")
        interval = Interval(chrom, variant.pos, variant.pos).resize(SEQUENCE_LENGTH)
        center = interval.center() - interval.start
        reference = _extractor.extract(interval, [], anchor=center)
        alternate = _extractor.extract(interval, [variant], anchor=center)
        one_hot = kipoiseq.transforms.functional.one_hot_dna
        ref_1h = one_hot(reference).astype(np.float32)[np.newaxis]
        alt_1h = one_hot(alternate).astype(np.float32)[np.newaxis]
        scores = _scorer.score(ref_1h, alt_1h)
        out = _summarize(scores)
        out.update({{"model": "enformer", "sequence_length": SEQUENCE_LENGTH,
                     "score": out["enformer_max_abs"]}})
        return out
    except Exception as e:
        return {{"error": str(e)}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={_SERVICE_PORTS["enformer"]})
'''

    elif model_name == "evo2":
        default_model = cfg.get("evo2_model_name", "evo2_7b")
        model_dirs = {name: _get_evo2_model_dir(name) for name in EVO2_VARIANTS}

        code = f'''#!/usr/bin/env python3
"""MERGE local Evo2 inference service (evo2_7b / evo2_7b_base / evo2_1b_base).

Score口径 per FEATURE_METHODS.md:
    score = mean_PLL(REF) - mean_PLL(ALT)
over an 8192 bp window centred on the variant, reduce_method="mean" and
average_reverse_complement=False. Do not change these without recomputing the
ensemble models.

The three variants are GPU-exclusive: a request naming a different model unloads
the current one and loads the requested one. That is slow but keeps VRAM bounded,
and mirrors the website's behaviour. evo2_1b_base additionally needs FP8
(compute capability >= 8.9).
"""
import os, sys, gc, torch, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# evo2's official weights carry a Transformer-Engine FP8 extra-state pickled
# object; loading it requires this opt-in.
os.environ.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
''' + _context_helper_code() + f'''
MODEL_DIRS = {model_dirs!r}
DEFAULT_MODEL = {default_model!r}
WINDOW_BP = 8192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
_model = None
_model_name = None
_load_error = None


def _unshadow_evo2():
    """Drop cwd-like entries from sys.path so `import evo2` finds the installed
    package rather than any evo2.py sitting in the working directory."""
    here = os.path.realpath(os.getcwd())
    keep = []
    for item in sys.path:
        try:
            resolved = os.path.realpath(item or os.getcwd())
        except Exception:
            keep.append(item)
            continue
        if resolved == here and os.path.exists(os.path.join(resolved, "evo2.py")):
            continue
        keep.append(item)
    sys.path[:] = keep


def _load_model(name):
    """Load one Evo2 variant, unloading whatever is currently resident."""
    global _model, _model_name, _load_error
    if _model is not None and _model_name == name:
        return
    if _model is not None:
        print("[Evo2] unloading %s to make room for %s" % (_model_name, name), flush=True)
        _model = None
        _model_name = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    _load_error = None
    model_dir = MODEL_DIRS.get(name) or ""
    # evo2 resolves configs/*.yml relative to the process cwd, so chdir there.
    # Do NOT put the weight directory on sys.path: a stray evo2.py next to the
    # weights would shadow the installed evo2 package, and the resulting partial
    # import surfaces as a confusing flash_attn ABI error rather than an
    # ImportError. The reference pipeline guards against the same trap.
    if os.path.isdir(model_dir):
        os.chdir(model_dir)
    _unshadow_evo2()

    print("[Evo2] loading %s (device %s, dir %s) ..." % (name, DEVICE, model_dir), flush=True)
    try:
        from evo2.models import Evo2

        pt_file = os.path.join(model_dir, name + ".pt")
        local_path = pt_file if os.path.isfile(pt_file) else None

        if DEVICE == "cuda":
            torch.set_default_dtype(torch.bfloat16)
        try:
            _model = Evo2(model_name=name, local_path=local_path)
        finally:
            torch.set_default_dtype(torch.float32)
        _model_name = name
        print("[Evo2] ready: %s @ %s" % (name, DEVICE), flush=True)
    except Exception as e:
        import traceback
        _load_error = str(e)
        _model = None
        _model_name = None
        print("[Evo2] load failed for %s: %s" % (name, e), flush=True)
        traceback.print_exc()


_load_model(DEFAULT_MODEL)


class V(BaseModel):
    chrom: str
    pos: int
    ref: str
    alt: str
    genome: str = "hg38"
    genome_version: str = "hg38"
    model: str = ""
    model_name: str = ""


@app.get("/health")
def health():
    if _model is not None:
        return {{"status": "ok", "model": _model_name, "device": DEVICE,
                 "available": list(MODEL_DIRS.keys())}}
    return {{"status": "error" if _load_error else "loading", "error": _load_error}}


@app.post("/predict")
def predict(v: V):
    name = v.model or v.model_name or DEFAULT_MODEL
    if name not in MODEL_DIRS:
        return {{"llr_score": None, "error": "unknown Evo2 variant: %s" % name}}
    try:
        _load_model(name)
    except Exception as e:
        return {{"llr_score": None, "error": "load failed: %s" % e}}
    if _model is None:
        return {{"llr_score": None, "error": "model not loaded: %s" % (_load_error or "initializing")}}

    try:
        ref, alt = str(v.ref).upper().strip(), str(v.alt).upper().strip()
        if _is_snv(ref, alt):
            variant_type = "snv"
            ref_seq, alt_seq, err = extract_context(v.chrom, v.pos, ref, alt, WINDOW_BP)
        else:
            variant_type = "non_snv"
            ref_seq, alt_seq, err = build_haplotype(v.chrom, v.pos, ref, alt, WINDOW_BP)
        if err:
            return {{"llr_score": None, "error": err}}

        with torch.inference_mode():
            scores = _model.score_sequences(
                [ref_seq, alt_seq],
                batch_size=1,
                reduce_method="mean",
                average_reverse_complement=False,
            )
        ref_ll, alt_ll = float(scores[0]), float(scores[1])
        score = ref_ll - alt_ll     # positive => ALT less likely => more disruptive

        out = {{"model": name, "model_name": name,
                "ref_score": ref_ll, "var_score": alt_ll,
                "context_length": WINDOW_BP,
                "variant_type": variant_type,
                "score_class": ("deleterious" if score > 0 else
                                "benign" if score < 0 else "unknown")}}
        # The MERGE feature is the same quantity either way; the two names just
        # mirror what the SNV and non-SNV result shapes have always been called.
        if variant_type == "snv":
            out["llr_score"] = score
        else:
            out["score_mean"] = score
            out["llr_score"] = None
        out["score"] = score
        return out
    except Exception as e:
        return {{"llr_score": None, "error": str(e)}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
'''

    else:
        code = "# (Script logic)"

    # Resolve model references last, so NT and HyenaDNA can be pointed at a
    # local weights directory. Both are needed when the HuggingFace download is
    # unavailable (air-gapped machines) or when the cached copy is incomplete —
    # transformers then fails with a misleading "make sure ... is the correct
    # path" message rather than saying the download never finished.
    code = (code
            .replace("__NT_MODEL__", repr(_model_ref("nt", "InstaDeepAI/nucleotide-transformer-500m-human-ref")))
            .replace("__HYENA_MODEL__", repr(_model_ref("hyenadna", "LongSafari/hyenadna-medium-160k-seqlen-hf"))))

    p.write_text(code, encoding="utf-8")
    return str(p)


# ════════════════════════════════════════════════════════════════
#  start_local_service — evo2 uses cwd= parameter
# ════════════════════════════════════════════════════════════════

def serving_env(model_name: str) -> Optional[str]:
    """Name of the conda environment actually serving this model, or None.

    A service can legitimately run from a different environment than the one
    `env setup` would create — several models share the same dependencies, so
    one environment can serve them all. Reporting the environment that is really
    behind the port avoids the confusing "online, but env not deployed".
    """
    port = _SERVICE_PORTS.get(model_name)
    if not port:
        return None
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"],
                                      text=True, timeout=5).split()
    except Exception:
        return None
    for pid in out:
        try:
            exe = os.path.realpath(f"/proc/{pid.strip()}/exe")
        except Exception:
            continue
        # <conda_root>/envs/<name>/bin/python -> <name>
        parts = Path(exe).parts
        if "envs" in parts:
            i = parts.index("envs")
            if i + 1 < len(parts):
                return parts[i + 1]
        return exe
    return None


def _service_process_gone(log_file: str) -> bool:
    """True when the log's tail shows the process already gave up.

    Used to stop waiting early: a service that raised on import will never come
    online, so there is no point sitting through the full timeout.
    """
    try:
        with open(log_file, "r", errors="replace") as fh:
            tail = fh.read()[-4000:]
    except Exception:
        return False
    markers = ("Traceback (most recent call last)", "ModuleNotFoundError",
               "EnvironmentLocationNotFound", "Address already in use",
               "conda: error", "CondaEnvironmentError")
    if not any(m in tail for m in markers):
        return False
    # Uvicorn/Flask print their banner once serving; if that came after the
    # error the process is alive and the error was non-fatal.
    for banner in ("Uvicorn running on", "Running on http"):
        if banner in tail and tail.rindex(banner) > max(
                (tail.rindex(m) for m in markers if m in tail), default=-1):
            return False
    return True


def _preflight_service(model_name: str, env_name: str) -> Optional[str]:
    """Return a reason the service cannot start, or None when it looks ready.

    Checked before launching so a half-finished environment fails in a second
    with an actionable message, instead of after a silent 150-second wait.
    """
    if not _conda_env_exists(env_name):
        return (f"conda environment '{env_name}' does not exist.\n"
                f"    Run: merge local env setup --model {model_name}")

    python_exe = _get_conda_env_python(env_name)
    if not python_exe:
        return (f"conda environment '{env_name}' has no Python interpreter — the setup was\n"
                f"    probably interrupted. Re-run: merge local env setup --model {model_name}")

    # Each service needs its own runtime; a missing one means setup did not finish.
    required = {
        "alphagenome": ["alphagenome", "flask"],
        "enformer":    ["tensorflow", "tensorflow_hub", "kipoiseq"],
        "evo2":        ["evo2", "torch"],
    }.get(model_name, ["torch", "transformers"])

    probe = "import " + ", ".join(required)
    try:
        r = subprocess.run([python_exe, "-c", probe], capture_output=True, text=True, timeout=120)
    except Exception:
        return None  # cannot tell; let the service try
    if r.returncode != 0:
        missing = (r.stderr or "").strip().splitlines()[-1:] or ["unknown import error"]
        return (f"the '{env_name}' environment is missing dependencies ({missing[0]}).\n"
                f"    The setup step may have been interrupted. Re-run:\n"
                f"      merge local env setup --model {model_name}")
    return None


def start_local_service(model_name: str, console=None, gpu: Optional[str] = None) -> bool:
    """Start one model service. `gpu` pins it to specific CUDA device(s), e.g.
    "2" or "0,1" — needed on multi-GPU boxes where models must be spread out, and
    required for Evo2, which only runs on FP8-capable cards."""
    env_name = ENV_NAMES.get(model_name, model_name)

    problem = _preflight_service(model_name, env_name)
    if problem:
        msg = f"[red]✗ Cannot start {model_name}: {problem}[/red]"
        if console: console.print(msg)
        else: print(f"✗ Cannot start {model_name}: {problem}")
        return False

    script   = _write_server_script(model_name)
    log_file = str(Path.home() / f".merge-{model_name}.log")

    # evo2 must run from its model directory (configs/ must be in cwd)
    cwd = None
    if model_name == "evo2":
        model_dir = _get_evo2_model_dir()
        if os.path.isdir(model_dir):
            cwd = model_dir
            if console:
                console.print(f"  [dim]Working directory: {cwd}[/dim]")
        else:
            msg = (f"[yellow]⚠ Evo2 model directory not found: {model_dir}\n"
                   "  Download weights first: merge local env setup --model evo2[/yellow]")
            if console: console.print(msg)
            else: print(msg)

    # Launch the environment's interpreter directly rather than via `conda run`:
    # conda is often missing from PATH in non-interactive shells, and `conda run`
    # also buffers the child's output, which hides startup errors from the log.
    python_exe = _get_conda_env_python(env_name)
    cmd = ([python_exe, "-u", script] if python_exe
           else ["conda", "run", "-n", env_name, "python", "-u", script])

    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # These models ship offline-capable weights; avoid a HuggingFace round-trip
    # (and its long timeout) on machines without open internet.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if model_name == "evo2":
        # evo2's official checkpoints store a Transformer-Engine FP8 extra-state
        # as a pickle; loading it needs this opt-in.
        env.setdefault("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")

    gpu_note = f" on GPU {gpu}" if gpu is not None else ""
    print(f"▶ Starting {model_name} inference service{gpu_note}… (log: {log_file})")
    with open(log_file, "a") as lf:
        subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            cwd=cwd,               # ensures configs/ can be found
            env=env,
            start_new_session=True,
        )

    # Load times differ by an order of magnitude across these models, so wait
    # proportionally and describe the actual model rather than always Evo2.
    wait_s, hint = {
        "alphagenome":  (60,  "connecting to the AlphaGenome API"),
        "evo2":         (300, "Evo2 loads a 7B checkpoint, ~60-120s"),
        "carbon":       (300, "Carbon-3B is a large checkpoint"),
        "enformer":     (180, "Enformer loads a TensorFlow SavedModel"),
        "generator":    (180, "loading a 1.2B checkpoint"),
        "generator_v2": (180, "loading a 1.2B checkpoint"),
    }.get(model_name, (150, "loading model weights"))

    print(f"  Waiting for {model_name} to be ready (up to {wait_s} seconds)…")
    for i in range(wait_s // 5):
        time.sleep(5)
        health = check_service_health(model_name)
        if health["online"]:
            print(f"  ✓ {model_name} service ready ({health['latency_ms']} ms)")
            return True
        # A service that died on startup will never come up; stop early and say so.
        if (i + 1) % 4 == 0 and _service_process_gone(log_file):
            print(f"  ✗ {model_name} exited during startup — check log: {log_file}")
            return False
        if (i + 1) % 6 == 0:
            print(f"  … waited {(i+1)*5}s, still waiting ({hint})")

    print(f"  ✗ {model_name} service startup timed out — check log: {log_file}")
    return False


def stop_local_service(model_name: str, console=None):
    port = _SERVICE_PORTS.get(model_name)
    if port is None:
        if console: console.print(f"  [yellow]⚠[/yellow] {model_name}: unknown port, cannot stop")
        return
    # Resolve PIDs first so we can report truthfully instead of trusting the
    # exit status of a pipeline whose head may have matched nothing.
    found = subprocess.run(f"lsof -ti:{port}", shell=True, capture_output=True, text=True)
    pids = [p for p in found.stdout.split() if p.strip()]
    if not pids:
        if console:
            console.print(f"  [dim]○[/dim] {model_name} service was not running (port {port}).")
        return
    subprocess.run(["kill", "-9"] + pids, capture_output=True)
    if console:
        console.print(f"  [green]✓[/green] {model_name} service stopped (port {port}, pid {', '.join(pids)}).")


# ─── Ensemble Model Prediction ────────────────────────────────────

_ep_module_cache: dict = {}

# The names run_local_ensemble and predict_local rely on. Checked after loading
# so that picking up a wrong or outdated ensemble_predict.py fails with a
# message that names the file, instead of an AttributeError several frames later
# that reads like the ensemble itself is broken.
_EP_REQUIRED_ATTRS = ("ENSEMBLE_TYPE_TO_TASK", "TASK_PKL", "run_task",
                      "determine_variant_type", "load_bundles")


def _load_ep_module():
    if "bundled" in _ep_module_cache:
        return _ep_module_cache["bundled"]
    # packaged_only: this is executable code, so it always comes from the
    # installed package, never from the user's model_dir.
    ep_path = str(get_bundled_model_path("ensemble_predict.py", packaged_only=True))
    model_dir = str(Path(ep_path).parent)
    if model_dir not in sys.path: sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location("ep_local", ep_path)
    ep = importlib.util.module_from_spec(spec)
    ep.__file__ = ep_path
    spec.loader.exec_module(ep)

    missing = [a for a in _EP_REQUIRED_ATTRS if not hasattr(ep, a)]
    if missing:
        raise RuntimeError(
            f"{ep_path} is not the MERGE ensemble module — it is missing "
            f"{', '.join(missing)}. This is almost always an ensemble_predict.py "
            "left behind by an older merge-cli, which used a different API and "
            "different model bundles. Remove or rename that file and reinstall "
            "merge-cli.")

    _ep_module_cache["bundled"] = ep
    return ep


def run_local_ensemble(pred_data, all_transcripts, ensemble_type=None, skip_shap=False, variant_info=None):
    """Score with the bundled MERGE model for the resolved task.

    `ensemble_type` may be any of coding / splice / noncoding / nonsnv (the
    ENSEMBLE_TYPE_TO_TASK aliases are accepted too). When omitted, the task is
    detected from the variant itself: non-SNV by allele length, else ANNOVAR.
    """
    ep = _load_ep_module()
    if not all_transcripts:
        all_transcripts = [{'annotations': {}, 'dl_models': {}}]

    if ensemble_type:
        task = ep.ENSEMBLE_TYPE_TO_TASK.get(ensemble_type, ensemble_type)
    else:
        task = ep.determine_variant_type(pred_data, all_transcripts)

    return ep.run_task(task, pred_data, all_transcripts,
                       skip_shap=skip_shap, variant_info=variant_info or '')


# ─── Pre-computed VCF Cache ───────────────────────────────────────

# VCF filenames mirror the server-side PRECOMPUTED_VCFS config in views.py
_PRECOMPUTED_VCF_NAMES = {
    "hg38": {
        "coding":    "coding_merged.vcf.gz",
        "splicing":  "splicing_merged.vcf.gz",
    },
    "hg19": {
        "coding":    "coding_merged_hg19.vcf.gz",
        "splicing":  "splicing_merged_hg19.vcf.gz",
    },
}


def get_precomputed_data_dir() -> str:
    cfg = _cfg()
    candidates = [
        cfg.get("precomputed_data_dir"),
        os.environ.get("MERGE_PRECOMPUTED_DIR"),
        str(Path.home() / "precom_data"),
        str(Path.home() / "precom_daa"),
        os.path.join(cfg.get("data_dir", ""), "precomputed"),
        cfg.get("data_dir"),
    ]
    marker_files = list(_PRECOMPUTED_VCF_NAMES["hg38"].values())
    fallback = None
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.expanduser(str(candidate))
        if fallback is None:
            fallback = candidate
        if any(os.path.exists(os.path.join(candidate, f)) for f in marker_files):
            return candidate
    return fallback or str(Path.home() / ".merge-data" / "precomputed")


def precomputed_file_status(data_dir: Optional[str] = None) -> dict:
    root = os.path.expanduser(data_dir or get_precomputed_data_dir())
    status = {}
    for genome, by_type in _PRECOMPUTED_VCF_NAMES.items():
        for source, fname in by_type.items():
            vcf_path = os.path.join(root, fname)
            tbi_path = vcf_path + ".tbi"
            status[f"{genome}_{source}"] = {
                "genome": genome,
                "source": source,
                "vcf": vcf_path,
                "tbi": tbi_path,
                "vcf_exists": os.path.exists(vcf_path),
                "tbi_exists": os.path.exists(tbi_path),
            }
    return status


def _clean_info_value(value):
    if value in (None, ".", ""):
        return None
    if isinstance(value, (int, float, bool)):
        return value
    f = _safe_float(value)
    return f if f is not None else str(value)


def precomputed_prediction_response(chrom: str, pos, ref: str, alt: str,
                                    genome: str, cached: dict) -> dict:
    """Turn a cache hit into a prediction payload.

    The cache supplies FEATURES only. Its baked-in ENSEMBLE field came from an
    older ensemble pipeline, so re-using it would silently return a score from a
    different model than a cache miss does — for some variants the two disagree
    completely (0.0% vs 75.7% on chr17:43092919 G>A). The MERGE score is always
    recomputed from the bundled models; the cache just saves the model calls.
    """
    ens = cached["ensemble"]
    subs = cached["sub_models"]
    return {
        "success": True,
        "from_cache": True,
        "prediction": {
            "input": {"chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt},
            "genome_version": genome,
            "genome_build": genome,
            "_variant_type": ens.get("variant_type"),
            "precomputed_hit": True,
            "precomputed_source": cached.get("precomputed_source"),
            "precomputed_scores": cached.get("precomputed_scores", {}),
            # Deliberately NOT setting "precomputed": the stale ENSEMBLE score
            # must not short-circuit scoring with the current models.
            "dbnsfp": {
                "annotations": {},
                "dl_models": {
                    "AlphaMissense": subs.get("AlphaMissense", {}),
                    "ESM1b": subs.get("ESM1b", {}),
                },
            },
            "alphagenome": subs.get("alphagenome"),
            "hyenadna": subs.get("hyenadna"),
            "nt": subs.get("nt"),
            "gpn_msa": subs.get("gpn_msa"),
            "evo2": subs.get("evo2"),
            "enformer": subs.get("enformer"),
            "errors": {},
        },
    }


# The precomputed VCF cache is disabled until the tables are regenerated for the
# current models. The shipped tables predate them: they carry the older feature
# set (no Carbon / GENERATOR(-v2) / NT-v2 / Enformer) and a baked-in ENSEMBLE
# score from the previous ensemble pipeline, which disagrees with the bundled
# models on real variants — chr17:43092919 G>A reads 0.0% from the cache and
# 75.7% when scored properly. Serving that silently would be worse than being
# slower, so every variant is scored with the current models.
#
# To re-enable once the new tables exist: set this to True (or export
# MERGE_ENABLE_PRECOMPUTED=1) and make sure the VCFs carry the full INFO field
# set that build_feature_values() reads.
PRECOMPUTED_CACHE_ENABLED = os.environ.get("MERGE_ENABLE_PRECOMPUTED", "").lower() in ("1", "true", "yes")


def query_precomputed_local(chrom: str, pos, ref: str, alt: str,
                            genome: str = "hg38") -> Optional[dict]:
    if not PRECOMPUTED_CACHE_ENABLED:
        return None
    return _query_precomputed_local_impl(chrom, pos, ref, alt, genome)


def _query_precomputed_local_impl(chrom: str, pos, ref: str, alt: str,
                                  genome: str = "hg38") -> Optional[dict]:
    """
    Query the local pre-computed VCF cache files (same format as the server).
    Splicing is checked first to prevent coding-model mis-classification.

    Returns the same dict shape as views.query_precomputed_vcf on hit, or
    None on miss / unavailable.

    INFO fields expected (set by the pre-computation pipeline):
      ENSEMBLE  – MERGE score (0-1)
      INTERP    – interpretation label (spaces encoded as underscores)
      HYENA     – HyenaDNA score
      NT        – Nucleotide Transformer score
      GPN       – GPN-MSA score
      EVO2      – Evo2 LLR score
      AM        – AlphaMissense score
      ESM1B     – ESM1b score
      AG_SPLICE           – AlphaGenome splicing composite (splice variants)
      AG_RAW_MEAN / MAX / MIN         – AlphaGenome raw scores (coding)
      AG_Q_MEAN / MAX / MIN           – AlphaGenome quantile scores (coding)
    """
    pysam = _import_pysam()
    if pysam is None:
        return None  # optional dependency absent; the caller scores for real

    data_dir = get_precomputed_data_dir()
    names = _PRECOMPUTED_VCF_NAMES.get(genome, {})

    for vtype in ("splicing", "coding"):   # splicing first — mirrors server logic
        fname = names.get(vtype)
        if not fname:
            continue
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        if not os.path.exists(fpath + ".tbi"):
            logger.debug(f"[precomputed_local] missing tabix index: {fpath}.tbi")
            continue

        try:
            old_verbosity = pysam.set_verbosity(0)
            try:
                tbx = pysam.TabixFile(fpath)
            finally:
                pysam.set_verbosity(old_verbosity)
            pos_int = int(pos)
            chrom_queries = [chrom,
                             ("chr" + chrom) if not chrom.startswith("chr") else chrom.lstrip("chr")]
            records = []
            for cq in chrom_queries:
                try:
                    records = list(tbx.fetch(cq, pos_int - 1, pos_int))
                    if records:
                        break
                except Exception:
                    continue
            tbx.close()

            for record in records:
                parts = record.split("\t")
                if len(parts) < 8 or parts[3] != ref or parts[4] != alt:
                    continue

                info: dict = {}
                for item in parts[7].split(";"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        info[k] = v

                if "ENSEMBLE" not in info:
                    continue

                score_val  = _safe_float(info["ENSEMBLE"]) or 0.0
                interp_val = info.get("INTERP", "Precomputed").replace("_", " ")
                precomputed_scores = {k: _clean_info_value(v) for k, v in info.items()}

                sub_models = {
                    "alphagenome": {"statistics": {}},
                    "hyenadna":    {"score": _safe_float(info.get("HYENA"))},
                    "nt":          {"score": _safe_float(info.get("NT"))},
                    "gpn_msa":     {"score": _safe_float(info.get("GPN"))},
                    "evo2": {
                        "llr_score":      _safe_float(info.get("EVO2")),
                        "ref_score":      None,
                        "var_score":      None,
                        "context_length": 8192,
                        "interpretation": "Precomputed cache",
                        "score_class": (
                            "benign"     if (_safe_float(info.get("EVO2")) or 0) > 0
                            else "deleterious" if (_safe_float(info.get("EVO2")) or 0) < 0
                            else "unknown"
                        ),
                    },
                    "AlphaMissense": {"score": _safe_float(info.get("AM"))},
                    "ESM1b":         {"score": _safe_float(info.get("ESM1B"))},
                    "enformer":      {"score": _safe_float(info.get("ENFORMER"))} if "ENFORMER" in info else None,
                }

                if vtype == "splicing":
                    sub_models["alphagenome"]["statistics"]["alphagenome_splicing"] = \
                        _safe_float(info.get("AG_SPLICE"))
                else:
                    sub_models["alphagenome"]["statistics"].update({
                        "raw_score_mean":      _safe_float(info.get("AG_RAW_MEAN")),
                        "raw_score_max":       _safe_float(info.get("AG_RAW_MAX")),
                        "raw_score_min":       _safe_float(info.get("AG_RAW_MIN")),
                        "quantile_score_mean": _safe_float(info.get("AG_Q_MEAN")),
                        "quantile_score_max":  _safe_float(info.get("AG_Q_MAX")),
                        "quantile_score_min":  _safe_float(info.get("AG_Q_MIN")),
                    })

                return {
                    "precomputed_hit": True,
                    "precomputed_source": vtype,
                    "genome_build": genome,
                    "precomputed_scores": precomputed_scores,
                    "ensemble": {
                        "score":        score_val,
                        "interpretation": {
                            "label":  interp_val,
                            "badge":  "⚡ Cache",
                            "color":  "#ff6b6b" if score_val > 0.5 else "#6bcf7f",
                        },
                        "variant_type": "splice" if vtype == "splicing" else "coding",
                        "model_used":   ("Splice_ClinVar_GnomAD"
                                         if vtype == "splicing" else "ClinVar"),
                        "from_cache":        True,
                        "precomputed_hit":   True,
                        "precomputed_source": vtype,
                        "genome_build":      genome,
                        "precomputed_scores": precomputed_scores,
                        "features_raw_aligned": {},
                        "imputed_features":     {},
                        "shap_plot":            None,
                    },
                    "sub_models": sub_models,
                }

        except Exception as exc:
            logger.debug(f"[precomputed_local] {vtype}/{genome}: {exc}")
            continue

    return None   # cache miss


# ─── Top-level Prediction Entry ───────────────────────────────────

def call_dna_lm(model_name, chrom, pos, ref, alt, genome="hg38", timeout=600) -> dict:
    """Score one DNA-LM against this machine's service for that model."""
    return call_local_service(model_name, chrom, pos, ref, alt, genome, timeout=timeout)


# Which cache INFO field carries each model's score. A regenerated cache that
# includes these fields will be used for those models too; the tables shipped
# so far stop at the older set.
_CACHE_INFO_FIELD = {
    "use_generator":    "GENERATOR",
    "use_generator_v2": "GENERATOR_V2",
    "use_nt_v2":        "NT_V2",
    "use_carbon":       "CARBON",
    "use_enformer":     "ENFORMER_MAX_ABS",
}


def _cache_missing_requested_models(cached: dict, flags: dict) -> bool:
    """True when the cached record lacks a model the caller asked for."""
    info = cached.get("precomputed_scores") or {}
    for flag, field in _CACHE_INFO_FIELD.items():
        if flags.get(flag, True) and info.get(field) is None:
            return True
    return False


def predict_local(chrom, pos, ref, alt, genome="hg38", ensemble_type=None, **flags) -> dict:
    ep = _load_ep_module()
    non_snv = ep.is_non_snv(ref, alt)
    vtype = ensemble_type or ("nonsnv" if non_snv
                              else get_variant_type(chrom, pos, ref, alt, genome))

    # ── Pre-computed cache fast path ──────────────────────────────
    # The cache holds SNVs only, and only the older feature set (no Carbon,
    # GENERATOR(-v2), NT-v2 or Enformer). It is used when the request does not
    # need any of those; otherwise the models are called for real.
    cached = None if non_snv else query_precomputed_local(chrom, pos, ref, alt, genome)
    if cached and _cache_missing_requested_models(cached, flags):
        # The cached record predates some of the requested models, so it cannot
        # answer this query on its own. Score everything instead of silently
        # returning a partial feature set. (Keyed off what the record actually
        # contains, so a regenerated cache carrying these fields is used again.)
        cached = None
    if cached:
        logger.debug(f"[precomputed_local] cache hit: {chrom}:{pos} {ref}>{alt}")
        return precomputed_prediction_response(chrom, pos, ref, alt, genome, cached)
    # ─────────────────────────────────────────────────────────────

    tasks = {"dbnsfp": lambda: query_dbnsfp_local(chrom, pos, ref, alt, genome)}
    if flags.get("use_hyenadna", True):
        tasks["hyenadna"] = lambda: call_local_service("hyenadna", chrom, pos, ref, alt, genome)
    if flags.get("use_nt", True):
        tasks["nt"] = lambda: call_local_service("nt", chrom, pos, ref, alt, genome)
    if flags.get("use_gpn_msa", True):
        tasks["gpn_msa"] = lambda: _query_remote("gpn_msa", chrom=chrom, pos=pos, ref=ref, alt=alt)
    if flags.get("use_enformer", True):
        tasks["enformer"] = lambda: call_local_service("enformer", chrom, pos, ref, alt, genome)

    # ── Evo2 variants ─────────────────────────────────────────────
    # The ensemble panels use evo2_7b / evo2_7b_base / evo2_1b_base as separate
    # features. Score whichever variants this machine can serve; the rest stay
    # absent and the pipeline median-imputes them.
    if flags.get("use_evo2", True):
        evo2_variants = [v for v in (flags.get("evo2_models") or ["evo2_7b"])
                         if v in EVO2_VARIANTS]
        # FP8-only variants cannot run on older cards: asking for them there
        # costs a model reload and returns an error instead of a score, so drop
        # them and let the ensemble impute those features.
        if not gpu_supports_fp8():
            runnable = [v for v in evo2_variants if not EVO2_VARIANTS[v]["needs_fp8"]]
            for v in evo2_variants:
                if v not in runnable:
                    logger.debug("[evo2] skipping %s: GPU lacks FP8 (compute >= 8.9)", v)
            evo2_variants = runnable

        def _evo2_task():
            # One service, one GPU-resident model: ask for the variants one at a
            # time. Issuing them concurrently would make the service thrash,
            # reloading a 7B model per in-flight request.
            out = {}
            for v in evo2_variants:
                out[v] = call_local_service("evo2", chrom, pos, ref, alt, genome,
                                            timeout=900,
                                            extra={"model": v, "model_name": v})
            return {"_evo2_results": out}
        if evo2_variants:
            tasks["_evo2"] = _evo2_task

    # ── DNA-LMs, each on its own local service ────────────────────
    for key, flag in (("carbon", "use_carbon"), ("generator", "use_generator"),
                      ("generator_v2", "use_generator_v2"), ("nt_v2", "use_nt_v2")):
        if flags.get(flag, False):
            tasks[key] = (lambda k=key:
                          call_dna_lm(k, chrom, pos, ref, alt, genome))

    def _ag_task():
        res = call_local_service("alphagenome", chrom, pos, ref, alt, genome)
        if "error" in res: return res
        # Pass statistics through directly without renaming keys
        #   ensemble_predict.py reads fields by their correct key names
        return {"statistics": res.get("result", {}).get("statistics") or {}}
    if flags.get("use_alphagenome", True):
        tasks["alphagenome"] = _ag_task

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn): k for k, fn in tasks.items()}
        for f in as_completed(futures):
            key = futures[f]
            try:
                results[key] = f.result()
            except Exception as exc:
                results.setdefault("errors", {})[key] = str(exc)

    # Flatten the serialised Evo2 variant results into top-level keys
    # (evo2_7b / evo2_7b_base / evo2_1b_base) the feature mapper reads.
    evo2_bundle = results.pop("_evo2", {}) or {}
    for _k, _v in (evo2_bundle.get("_evo2_results") or {}).items():
        results[_k] = _v
        # Keep a legacy 'evo2' key pointing at the primary variant so existing
        # output/rendering paths keep working.
        if _k == "evo2_7b":
            results.setdefault("evo2", _v)

    errors = results.setdefault("errors", {})
    for key, val in list(results.items()):
        if key == "errors":
            continue
        if isinstance(val, dict) and val.get("error"):
            errors[key] = val["error"]

    prediction = {"input": {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt},
                  "genome_version": genome, "genome_build": genome,
                  "precomputed_hit": False, "precomputed_source": None,
                  "_variant_type": vtype, **results}
    return {"success": True, "prediction": prediction}


def predict_local_batch(vcf_path: str, genome: str = "hg38",
                        output_path: Optional[str] = None,
                        ensemble_type: Optional[str] = None,
                        console=None, **model_flags) -> str:
    import csv
    def _log(m):
        if console: console.print(m)

    variants = []
    with open(vcf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) < 4: parts = line.split()
            if len(parts) < 4: continue
            chrom = parts[0] if parts[0].startswith("chr") else f"chr{parts[0]}"
            try: pos = int(parts[1])
            except ValueError: continue
            ref = parts[3] if len(parts) > 3 else parts[2]
            alt = parts[4] if len(parts) > 4 else parts[3]
            variants.append((chrom, pos, ref, alt))

    if not variants: raise ValueError(f"No valid variants found in VCF file: {vcf_path}")
    _log(f"\n[bold]Local Batch Prediction[/bold]: {len(variants)} variants total")
    if output_path is None:
        output_path = vcf_path.replace(".vcf", "").replace(".txt", "") + "_merge_local.csv"

    results_rows = []
    for i, (chrom, pos, ref, alt) in enumerate(variants, 1):
        _log(f"  [{i}/{len(variants)}] {chrom}:{pos} {ref}>{alt}…")
        try:
            vtype = get_variant_type(chrom, pos, ref, alt, genome, ensemble_type)

            # ── Pre-computed cache fast path ──────────────────────
            # The cache holds SNVs scored with the older feature set only, so a
            # non-SNV or an opt-in DNA-LM request has to take the full path.
            cached = (None if is_non_snv(ref, alt)
                      else query_precomputed_local(chrom, pos, ref, alt, genome))
            if cached and _cache_missing_requested_models(cached, model_flags):
                cached = None
            if cached:
                _log(f"    ⚡ Pre-computed cache hit")
                ens  = cached["ensemble"]
                subs = cached["sub_models"]
                ag_stats = (subs.get("alphagenome") or {}).get("statistics", {})
                row = {
                    "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                    "genome":         genome,
                    "variant_type":   ens.get("variant_type", vtype),
                    "gene":           "-",
                    "transcript":     "-",
                    "merge_ensemble_score": ens.get("score"),
                    "merge_label":    (ens.get("interpretation") or {}).get("label"),
                    "from_cache":     True,
                    "alphamissense_score": (subs.get("AlphaMissense") or {}).get("score"),
                    "esm1b_score":         (subs.get("ESM1b") or {}).get("score"),
                    "gpn_msa_score":  (subs.get("gpn_msa") or {}).get("score"),
                    "hyenadna_score": (subs.get("hyenadna") or {}).get("score"),
                    "nt_score":       (subs.get("nt") or {}).get("score"),
                    "evo2_score":     (subs.get("evo2") or {}).get("llr_score"),
                    # Not present in the precomputed cache; kept so both row
                    # shapes share one column set for the CSV writer.
                    "evo2_7b_base_score": None,
                    "evo2_1b_base_score": None,
                    "carbon_score":       None,
                    "generator_score":    None,
                    "generator_v2_score": None,
                    "nt_v2_score":        None,
                    "alphagenome_raw_max":       ag_stats.get("raw_score_max"),
                    "alphagenome_raw_min":       ag_stats.get("raw_score_min"),
                    "alphagenome_raw_mean":      ag_stats.get("raw_score_mean"),
                    "alphagenome_quantile_max":  ag_stats.get("quantile_score_max"),
                    "alphagenome_quantile_min":  ag_stats.get("quantile_score_min"),
                    "alphagenome_quantile_mean": ag_stats.get("quantile_score_mean"),
                    "alphagenome_splicing":      ag_stats.get("alphagenome_splicing"),
                    "errors": "",
                }
                results_rows.append(row)
                continue
            # ─────────────────────────────────────────────────────
            resp  = predict_local(chrom, pos, ref, alt, genome,
                                  local_genome_path=None, ensemble_type=vtype, **model_flags)
            pred  = resp.get("prediction", {})
            pred["_variant_type"] = vtype

            ens_score = ens_label = None
            try:
                dbnsfp = pred.get("dbnsfp") or {}
                transcripts = dbnsfp.get("transcripts") or ([dbnsfp] if dbnsfp else [])
                ens_result, _ = run_local_ensemble(pred, transcripts, ensemble_type=vtype)
                if ens_result:
                    first = next(iter(ens_result.values()))
                    ens_score = first.get("score")
                    ens_label = (first.get("interpretation") or {}).get("label")
            except Exception: pass

            ag_stats = (pred.get("alphagenome", {}).get("statistics") or {})
            dbnsfp   = pred.get("dbnsfp") or {}
            ann_raw  = dbnsfp.get("annotations") or []
            ann      = (ann_raw[0] if isinstance(ann_raw, list) and ann_raw
                        else ann_raw if isinstance(ann_raw, dict) else {})
            dlm_raw  = dbnsfp.get("dl_models") or []
            dlm      = (dlm_raw[0] if isinstance(dlm_raw, list) and dlm_raw
                        else dlm_raw if isinstance(dlm_raw, dict) else {})

            row = {
                "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                "genome": genome,
                "variant_type": vtype,
                "gene": ann.get("genename", "-"),
                "transcript": ann.get("Ensembl_transcriptid", "-"),
                "merge_ensemble_score":  ens_score,
                "merge_label":           ens_label,
                "alphamissense_score": (dlm.get("AlphaMissense") or {}).get("score"),
                "esm1b_score":         (dlm.get("ESM1b") or {}).get("score"),
                "gpn_msa_score":  pred.get("gpn_msa", {}).get("score"),
                "hyenadna_score": pred.get("hyenadna", {}).get("score"),
                "nt_score":       pred.get("nt", {}).get("score"),
                "evo2_score":     _evo2_score_of(pred.get("evo2") or pred.get("evo2_7b")),
                "evo2_7b_base_score": _evo2_score_of(pred.get("evo2_7b_base")),
                "evo2_1b_base_score": _evo2_score_of(pred.get("evo2_1b_base")),
                "carbon_score":       (pred.get("carbon") or {}).get("score"),
                "generator_score":    (pred.get("generator") or {}).get("score"),
                "generator_v2_score": (pred.get("generator_v2") or {}).get("score"),
                "nt_v2_score":        (pred.get("nt_v2") or {}).get("score"),
                "alphagenome_raw_max":      ag_stats.get("alphagenome_raw_score_max"),
                "alphagenome_raw_min":      ag_stats.get("alphagenome_raw_score_min"),
                "alphagenome_raw_mean":     ag_stats.get("alphagenome_raw_score_mean"),
                "alphagenome_quantile_max": ag_stats.get("alphagenome_quantile_score_max"),
                "alphagenome_quantile_min": ag_stats.get("alphagenome_quantile_score_min"),
                "alphagenome_quantile_mean":ag_stats.get("alphagenome_quantile_score_mean"),
                "alphagenome_splicing":     ag_stats.get("alphagenome_splicing"),
                "errors": ";".join(
                    f"{k}:{v}" for k, v in pred.get("errors", {}).items() if v and str(v) != "None"
                ),
            }
            results_rows.append(row)
        except Exception as e:
            _log(f"  [red]✗ Prediction failed: {e}[/red]")

    if results_rows:
        # Union of every row's keys, first-seen order: a cache-hit row and a
        # fully-computed row carry different fields, and keying off row 0 alone
        # would silently drop whatever the first row happened not to have.
        fieldnames = []
        for row in results_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    _log(f"\n[bold green]✓ Batch prediction complete. Results saved to: {output_path}[/bold green]")
    return output_path


def check_local_files() -> dict:
    cfg = _cfg()
    data_dir = cfg["data_dir"]
    precomputed_dir = get_precomputed_data_dir()
    evo2_model_name = cfg.get("evo2_model_name", "evo2_7b")
    evo2_model_dir  = _get_evo2_model_dir()
    evo2_weight_path = (
        evo2_model_dir if os.path.isfile(evo2_model_dir)
        else os.path.join(evo2_model_dir, f"{evo2_model_name}.pt")
    )

    files = {
        "dbNSFP (hg38)": os.path.join(data_dir, "dbNSFP5.3a_grch38.gz"),
        "dbNSFP (hg19)": os.path.join(data_dir, "dbNSFP5.3a_grch37.gz"),
        "GPN-MSA":       os.path.join(data_dir, "scores.tsv.bgz"),
        # ── Pre-computed VCF cache ──────────────────────────────────
        "Pre-computed coding (hg38)":    os.path.join(precomputed_dir, "coding_merged.vcf.gz"),
        "Pre-computed splicing (hg38)":  os.path.join(precomputed_dir, "splicing_merged.vcf.gz"),
        "Pre-computed coding (hg19)":    os.path.join(precomputed_dir, "coding_merged_hg19.vcf.gz"),
        "Pre-computed splicing (hg19)":  os.path.join(precomputed_dir, "splicing_merged_hg19.vcf.gz"),
        # ───────────────────────────────────────────────────────────
        f"Evo2 local weights ({evo2_model_name}, optional)": evo2_weight_path,
    }
    result = {}
    for label, path in files.items():
        exists  = os.path.exists(path)
        size_gb = round(os.path.getsize(path) / 1e9, 2) if exists else None
        result[label] = {"exists": exists, "path": path, "size_gb": size_gb}

    for key, fname in _BUNDLED_MODELS.items():
        label = f"Ensemble model ({fname})"
        try:
            path    = str(get_bundled_model_path(fname))
            exists  = os.path.exists(path)
            size_gb = round(os.path.getsize(path) / 1e9, 4) if exists else None
            result[label] = {"exists": exists, "path": path, "size_gb": size_gb, "bundled": True}
        except FileNotFoundError:
            result[label] = {"exists": False, "path": "(bundled package)", "size_gb": None, "bundled": True}

    return result
