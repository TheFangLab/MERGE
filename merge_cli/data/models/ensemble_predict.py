"""
ensemble_predict.py  —  MERGE ensemble + the MERGE + AlphaGenome average
==============================================================================
Loads the four deployment bundles MERGE_noAlphaGenome_{coding,splicing,noncoding,
nonsnv}.pkl (joblib dict: {'pipeline', 'features', ...}). Preprocessing (median
imputation → z-score scaling → classifier) is EMBEDDED in the pipeline, so raw
model-score features are fed directly; missing features are median-imputed by
the pipeline.

What comes back per variant:

  score       the ensemble probability, for ALL FOUR tasks. Its feature panels
              contain NO alphagenome_* column — no AlphaGenome output was used
              to train it.
  merge_alphagenome_average
              0.5 x score + 0.5 x the percentile of one AlphaGenome column
              against the MERGE benchmark variants
              (alphagenome_reference_quantiles.json): a label-free empirical
              CDF, not a fitted model, so no model is trained on AlphaGenome
              output. The column is fixed per variant type --
              alphagenome_splicing for splicing, alphagenome_raw_score_max for
              non-coding. Offered for SPLICING and NON-CODING ONLY. Coding and non-SNV return None
              with status 'not_offered_for_this_variant_type': their best single
              AlphaGenome feature reaches only 0.571 / 0.512, near enough to
              chance that the average would move the score without adding
              evidence. On splicing and non-coding it is None whenever
              AlphaGenome has no value for the variant.

AlphaGenome feature extraction is unchanged and still runs for every task:
build_feature_values reports every alphagenome_* summary as before. What changed
is that those values no longer enter the ensemble's inputs — they only feed the
post-hoc average.

The two scores are NOT on the same scale: averaging two probabilities pulls the
result toward 0.5, so merge_alphagenome_average has its own per-task
interpretation cut-points (merge_final_thresholds.json), call-rate matched to
the ensemble bands on held-out predictions. Reusing the ensemble's
0.8/0.6/0.4/0.2 bands on the average would put a quarter of splicing variants in
a different band than the ensemble reports for the same variant.

Task selection:
  * non-SNV (indel / MNV / delins)  -> 'nonsnv' model  (checked first, by REF/ALT length)
  * else ANNOVAR gene annotation    -> 'coding' | 'splice' | 'noncoding'

Public API:
  run_ensemble(pred_data, all_transcripts, skip_shap=False) -> ({model_key: result}, err)
  run_task(task, pred_data, all_transcripts, ...)           -> ({model_key: result}, err)
  extract_features(pred_data, all_transcripts, is_splice=False) -> {training_col: value}
  load_bundles() -> {task: bundle}

Feature column names are the unified training columns (see MODEL_CARD.md). The
CLI's live model outputs are mapped onto them in build_feature_values().

This mirrors the website implementation (merge.fanglab.cn) so that CLI and web
produce identical MERGE scores. One intentional difference: the CLI's AlphaGenome
service emits statistics keys WITH an 'alphagenome_' prefix while the website's
emits them without, so both spellings are accepted here.
"""
import os
import io
import json
import base64
import pickle
import logging
import subprocess
import tempfile
import warnings
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# 1. Bundle paths
# ─────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get('MERGE_MODEL_DIR', _THIS_DIR)

TASK_PKL = {
    'coding':    'MERGE_noAlphaGenome_coding.pkl',
    'splice':    'MERGE_noAlphaGenome_splicing.pkl',
    'noncoding': 'MERGE_noAlphaGenome_noncoding.pkl',
    'nonsnv':    'MERGE_noAlphaGenome_nonsnv.pkl',
}

# The AlphaGenome half of the average: every distinct AlphaGenome score observed in
# the MERGE benchmark paired with its mid-rank percentile. No labels, no fitted
# parameters -- see MERGE-benchmark 05_results_alphagenome_average/README.md.
# A missing file is not fatal: the ensemble score still works and
# merge_alphagenome_average is reported as unavailable.
# Only splicing and non-coding offer the average. On coding and non-SNV the best
# single AlphaGenome feature scores AUROC 0.571 / 0.512 on held-out data -- close
# enough to chance that averaging it in only pulls the ensemble score toward 0.5,
# so those two tasks report the ensemble score alone.
AG_AVERAGE_TASKS = ('splice', 'noncoding')

AG_REFERENCE_JSON = 'alphagenome_reference_quantiles.json'
TASK_AG_REFERENCE_KEY = {
    'splice':    'splicing',
    'noncoding': 'noncoding',
}

# Weight of the ensemble in the average. Fixed at 0.5 — a 1:1 mean, not a fitted
# blend, so no held-out data was used to tune it.
MERGE_AG_AVERAGE_ENSEMBLE_WEIGHT = 0.5

THRESHOLDS_JSON = 'merge_final_thresholds.json'

# Human-facing model key returned to the caller, per task.
TASK_MODEL_KEY = {
    'coding':    'MERGE_Coding',
    'splice':    'MERGE_Splicing',
    'noncoding': 'MERGE_NonCoding',
    'nonsnv':    'MERGE_NonSNV',
}

# CLI --ensemble-type value -> internal task name.
ENSEMBLE_TYPE_TO_TASK = {
    'coding':    'coding',
    'splice':    'splice',
    'splicing':  'splice',
    'noncoding': 'noncoding',
    'nonsnv':    'nonsnv',
    'non_snv':   'nonsnv',
}

_bundles: dict = {}
_load_errors: dict = {}
_ag_bundles: dict = {}
_ag_load_errors: dict = {}
_thresholds: dict = {}


def _patch_lr(obj):
    """Pickles of LogisticRegression made with sklearn>=1.5 lack `multi_class`
    when loaded under older sklearn. Restore it so predict_proba works."""
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return obj
    steps = []
    if hasattr(obj, 'steps'):
        steps = [s for _, s in obj.steps]
    for target in [obj] + steps:
        if isinstance(target, LogisticRegression) and not hasattr(target, 'multi_class'):
            target.multi_class = 'auto'
    return obj


def _bundle_path(fn: str) -> str:
    """Resolve a bundle filename: MERGE_MODEL_DIR first, else next to this file
    (the pip-bundled location)."""
    p = os.path.join(MODEL_DIR, fn)
    if os.path.exists(p):
        return p
    return os.path.join(_THIS_DIR, fn)


def _read_bundle(path: str):
    """Load one joblib bundle. Returns (raw, error_message); one of them is None."""
    if not os.path.exists(path):
        return None, f"bundle not found: {path}"
    errs = []
    first_exc = None
    for name, loader in (
        ('joblib', lambda p: __import__('joblib').load(p)),
        ('pickle', lambda p: pickle.load(open(p, 'rb'))),
    ):
        try:
            with warnings.catch_warnings():
                # The non-SNV bundle was pickled with xgboost 3.4.1; Python 3.11
                # installs top out at xgboost 3.2.0, which warns on loading a
                # newer pickle. Predictions on all 36,862 benchmark non-SNVs are
                # identical under the two versions, so the warning is noise.
                warnings.filterwarnings('ignore', message=r'(?s).*older version of XGBoost',
                                        category=UserWarning)
                # Likewise scikit-learn warns on any version difference from the
                # 1.9.0 the bundles were pickled with; 1.9.1 gives identical
                # predictions on every benchmark variant, and
                # tests/test_sklearn_compat.py guards loading and scoring.
                warnings.filterwarnings('ignore', message=r'Trying to unpickle estimator')
                return loader(path), None
        except Exception as exc:
            import traceback
            if first_exc is None:
                first_exc = exc
            errs.append(f"[{name}]\n{traceback.format_exc()}")
    # Report joblib's failure, not the pickle fallback's. These bundles are
    # joblib archives, so the fallback always dies with a cryptic "invalid load
    # key" that hides the real cause — most often a missing estimator library
    # such as catboost.
    hint = ''
    if isinstance(first_exc, ModuleNotFoundError):
        missing = getattr(first_exc, 'name', None) or str(first_exc)
        hint = (f" — the '{missing}' package is required to load this model. "
                f"Install it with: pip install {missing}")
    return None, f"{type(first_exc).__name__}: {first_exc}{hint}\n\n" + "\n".join(errs)


def load_bundles() -> dict:
    global _bundles, _load_errors
    for task, fn in TASK_PKL.items():
        if task in _bundles:
            continue
        path = _bundle_path(fn)
        raw, err = _read_bundle(path)
        if raw is None:
            _load_errors[task] = err
            continue
        try:
            if not isinstance(raw, dict) or 'pipeline' not in raw or 'features' not in raw:
                raise RuntimeError("bundle missing 'pipeline'/'features' keys")
            feats = list(raw['features'])
            # Guard the whole point of these bundles: an alphagenome_* column in
            # an ensemble panel would mean an AlphaGenome-trained model got
            # shipped again. Refuse it rather than score with it.
            leaked = [c for c in feats if c.lower().startswith('alphagenome')]
            if leaked:
                raise RuntimeError(
                    f"bundle {fn} is not AlphaGenome-free: panel contains {leaked}")
            _patch_lr(raw['pipeline'])
            _bundles[task] = {
                'pipeline': raw['pipeline'],
                'features': feats,
                'algorithm': raw.get('algorithm'),
                'feature_selection': raw.get('feature_selection'),
                'sklearn_version': (raw.get('sklearn_version')
                                    or (raw.get('software_versions') or {}).get('scikit_learn')),
            }
            logger.info("[MERGE] loaded %s (%d features, algo=%s)",
                        fn, len(feats), raw.get('algorithm'))
        except Exception:
            import traceback
            _load_errors[task] = traceback.format_exc()
    return _bundles


def load_ag_bundles() -> dict:
    """Load the AlphaGenome reference percentile table. Absence is tolerated:
    without it the ensemble score is still returned and merge_alphagenome_average is None."""
    global _ag_bundles, _ag_load_errors
    if _ag_bundles or _ag_load_errors:
        return _ag_bundles
    path = _bundle_path(AG_REFERENCE_JSON)
    try:
        with open(path) as fh:
            ref = json.load(fh)
        if ref.get('labels_used') is not False:
            raise RuntimeError('reference table does not declare labels_used = false')
        for task, key in TASK_AG_REFERENCE_KEY.items():
            entry = ref['tasks'][key]
            scores = np.asarray(entry['scores'], dtype=float)
            pct = np.asarray(entry['percentiles'], dtype=float)
            if scores.size < 2 or scores.size != pct.size or np.any(np.diff(scores) <= 0):
                raise RuntimeError(f"malformed reference table for {key}")
            _ag_bundles[task] = {'feature': entry['feature'], 'scores': scores,
                                 'percentiles': pct}
    except Exception as exc:
        _ag_bundles = {}
        for task in TASK_AG_REFERENCE_KEY:
            _ag_load_errors[task] = f"{type(exc).__name__}: {exc} ({path})"
    return _ag_bundles


def _first_auroc(raw: dict):
    """Held-out AUROC of the selected AlphaGenome feature, from the bundle's own
    selection table. Used to warn when the component is near chance."""
    table = raw.get('selection_table')
    if isinstance(table, list) and table and isinstance(table[0], dict):
        return table[0].get('auroc_mean')
    return raw.get('oof_auroc')


def load_merge_alphagenome_average_thresholds() -> dict:
    """Per-task interpretation cut-points for merge_alphagenome_average, in descending order."""
    global _thresholds
    if _thresholds:
        return _thresholds
    path = _bundle_path(THRESHOLDS_JSON)
    try:
        with open(path, encoding='utf-8') as fh:
            _thresholds = json.load(fh)
    except Exception as exc:
        logger.warning("[MERGE] merge_alphagenome_average thresholds unavailable (%s): %s", path, exc)
        _thresholds = {}
    return _thresholds


# ─────────────────────────────────────────────────────────────
# 2. Feature extraction — map live model outputs -> training columns
# ─────────────────────────────────────────────────────────────
_MISSING = (None, '.', 'N/A', '', 'NA', '-')


def _f(value) -> float:
    """Coerce to float, mapping empties/sentinels to NaN.

    dbNSFP ships its per-transcript columns as a single ';'-separated string
    (absent positions written as '.'), e.g. AlphaMissense '.;.;0.0424;.;0.0403'.
    When handed one of those, average the valid positions. That is exactly the
    value the website ends up with: it explodes the dbNSFP row into one dict per
    transcript (`_parse_transcripts_backend`, positional split on ';') and then
    build_feature_values() takes the cross-transcript mean. Callers that hand us
    the un-exploded row — the MCP server, `merge predict`, and the website's own
    /ensemble/ fallback — used to get NaN here and be silently median-imputed,
    which for AlphaMissense means 0.7536 (pathogenic side) replacing a real
    benign 0.04. See the BRCA1 rs799917 regression test.
    """
    if isinstance(value, str) and ';' in value:
        parts = [_f(p) for p in value.split(';')]
        parts = [p for p in parts if not np.isnan(p)]
        return float(np.mean(parts)) if parts else float('nan')
    try:
        if isinstance(value, str):
            value = value.strip()
        if value in _MISSING:
            return float('nan')
        return float(value)
    except Exception:
        return float('nan')


def _evo2_value(evo2_res) -> float:
    """Evo2 feature = mean_logP(REF) - mean_logP(ALT). Services return llr_score
    for SNV and score_mean for non-SNV — both are that quantity."""
    if not isinstance(evo2_res, dict):
        return float('nan')
    for key in ('llr_score', 'score_mean', 'score'):
        v = evo2_res.get(key)
        if v is not None:
            return _f(v)
    return float('nan')


def _score_of(node) -> float:
    return _f(node.get('score')) if isinstance(node, dict) else float('nan')


def _ag(ag_stats: dict, name: str) -> float:
    """AlphaGenome statistics lookup accepting both key spellings: the CLI's
    local service emits 'alphagenome_raw_score_max', the website and the
    precomputed VCF cache emit 'raw_score_max'."""
    if not isinstance(ag_stats, dict):
        return float('nan')
    for key in (name, f'alphagenome_{name}'):
        if key in ag_stats:
            return _f(ag_stats.get(key))
    return float('nan')


def build_feature_values(pred_data: dict, all_transcripts: list, is_splice: bool = False) -> dict:
    """Return a dict keyed by the unified TRAINING column names. Missing values
    are NaN (the pipeline median-imputes them)."""
    pred_data = pred_data or {}
    all_transcripts = all_transcripts or []
    feats: dict = {}

    # --- AlphaMissense / ESM1b: per-transcript mean of dbNSFP dl_models ---
    # Two transcript shapes are in circulation: the website nests the model keys
    # under 'dl_models', while the CLI's query_dbnsfp_local puts them at the top
    # level of each transcript. Accept either, or the score would silently be NaN.
    am_vals, esm_vals = [], []
    for t in all_transcripts:
        if not isinstance(t, dict):
            continue
        dm = t.get('dl_models')
        if not isinstance(dm, dict):
            dm = t
        v_am = _f((dm.get('AlphaMissense') or {}).get('score'))
        if not np.isnan(v_am):
            am_vals.append(v_am)
        v_esm = _f((dm.get('ESM1b') or {}).get('score'))
        if not np.isnan(v_esm):
            esm_vals.append(v_esm)
    feats['AlphaMissense_score'] = float(np.mean(am_vals)) if am_vals else float('nan')
    feats['ESM1b_score'] = float(np.mean(esm_vals)) if esm_vals else float('nan')

    # --- lookups / single-value services ---
    feats['GPN_MSA_score'] = _score_of(pred_data.get('gpn_msa'))

    ag_stats = (pred_data.get('alphagenome') or {}).get('statistics') or {}
    feats['alphagenome_raw_score_max'] = _ag(ag_stats, 'raw_score_max')
    feats['alphagenome_raw_score_mean'] = _ag(ag_stats, 'raw_score_mean')
    feats['alphagenome_raw_score_min'] = _ag(ag_stats, 'raw_score_min')
    feats['alphagenome_quantile_score_max'] = _ag(ag_stats, 'quantile_score_max')
    feats['alphagenome_quantile_score_min'] = _ag(ag_stats, 'quantile_score_min')
    feats['alphagenome_quantile_score_mean'] = _ag(ag_stats, 'quantile_score_mean')
    feats['alphagenome_splicing'] = _ag(ag_stats, 'alphagenome_splicing')

    # --- Enformer 7-summary ---
    enf = pred_data.get('enformer') or {}
    for stat in ('absmean', 'mean', 'std', 'max_abs', 'top5_absmean', 'pos_sum', 'neg_sum'):
        col = f'enformer_{stat}'
        feats[col] = _f(enf.get(col, enf.get(stat)))

    # --- Evo2 variants (enriched keys evo2_7b / evo2_7b_base / evo2_1b_base) ---
    evo2_7b = pred_data.get('evo2_7b')
    evo2_7b_base = pred_data.get('evo2_7b_base')
    evo2_1b_base = pred_data.get('evo2_1b_base')
    if evo2_7b is None:  # legacy single-Evo2 pred_data
        legacy = pred_data.get('evo2') or {}
        mn = (legacy.get('model_name') or legacy.get('model') or 'evo2_7b')
        if mn == 'evo2_7b_base' and evo2_7b_base is None:
            evo2_7b_base = legacy
        elif mn == 'evo2_1b_base' and evo2_1b_base is None:
            evo2_1b_base = legacy
        else:
            evo2_7b = legacy
    feats['evo2_7b_score'] = _evo2_value(evo2_7b)
    feats['evo2_7b_base_score'] = _evo2_value(evo2_7b_base)
    # evo2_1b_base needs FP8 (compute capability >= 8.9). Where the GPU cannot
    # run it this stays NaN and the pipeline median-imputes it.
    feats['evo2_1b_base_score'] = _evo2_value(evo2_1b_base)

    # --- DNA-LM models (SUM-delta carbon; causal/masked others) ---
    feats['carbon_score'] = _score_of(pred_data.get('carbon'))
    feats['generator_score'] = _score_of(pred_data.get('generator'))
    feats['generator_v2_score'] = _score_of(pred_data.get('generator_v2'))
    feats['nt_v2_score'] = _score_of(pred_data.get('nt_v2'))

    nt_val = _score_of(pred_data.get('nt'))          # old NT (masked L2)
    feats['NT_score'] = nt_val                        # SNV-set naming
    feats['nt_old_score'] = nt_val                    # non-SNV naming (same model)

    hy_val = _score_of(pred_data.get('hyenadna'))
    feats['heyenadna_score'] = hy_val                 # SNV-set naming
    feats['hyenadna_score'] = hy_val                  # non-SNV naming

    return feats


# extract_features is the name callers import for raw-feature display.
def extract_features(pred_data: dict, all_transcripts: list, is_splice: bool = False) -> dict:
    return build_feature_values(pred_data, all_transcripts, is_splice=is_splice)


# ─────────────────────────────────────────────────────────────
# 3. Score interpretation
# ─────────────────────────────────────────────────────────────
# Same five labels as the web server. They name score ranges of the ensemble;
# they are not ACMG/AMP classifications and not calibrated probabilities.
_BANDS = (
    ('Pathogenic',             '#ff4444', '🔴'),
    ('Likely Pathogenic',      '#ff9500', '🟠'),
    ('Uncertain Significance', '#ffd93d', '🟡'),
    ('Likely Benign',          '#4ecb71', '🟢'),
    ('Benign',                 '#00cc55', '🟢'),
)

# Ensemble cut-points. Unchanged — the ensemble score means what it always did.
ENSEMBLE_CUTS = (0.8, 0.6, 0.4, 0.2)


def _band(score: float, cuts) -> dict:
    for i, cut in enumerate(cuts):
        if score >= cut:
            label, color, badge = _BANDS[i]
            return {'label': label, 'color': color, 'badge': badge}
    label, color, badge = _BANDS[-1]
    return {'label': label, 'color': color, 'badge': badge}


def interpret_score(score: float) -> dict:
    return _band(score, ENSEMBLE_CUTS)


# Threshold-file keys use the dataset names from the benchmark ('splicing'),
# while the task names here follow ANNOVAR's ('splice').
TASK_THRESHOLD_KEY = {
    'coding': 'coding', 'splice': 'splicing',
    'noncoding': 'noncoding', 'nonsnv': 'nonsnv',
}


def merge_alphagenome_average_cuts(task: str):
    """Cut-points for merge_alphagenome_average on `task`, or None when unavailable.

    NOT the ensemble's cut-points. The average mixes the ensemble with an
    AlphaGenome percentile, so it is on a different scale and the ensemble's
    bands would misfire on it. These
    are call-rate matched to the ensemble bands on chromosome-held-out
    predictions — no labels involved.
    """
    entry = load_merge_alphagenome_average_thresholds().get(TASK_THRESHOLD_KEY.get(task, task))
    if not entry:
        return None
    cuts = entry.get('cuts')
    return tuple(cuts) if cuts and len(cuts) == 4 else None


def interpret_merge_alphagenome_average(score: float, task: str) -> dict:
    """Band for a merge_alphagenome_average score. Falls back to the ensemble cut-points only
    if the threshold file is missing, and says so, rather than silently
    presenting a band that means something different."""
    cuts = merge_alphagenome_average_cuts(task)
    if cuts is None:
        out = _band(score, ENSEMBLE_CUTS)
        out['calibration'] = 'ensemble cut-points (merge_alphagenome_average thresholds unavailable)'
        return out
    out = _band(score, cuts)
    out['calibration'] = 'merge_alphagenome_average cut-points, call-rate matched to the ensemble bands'
    out['cuts'] = list(cuts)
    return out


# ─────────────────────────────────────────────────────────────
# 4. Variant type — non-SNV first, else ANNOVAR gene annotation
# ─────────────────────────────────────────────────────────────
ANNOVAR_DIR = os.environ.get('MERGE_ANNOVAR_DIR', os.path.join(_THIS_DIR, 'annovar'))


def is_non_snv(ref, alt) -> bool:
    """non-SNV = indel / MNV / delins. Either allele longer than one base, or an
    ANNOVAR-style '-' placeholder (which marks an insertion/deletion)."""
    if not ref or not alt:
        return False
    ref, alt = str(ref).strip(), str(alt).strip()
    if ref == '-' or alt == '-':
        return True
    return len(ref) != 1 or len(alt) != 1


def get_variant_type_by_annovar(chrom, pos, ref, alt, genome_version='hg19', annovar_dir=None):
    adir = annovar_dir or ANNOVAR_DIR
    pl = os.path.join(adir, 'annotate_variation.pl')
    if not os.path.exists(pl):
        logger.debug("[ANNOVAR] script not found: %s", pl)
        return None
    build = 'hg19' if genome_version == 'hg19' else 'hg38'
    if not str(chrom).startswith('chr'):
        chrom = 'chr' + str(chrom)
    try:
        pos_int = int(pos)
    except Exception:
        return None
    end_int = pos_int + len(ref) - 1 if ref != '-' else pos_int
    avinput = f"{chrom}\t{pos_int}\t{end_int}\t{ref}\t{alt}\n"
    perl_bin = shutil.which('perl') or '/usr/bin/perl'
    fin_path = None
    out_prefix = None
    try:
        project_tmp = os.path.join(adir, 'tmp')
        os.makedirs(project_tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', dir=project_tmp, suffix='.avinput', delete=False) as fin:
            fin.write(avinput)
            fin_path = fin.name
        out_prefix = fin_path + '_out'
        cmd = [perl_bin, pl, '-geneanno', '-dbtype', 'refGene',
               '-buildver', build, fin_path, os.path.join(adir, 'humandb'), '-out', out_prefix]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        vf_path = out_prefix + '.variant_function'
        if os.path.exists(vf_path):
            with open(vf_path) as fh:
                line = fh.readline().strip()
            if not line:
                return None
            func_field = line.split('\t')[0].lower()
            if 'splicing' in func_field: return 'splice'
            if 'exonic' in func_field:   return 'coding'
            return 'noncoding'
        return None
    except Exception as exc:
        logger.warning("[ANNOVAR] exception: %s", exc)
        return None
    finally:
        for p in [fin_path] + ([out_prefix + e for e in ('.variant_function', '.exonic_variant_function', '.log')] if out_prefix else []):
            if p and os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass


def _variant_coords(pred_data: dict, all_transcripts: list):
    inp = pred_data.get('input') or {}
    chrom = inp.get('chrom') or inp.get('chr')
    pos = inp.get('pos')
    ref = inp.get('ref')
    alt = inp.get('alt')
    if not (chrom and pos and ref and alt):
        db = pred_data.get('dbnsfp') or (all_transcripts[0] if all_transcripts else {}) or {}
        chrom = chrom or db.get('chr') or db.get('chrom')
        pos = pos or db.get('pos')
        ref = ref or db.get('ref')
        alt = alt or db.get('alt')
    gv = pred_data.get('genome_version', 'hg19')
    return chrom, pos, ref, alt, gv


def determine_variant_type(pred_data: dict, all_transcripts: list) -> str:
    chrom, pos, ref, alt, gv = _variant_coords(pred_data, all_transcripts)
    # non-SNV (indel / MNV / delins) always uses the dedicated nonsnv model.
    if is_non_snv(ref, alt):
        return 'nonsnv'
    if chrom and pos and ref and alt:
        vt = get_variant_type_by_annovar(chrom, pos, ref, alt, gv)
        if vt:
            return vt
    logger.warning("[MERGE] ANNOVAR type undetermined (%s:%s %s>%s); fallback noncoding",
                   chrom, pos, ref, alt)
    return 'noncoding'


# ─────────────────────────────────────────────────────────────
# 5. SHAP / importance plot (best-effort; embedded-pipeline aware)
# ─────────────────────────────────────────────────────────────
def _final_estimator(pipeline):
    if hasattr(pipeline, 'steps'):
        return pipeline.steps[-1][1]
    return pipeline


def _pre_transform(pipeline, X_df):
    """Apply everything except the final classifier (impute + scale). Pass an
    ndarray (the pipeline was fitted on arrays) to avoid feature-name warnings."""
    Xv = X_df.values if hasattr(X_df, 'values') else X_df
    if hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
        return pipeline[:-1].transform(Xv)
    return Xv


def _shap_1d(sv, n):
    if hasattr(sv, 'values'): sv = sv.values
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    arr = np.array(sv, dtype=float)
    if arr.ndim == 3: arr = arr[0, :, -1]
    elif arr.ndim == 2 and arr.shape[0] == 1: arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 2: arr = arr[:, 1]
    if arr.shape == (1,): arr = np.full(n, arr[0])
    return arr.astype(float)


def _shap_values(estimator, X_trans, n):
    try:
        import shap
    except Exception:
        return None
    bg = np.zeros((1, X_trans.shape[1]))
    for fn in (
        lambda: shap.TreeExplainer(estimator).shap_values(X_trans),
        lambda: shap.LinearExplainer(estimator, bg).shap_values(X_trans),
        lambda: shap.KernelExplainer(
            (lambda x: estimator.predict_proba(x)[:, 1]), bg
        ).shap_values(X_trans, nsamples=200, silent=True),
    ):
        try:
            return _shap_1d(fn(), n)
        except Exception:
            continue
    return None


def generate_shap_plot(bundle, X_df, feature_names, model_name='', variant_info=''):
    """SHAP bar chart; the values shown in labels are the RAW feature inputs."""
    try:
        pipeline = bundle['pipeline']
        estimator = _final_estimator(pipeline)
        X_trans = np.asarray(_pre_transform(pipeline, X_df))
        raw_vals = X_df.values[0]
        n = len(feature_names)
        sv = _shap_values(estimator, X_trans, n)

        if sv is None:
            if not hasattr(estimator, 'feature_importances_'):
                return None
            sv = np.array(estimator.feature_importances_, dtype=float)
            title = f"Feature Importance — {model_name} (SHAP unavailable)"
            signed = False
        else:
            title = f"SHAP Feature Contributions — {model_name}"
            signed = True

        order = np.argsort(np.abs(sv))[::-1][:min(n, 10)]
        names_s = [feature_names[i] for i in order]
        vals_s = sv[order]
        fval_s = raw_vals[order]
        y_labels = [f"{nm}  [{fv:.4f}]" if not np.isnan(fv) else f"{nm}  [NA]"
                    for nm, fv in zip(names_s, fval_s)]

        left_inch = max(3.8, max(len(lb) for lb in y_labels) * 0.072)
        fig_w = left_inch + 7.5
        n_bars = len(names_s)
        fig, ax = plt.subplots(figsize=(fig_w, max(5, 0.75 * n_bars + 2.2)))
        fig.patch.set_facecolor('#1a2a55'); ax.set_facecolor('#1e3c72')
        colors = (['#ff6b6b' if v > 0 else '#6bcf7f' for v in vals_s] if signed else '#60aaff')
        y = np.arange(n_bars)
        ax.barh(y, vals_s, color=colors, alpha=0.88, height=0.55, edgecolor='none')
        ax.invert_yaxis(); ax.set_yticks(y); ax.set_yticklabels(y_labels, color='white', fontsize=9.5)
        ax.set_xlabel('SHAP value (→ pathogenicity)' if signed else 'Feature importance',
                      color='#c8d8ff', fontsize=10)
        if variant_info:
            title = f"{variant_info} | {title}"
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=12)
        ax.axvline(0, color='white', linewidth=0.6, alpha=0.35)
        ax.tick_params(colors='white', length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.spines['bottom'].set_visible(True); ax.spines['bottom'].set_color('white'); ax.spines['bottom'].set_alpha(0.25)
        if signed:
            ax.legend(handles=[mpatches.Patch(color='#ff6b6b', label='↑ pathogenic'),
                               mpatches.Patch(color='#6bcf7f', label='↓ benign')],
                      loc='lower right', facecolor='#1e3c72', edgecolor='none',
                      labelcolor='white', fontsize=9)
        fig.subplots_adjust(left=left_inch / fig_w + 0.01, right=0.97, top=0.92, bottom=0.10)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor='#1a2a55')
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as exc:
        logger.warning("[MERGE] SHAP plot failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────
# 6. Main entry points
# ─────────────────────────────────────────────────────────────
def compute_merge_alphagenome_average(task: str, ensemble_score: float, feats: dict) -> dict:
    """The MERGE + AlphaGenome average = 0.5 x ensemble + 0.5 x the percentile of
    the task's AlphaGenome column against the benchmark variants, read from the
    label-free reference table. Splicing and non-coding only — see AG_AVERAGE_TASKS.

    Returns a dict that always carries `status`, so a caller can tell "no
    AlphaGenome value for this variant" apart from "the reference table is missing"
    apart from a real score. `merge_alphagenome_average` is None unless status == 'ok'; there
    is deliberately no fallback that quietly returns the ensemble score under
    the MERGE + AlphaGenome average name.
    """
    if task not in AG_AVERAGE_TASKS:
        return {'merge_alphagenome_average': None,
                'status': 'not_offered_for_this_variant_type',
                'detail': ("the MERGE + AlphaGenome average is only offered for splicing "
                           "and non-coding variants; on %s the best single AlphaGenome "
                           "feature is close to chance, so averaging it in would move the "
                           "score without adding evidence" % task),
                'alphagenome_feature': None, 'alphagenome_component': None}

    bundle = load_ag_bundles().get(task)
    if bundle is None:
        return {'merge_alphagenome_average': None, 'status': 'calibrator_unavailable',
                'detail': _ag_load_errors.get(task, 'no AlphaGenome reference table for this task'),
                'alphagenome_feature': None, 'alphagenome_component': None}

    feature = bundle['feature']
    value = feats.get(feature, float('nan'))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {'merge_alphagenome_average': None, 'status': 'alphagenome_missing',
                'detail': f"no value for {feature}; MERGE + AlphaGenome average needs an AlphaGenome score",
                'alphagenome_feature': feature, 'alphagenome_component': None}

    try:
        comp = float(np.interp(float(value), bundle['scores'], bundle['percentiles'],
                               left=bundle['percentiles'][0],
                               right=bundle['percentiles'][-1]))
    except Exception as exc:
        return {'merge_alphagenome_average': None, 'status': 'calibrator_failed',
                'detail': f"{type(exc).__name__}: {exc}",
                'alphagenome_feature': feature, 'alphagenome_component': None}

    comp = max(0.0, min(1.0, comp))
    w = MERGE_AG_AVERAGE_ENSEMBLE_WEIGHT
    final = max(0.0, min(1.0, w * ensemble_score + (1.0 - w) * comp))
    return {
        'merge_alphagenome_average': round(final, 4),
        'status': 'ok',
        'alphagenome_feature': feature,
        'alphagenome_feature_value': round(float(value), 6),
        'alphagenome_component': round(comp, 4),
        'alphagenome_component_kind': 'label-free percentile against the benchmark variants',
        'ensemble_weight': w,
        'interpretation': interpret_merge_alphagenome_average(final, task),
    }


def run_task(task: str, pred_data: dict, all_transcripts: list,
             skip_shap: bool = False, variant_info: str = ''):
    """Score one variant with the bundle for `task`. Returns ({key: result}, err)."""
    task = ENSEMBLE_TYPE_TO_TASK.get(task, task)
    bundles = load_bundles()
    model_key = TASK_MODEL_KEY.get(task, f'MERGE_{task}')

    if task not in bundles:
        err = _load_errors.get(task, 'unknown load error')
        return {model_key: {
            'error': f"MERGE model for '{task}' failed to load: {err}",
            'variant_type': task, 'model_used': model_key,
        }}, f"MERGE model for '{task}' failed to load"

    bundle = bundles[task]
    cols = bundle['features']
    feats = build_feature_values(pred_data, all_transcripts, is_splice=(task == 'splice'))
    row = {c: feats.get(c, float('nan')) for c in cols}
    X = pd.DataFrame([row], columns=cols)

    try:
        proba = bundle['pipeline'].predict_proba(X.values)[:, 1]
        score = max(0.0, min(1.0, float(proba[0])))
    except Exception as exc:
        import traceback
        return {model_key: {'error': f"prediction failed: {exc}\n{traceback.format_exc()}",
                            'variant_type': task, 'model_used': model_key}}, str(exc)

    missing = [c for c in cols if (row[c] is None or (isinstance(row[c], float) and np.isnan(row[c])))]
    shap_b64 = None if skip_shap else generate_shap_plot(bundle, X, cols, model_key,
                                                         variant_info=variant_info)

    features_raw_aligned = {
        c: (None if (isinstance(row[c], float) and np.isnan(row[c])) else round(float(row[c]), 4))
        for c in cols
    }
    try:
        imp_arr = bundle['pipeline'].steps[0][1].transform(X.values)
        imputed_features = {c: round(float(imp_arr[0][i]), 4) for i, c in enumerate(cols)}
    except Exception:
        imputed_features = dict(features_raw_aligned)

    mf = compute_merge_alphagenome_average(task, score, feats)

    result = {
        'score': round(score, 4),
        'interpretation': interpret_score(score),
        'merge_alphagenome_average': mf['merge_alphagenome_average'],
        'merge_alphagenome_average_interpretation': mf.get('interpretation'),
        'merge_alphagenome_average_status': mf['status'],
        'merge_alphagenome_average_detail': mf.get('detail'),
        'merge_alphagenome_average_caveat': mf.get('caveat'),
        'alphagenome_feature': mf.get('alphagenome_feature'),
        'alphagenome_feature_value': mf.get('alphagenome_feature_value'),
        'alphagenome_component': mf.get('alphagenome_component'),
        'ensemble_uses_alphagenome': False,
        'variant_type': task,
        'model_used': model_key,
        'algorithm': bundle.get('algorithm'),
        'feature_selection': bundle.get('feature_selection'),
        'n_features': len(cols),
        'features_used': features_raw_aligned,
        'features_raw_aligned': features_raw_aligned,
        'imputed_features': imputed_features,
        'missing_features': missing,
        'shap_plot': shap_b64,
    }
    return {model_key: result}, None


def run_ensemble(pred_data: dict, all_transcripts: list, skip_shap: bool = False):
    """Auto-detect the task (non-SNV first, then ANNOVAR) and score."""
    task = determine_variant_type(pred_data, all_transcripts)
    return run_task(task, pred_data, all_transcripts, skip_shap=skip_shap)


# Load eagerly at import (mirrors the website module's behaviour).
load_bundles()
for _t, _e in _load_errors.items():
    logger.error("[MERGE] bundle '%s' load error: %s", _t, str(_e).splitlines()[-1] if _e else _e)
load_ag_bundles()
for _t, _e in _ag_load_errors.items():
    logger.warning("[MERGE] AlphaGenome reference table for '%s' unavailable, the MERGE + AlphaGenome average is disabled "
                   "for this task: %s", _t, str(_e).splitlines()[-1] if _e else _e)
load_merge_alphagenome_average_thresholds()
