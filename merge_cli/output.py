"""
merge_cli/output.py
Terminal and file output: variant types in colour, SHAP plot files named after the variant.
"""
import json as _json, sys, base64, shutil, subprocess, time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()

_LABEL_STYLE = {
    "Pathogenic":         "bold red",
    "Likely Pathogenic":  "red",
    "Uncertain Significance": "yellow",
    "Uncertain":          "yellow",
    "Likely Benign":      "green",
    "Benign":             "bold green",
}

def _fmt(val, decimals=4) -> str:
    if val is None or val in ("N/A", ".", ""): return "-"
    try: return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError): return str(val)

def _score_text(score, label) -> Text:
    if score is None: return Text("-", style="dim")
    pct = f"{score * 100:.1f}%"
    style = _LABEL_STYLE.get(label or "", "white")
    t = Text(pct, style=style)
    if label: t.append(f"  {label}", style=style)
    return t

def _safe_float(v):
    try:
        return float(v) if v is not None and str(v).strip() not in (".", "N/A", "") else None
    except Exception:
        return None

def _extract_flat(prediction: dict, ensemble=None) -> dict:
    inp  = prediction.get("input", {})
    db   = prediction.get("dbnsfp") or {}

    ann_raw = db.get("annotations") or []
    ann = (ann_raw[0] if isinstance(ann_raw, list) and ann_raw
           else ann_raw if isinstance(ann_raw, dict) else {})

    dlm_raw = db.get("dl_models") or []
    dlm = (dlm_raw[0] if isinstance(dlm_raw, list) and dlm_raw
           else dlm_raw if isinstance(dlm_raw, dict) else {})

    ag   = prediction.get("alphagenome") or {}
    hy   = prediction.get("hyenadna") or {}
    nt_d = prediction.get("nt") or {}
    gpn  = prediction.get("gpn_msa") or {}
    evo2 = prediction.get("evo2") or {}
    evo2_7b      = prediction.get("evo2_7b") or {}
    evo2_7b_base = prediction.get("evo2_7b_base") or {}
    evo2_1b_base = prediction.get("evo2_1b_base") or {}
    enf  = prediction.get("enformer") or {}
    gen  = prediction.get("generator") or {}
    gen2 = prediction.get("generator_v2") or {}
    nt2  = prediction.get("nt_v2") or {}
    carb = prediction.get("carbon") or {}

    def _evo2_val(node):
        if not isinstance(node, dict):
            return None
        for k in ("llr_score", "score_mean", "score"):
            if node.get(k) is not None:
                return node[k]
        return None

    def _enf(key):
        """Enformer summary lookup; accepts the bare stat name too."""
        v = enf.get(key)
        if v is None:
            v = enf.get(key.replace("enformer_", ""))
        return _safe_float(v)

    ens_score = ens_label = ens_model = ens_variant_type = None
    ens_n_features = None
    ens_missing = []
    imputed_feats = {}
    _from_cache = False
    mf_score = mf_label = mf_status = mf_detail = mf_caveat = None
    ag_feature = ag_component = None

    if ensemble:
        ens_results = ensemble.get("ensemble_results") or {
            k: v for k, v in ensemble.items() if k not in ("success", "error")
        }
        _from_cache = bool(ensemble.get("from_cache"))
        for _key, _val in ens_results.items():
            if isinstance(_val, dict):
                ens_score        = _val.get("score")
                ens_label        = (_val.get("interpretation") or {}).get("label")
                ens_model        = _key
                ens_variant_type = _val.get("variant_type")
                imputed_feats    = _val.get("imputed_features", {})
                ens_n_features   = _val.get("n_features")
                ens_missing      = _val.get("missing_features") or []
                mf_score         = _val.get("merge_alphagenome_average")
                mf_label         = (_val.get("merge_alphagenome_average_interpretation") or {}).get("label")
                mf_status        = _val.get("merge_alphagenome_average_status")
                mf_detail        = _val.get("merge_alphagenome_average_detail")
                mf_caveat        = _val.get("merge_alphagenome_average_caveat")
                ag_feature       = _val.get("alphagenome_feature")
                ag_component     = _val.get("alphagenome_component")
                if _val.get("from_cache"):
                    _from_cache = True
                break

    def _get_val(raw_val, imputed_key):
        v = _safe_float(raw_val)
        return v if v is not None else imputed_feats.get(imputed_key)

    ag_stats = (ag.get("statistics") or {})

    # Remote API returns keys without "alphagenome_" prefix; local returns with prefix.
    # Normalize: if prefixed keys exist, use them; otherwise fall back to unprefixed.
    def _ag(key_with_prefix, key_without_prefix):
        v = ag_stats.get(key_with_prefix)
        if v is None:
            v = ag_stats.get(key_without_prefix)
        return _safe_float(v)

    return {
        "Chr":         inp.get("chrom", ann.get("chr", "-")),
        "Pos":         inp.get("pos",   ann.get("pos", "-")),
        "Ref":         inp.get("ref",   ann.get("ref", "-")),
        "Alt":         inp.get("alt",   ann.get("alt", "-")),
        "Gene":        ann.get("genename", "-"),
        "Transcript":  ann.get("Ensembl_transcriptid", "-"),
        "MERGE_Score":        ens_score,
        "MERGE_n_features":   ens_n_features,
        "MERGE_missing":      ens_missing,
        "MERGE_Label":        ens_label,
        "MERGE_Model":        ens_model,
        "MERGE_AlphaGenome_Average":        mf_score,
        "MERGE_AlphaGenome_Average_Label":  mf_label,
        "MERGE_AlphaGenome_Average_Status": mf_status,
        "MERGE_AlphaGenome_Average_Detail": mf_detail,
        "MERGE_AlphaGenome_Average_Caveat": mf_caveat,
        "AlphaGenome_Feature":   ag_feature,
        "AlphaGenome_Component": ag_component,
        "variant_type":       ens_variant_type or prediction.get("_variant_type"),
        "AlphaMissense": _get_val((dlm.get("AlphaMissense") or {}).get("score"), "AlphaMissense_score"),
        "ESM1b":         _get_val((dlm.get("ESM1b") or {}).get("score"), "ESM1b_score"),
        "GPN_MSA":       _get_val(gpn.get("score"), "GPN_MSA_score"),
        "HyenaDNA":      _get_val(hy.get("score"), "heyenadna_score"),
        "NT":            _get_val(nt_d.get("score"), "NT_score"),
        "Evo2":          _evo2_val(evo2) or _evo2_val(evo2_7b),
        "Evo2_7b_base":  _evo2_val(evo2_7b_base),
        "Evo2_1b_base":  _evo2_val(evo2_1b_base),
        # Enformer contributes 7 summary statistics, not one number; the
        # ensemble panels use max_abs and pos_sum. Keep the legacy single
        # "Enformer" key pointing at max_abs for compatibility.
        "Enformer":            _enf("enformer_max_abs") if _enf("enformer_max_abs") is not None
                               else _safe_float(enf.get("score") or enf.get("delta_score")
                                                or enf.get("enformer_score")),
        "Enformer_absmean":    _enf("enformer_absmean"),
        "Enformer_mean":       _enf("enformer_mean"),
        "Enformer_std":        _enf("enformer_std"),
        "Enformer_max_abs":    _enf("enformer_max_abs"),
        "Enformer_top5_absmean": _enf("enformer_top5_absmean"),
        "Enformer_pos_sum":    _enf("enformer_pos_sum"),
        "Enformer_neg_sum":    _enf("enformer_neg_sum"),
        "GENERATOR":     gen.get("score"),
        "GENERATOR_v2":  gen2.get("score"),
        "NT_v2":         nt2.get("score"),
        "Carbon":        carb.get("score"),
        "AG_raw_max":      _ag("alphagenome_raw_score_max",      "raw_score_max"),
        "AG_quantile_max": _ag("alphagenome_quantile_score_max", "quantile_score_max"),
        "AG_raw_min":      _ag("alphagenome_raw_score_min",      "raw_score_min"),
        "AG_quantile_min": _ag("alphagenome_quantile_score_min", "quantile_score_min"),
        "AG_raw_mean":     _ag("alphagenome_raw_score_mean",     "raw_score_mean"),
        "AG_quantile_mean":_ag("alphagenome_quantile_score_mean","quantile_score_mean"),
        "AG_Splicing":     _ag("alphagenome_splicing",            "alphagenome_splicing"),
        "from_cache":      _from_cache,
        "precomputed_hit": prediction.get("precomputed_hit", False),
        "precomputed_source": prediction.get("precomputed_source"),
        "genome_build": prediction.get("genome_build") or prediction.get("genome_version"),
    }


def _save_shap(shap_b64: str, model_name: str = "", variant_label: str = "") -> str | None:
    try:
        ts = int(time.time())
        safe_model = (model_name or "merge").replace(" ", "_")
        if variant_label:
            safe_variant = variant_label.replace(" ", "_").replace(":", "_").replace(">", "to")
            fname = f".merge-shap-{safe_variant}-{safe_model}-{ts}.png"
        else:
            fname = f".merge-shap-{safe_model}-{ts}.png"
        path = Path.home() / fname
        path.write_bytes(base64.b64decode(shap_b64))
        return str(path)
    except Exception as e:
        console.print(f"[yellow]Failed to save SHAP plot: {e}[/yellow]")
        return None


def _open_image(path: str) -> None:
    for cmd in (["xdg-open", path], ["open", path], ["eog", path], ["display", path]):
        if shutil.which(cmd[0]):
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def render_single(prediction: dict, ensemble=None, fmt="table", errors=None) -> None:
    flat = _extract_flat(prediction, ensemble)
    is_splice = (flat.get("variant_type") or "").lower() == "splice"

    if fmt == "json":
        _json.dump({"prediction": prediction, "ensemble": ensemble},
                   sys.stdout, indent=2, ensure_ascii=False)
        print(); return

    if fmt == "tsv":
        print("\t".join(flat.keys()))
        print("\t".join(_fmt(v) for v in flat.values()))
        return

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
              title="MERGE Prediction Result", title_style="bold")
    t.add_column("Field",  style="dim", no_wrap=True, min_width=20)
    t.add_column("Result", min_width=30)

    # Variant
    t.add_section()
    t.add_row("Variant",
              f"[bold]{flat['Chr']}:{flat['Pos']}  {flat['Ref']} → {flat['Alt']}[/bold]")
    t.add_row("Gene / Transcript",
              f"{_fmt(flat['Gene'])}  /  {_fmt(flat['Transcript'])}")
    vt = flat.get("variant_type")
    if vt:
        vt_style = {"coding": "bold magenta", "noncoding": "bold blue",
                    "splice": "bold yellow", "nonsnv": "bold cyan"}.get(str(vt).lower(), "white")
        t.add_row("Variant Type", Text(str(vt), style=vt_style))

    # Cache indicator
    _is_cached = (
        (ensemble or {}).get("from_cache") or
        any(
            isinstance(v, dict) and v.get("from_cache")
            for v in ((ensemble or {}).get("ensemble_results") or {}).values()
        )
    )
    if _is_cached:
        t.add_row("Score Source", Text("⚡ Pre-computed cache", style="bold yellow"))
    t.add_row("Precomputed Hit", str(bool(flat.get("precomputed_hit"))))
    t.add_row("Precomputed Source", _fmt(flat.get("precomputed_source")))
    t.add_row("Genome Build", _fmt(flat.get("genome_build")))

    # MERGE ensemble, plus the MERGE + AlphaGenome average where it is offered
    t.add_section()
    # "MERGE" is the ensemble, trained without any AlphaGenome feature; it is
    # reported for every variant type. "MERGE + AlphaGenome average" is the 1:1
    # average of that score with a single AlphaGenome feature's calibrated
    # probability, offered for splicing and non-coding only. It sits on its own
    # scale and carries its own cut-points, so the two labels can legitimately
    # differ. On coding / non-SNV the row is omitted entirely rather than shown
    # as "unavailable", because nothing about the variant could make it appear.
    t.add_row("MERGE Pathogenicity", _score_text(flat["MERGE_Score"], flat["MERGE_Label"]))
    _avg_status = flat.get("MERGE_AlphaGenome_Average_Status")
    if flat.get("MERGE_AlphaGenome_Average") is not None:
        t.add_row("MERGE + AlphaGenome average",
                  _score_text(flat["MERGE_AlphaGenome_Average"],
                              flat["MERGE_AlphaGenome_Average_Label"]))
        _agf, _agc = flat.get("AlphaGenome_Feature"), flat.get("AlphaGenome_Component")
        if _agf:
            sub = Text(f"0.5 x MERGE + 0.5 x AlphaGenome({_agf})", style="dim")
            if _agc is not None:
                sub.append(f" = 0.5 x {flat['MERGE_Score']:.4f} + 0.5 x {_agc:.4f}", style="dim")
            t.add_row("", sub)
        if flat.get("MERGE_AlphaGenome_Average_Caveat"):
            t.add_row("", Text(flat["MERGE_AlphaGenome_Average_Caveat"], style="yellow"))
    elif _avg_status == "not_offered_for_this_variant_type":
        t.add_row("MERGE + AlphaGenome average",
                  Text("- (splicing and non-coding variants only)", style="dim"))
    elif flat.get("MERGE_Score") is not None:
        why = (flat.get("MERGE_AlphaGenome_Average_Detail")
               or _avg_status or "AlphaGenome not run")
        t.add_row("MERGE + AlphaGenome average",
                  Text(f"- (unavailable: {why})", style="dim"))
    _n = flat.get("MERGE_n_features")
    _missing = flat.get("MERGE_missing") or []
    if _n:
        _used = _n - len(_missing)
        # Imputed features pull the score toward the training-set base rate, so
        # say plainly how much of the panel was actually measured.
        if not _missing:
            style = "green"
        elif len(_missing) * 2 <= _n:
            style = "yellow"
        else:
            style = "red"
        txt = Text(f"{_used}/{_n} computed", style=style)
        if _missing:
            txt.append(f"  ({len(_missing)} imputed: {', '.join(_missing)})", style="dim")
        t.add_row("Features", txt)

    # Individual model scores
    t.add_section()
    for label, key in [
        ("AlphaMissense", "AlphaMissense"),
        ("ESM1b",         "ESM1b"),
        ("GPN-MSA",       "GPN_MSA"),
        ("HyenaDNA",      "HyenaDNA"),
        ("NT",            "NT"),
        ("Evo2 (7B)",     "Evo2"),
        ("Evo2 (7B-base)", "Evo2_7b_base"),
        ("Evo2 (1B-base)", "Evo2_1b_base"),
        ("GENERATOR",     "GENERATOR"),
        ("GENERATOR-v2",  "GENERATOR_v2"),
        ("NT-v2",         "NT_v2"),
        ("Carbon",        "Carbon"),
    ]:
        # Evo2 variants and the opt-in DNA-LMs are frequently absent; only show
        # the rows that actually carry a score.
        if flat.get(key) is None and key in (
                "Evo2_7b_base", "Evo2_1b_base", "GENERATOR", "GENERATOR_v2",
                "NT_v2", "Carbon"):
            continue
        t.add_row(label, _fmt(flat[key]))

    # AlphaGenome
    t.add_section()
    # Enformer contributes 7 summary statistics. Only shown when it ran —
    # otherwise this section would be seven empty rows.
    _enf_rows = [
        ("Enformer absmean",      "Enformer_absmean"),
        ("Enformer mean",         "Enformer_mean"),
        ("Enformer std",          "Enformer_std"),
        ("Enformer max_abs",      "Enformer_max_abs"),
        ("Enformer top5_absmean", "Enformer_top5_absmean"),
        ("Enformer pos_sum",      "Enformer_pos_sum"),
        ("Enformer neg_sum",      "Enformer_neg_sum"),
    ]
    if any(flat.get(k) is not None for _, k in _enf_rows):
        for label, key in _enf_rows:
            t.add_row(label, _fmt(flat[key]))
        t.add_section()
    elif flat.get("Enformer") is not None:
        t.add_row("Enformer", _fmt(flat["Enformer"]))
        t.add_section()

    if is_splice:
        t.add_row("AlphaGenome Splicing", _fmt(flat["AG_Splicing"]))
    else:
        for label, key in [
            ("AlphaGenome Raw Max",      "AG_raw_max"),
            ("AlphaGenome Quantile Max", "AG_quantile_max"),
            ("AlphaGenome Raw Min",      "AG_raw_min"),
            ("AlphaGenome Quantile Min", "AG_quantile_min"),
            ("AlphaGenome Raw Mean",     "AG_raw_mean"),
            ("AlphaGenome Quantile Mean","AG_quantile_mean"),
        ]:
            t.add_row(label, _fmt(flat[key]))

    console.print(t)

    # SHAP plot
    variant_label = f"{flat['Chr']}_{flat['Pos']}_{flat['Ref']}_{flat['Alt']}"
    if ensemble:
        ens_results = ensemble.get("ensemble_results") or {
            k: v for k, v in ensemble.items() if k not in ("success", "error")
        }
        for _key, _val in ens_results.items():
            if isinstance(_val, dict) and _val.get("shap_plot"):
                shap_path = _save_shap(_val["shap_plot"], _key, variant_label=variant_label)
                if shap_path:
                    console.print(f"\n[bold]SHAP Plot[/bold] → [cyan]{shap_path}[/cyan]")
                    _open_image(shap_path)
                break
    merged_errors = dict(errors) if errors else {}
    merged_errors.update(prediction.get("errors", {}))
    for mod_key in ["dbnsfp", "alphagenome", "evo2", "evo2_7b", "evo2_7b_base", "evo2_1b_base",
                    "enformer", "generator", "generator_v2", "nt_v2", "carbon",
                    "nt", "hyenadna", "gpn_msa"]:
        mod_val = prediction.get(mod_key) or {}
        if isinstance(mod_val, dict) and "error" in mod_val:
            merged_errors[mod_key] = mod_val["error"]

    if merged_errors:
        console.print("\n[yellow]Some models did not return results:[/yellow]")
        for src, msg in merged_errors.items():
            clean_msg = str(msg).replace('\n', ' ')
            console.print(f"  [dim]- {src}:[/dim] {clean_msg}")


def render_batch_submitted(job_id, email) -> None:
    console.print(f"\n[bold green]✓ Batch job submitted[/bold green]")
    console.print(f"  Job ID : [bold]{job_id}[/bold]")
    console.print(f"  Email  : {email}")
    console.print(f"  Results will be sent to your email upon completion.")
    console.print(f"\nCheck progress: [bold]merge status {job_id}[/bold]")


def render_status(status_data: dict) -> None:
    if not status_data.get("found"):
        console.print("[red]Job not found or service has restarted.[/red]"); return
    state = status_data.get("status", "unknown")
    color = {"processing": "yellow", "complete": "green", "error": "red"}.get(state, "white")
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("key", style="dim"); t.add_column("value")
    t.add_row("Status",       f"[{color}]{state}[/{color}]")
    t.add_row("Submitted At", status_data.get("submitted_at", "-"))
    t.add_row("Variants OK",  str(status_data.get("n_ok", "-")))
    t.add_row("Warnings/Errors", str(status_data.get("n_err", "-")))
    t.add_row("Email Sent",   "Yes" if status_data.get("mail_sent") else "No")
    console.print(t)
    for line in (status_data.get("log") or [])[-20:]:
        console.print(f"  [dim]{line}[/dim]")
