"""
merge_cli/cli.py — the `merge` command, with the same functions as the merge.fanglab.cn website:
  1. Four variant types: coding / splice / noncoding / **nonsnv** (indel/MNV/delins).
     non-SNV is detected by allele length before ANNOVAR is consulted.
  2. The four MERGE_{coding,splicing,noncoding,nonsnv} deployment bundles
     (preprocessing embedded in the pipeline).
  3. Carbon-3B (--carbon) and Evo2 variant selection (--evo2-model /
     --evo2-models, incl. evo2_7b_base and evo2_1b_base).
  4. GENERATOR / GENERATOR-v2 / NT-v2 / Carbon each get their own conda env and
     local service, like Evo2 (ports 5005-5008). Environments may be shared
     between models.
  5. Every model runs by default (--no-<model> to skip); all three Evo2 variants
     are scored locally, since they are separate ensemble features.
  6. NT / HyenaDNA local services use the training-time scoring conventions
     (they differ between SNV and non-SNV); HyenaDNA verifies the REF allele.
  7. Conda environment lookup does not need `conda` on PATH.
"""
import os, sys, time, subprocess
import requests
import click
from rich.console import Console
from rich.table import Table
from rich import box

from . import api, output
from . import __version__
from .config import (
    get_mode, set_mode,
    set_local_config, set_local_config_bulk,
    get_local_config, show_config, FIXED_API_URL,
)

console = Console()

_MODEL_OPTIONS = [
    click.option("--no-alphagenome",   is_flag=True, default=False, help="Skip AlphaGenome"),
    click.option("--no-hyenadna",      is_flag=True, default=False, help="Skip HyenaDNA"),
    click.option("--no-nt",            is_flag=True, default=False, help="Skip Nucleotide Transformer"),
    click.option("--no-alphamissense", is_flag=True, default=False, help="Skip AlphaMissense"),
    click.option("--no-esm1b",         is_flag=True, default=False, help="Skip ESM1b"),
    click.option("--no-gpn-msa",       is_flag=True, default=False, help="Skip GPN-MSA"),
    click.option("--no-evo2",          is_flag=True, default=False, help="Skip Evo2"),
    click.option("--no-enformer",      is_flag=True, default=False, help="Skip Enformer"),
    click.option("--no-generator",     is_flag=True, default=False, help="Skip GENERATOR"),
    click.option("--no-generator-v2",  is_flag=True, default=False, help="Skip GENERATOR-v2"),
    click.option("--no-nt-v2",         is_flag=True, default=False, help="Skip NT-v2"),
    click.option("--no-carbon",        is_flag=True, default=False, help="Skip Carbon-3B"),
    click.option("--all-models",       "all_models", is_flag=True, default=False,
                 help="(kept for compatibility; every model is on by default)"),
    click.option("--evo2-model", "evo2_model", default="evo2_7b",
                 type=click.Choice(["evo2_7b", "evo2_7b_base"]),
                 help="Evo2 weights to use for the primary Evo2 score (remote and local)"),
    click.option("--evo2-models", "evo2_models", default=None,
                 help="Local mode only: comma-separated Evo2 variants to score. Defaults to "
                      "all three (evo2_7b,evo2_7b_base,evo2_1b_base); narrow it to trade "
                      "feature coverage for speed, e.g. --evo2-models evo2_7b. They are "
                      "scored one at a time (GPU-exclusive); evo2_1b_base needs FP8 (compute >= 8.9)."),
]

# All three Evo2 variants are scored by default: evo2_7b_score, evo2_7b_base_score
# and evo2_1b_base_score are separate features, and the non-SNV / noncoding /
# splice panels each use more than one. They share one GPU-resident service and
# are scored one at a time, which costs a few extra seconds per variant.
EVO2_DEFAULT_VARIANTS = ("evo2_7b", "evo2_7b_base", "evo2_1b_base")

# The DNA language models are part of the ensemble feature panels, so they run
# by default like every other model. Locally each needs its own service; when one
# is not running its feature is median-imputed and the CLI says so.
_DNA_LM_MODELS = ("use_generator", "use_generator_v2", "use_nt_v2", "use_carbon")


def _add_model_options(f):
    for opt in reversed(_MODEL_OPTIONS):
        f = opt(f)
    return f


def _model_flags(**kwargs) -> dict:
    all_models = kwargs.get("all_models", False)
    flags = {
        "use_alphagenome":   not kwargs.get("no_alphagenome",   False),
        "use_hyenadna":      not kwargs.get("no_hyenadna",      False),
        "use_nt":            not kwargs.get("no_nt",            False),
        "use_alphamissense": not kwargs.get("no_alphamissense", False),
        "use_esm1b":         not kwargs.get("no_esm1b",         False),
        "use_gpn_msa":       not kwargs.get("no_gpn_msa",       False),
        "use_evo2":          not kwargs.get("no_evo2",           False),
        "use_enformer":      not kwargs.get("no_enformer",       False),
        "use_generator":     not kwargs.get("no_generator",      False),
        "use_generator_v2":  not kwargs.get("no_generator_v2",   False),
        "use_nt_v2":         not kwargs.get("no_nt_v2",          False),
        "use_carbon":        not kwargs.get("no_carbon",         False),
    }
    if all_models:
        for name in _DNA_LM_MODELS:
            flags[name] = True

    flags["evo2_model"] = kwargs.get("evo2_model") or "evo2_7b"

    # Which Evo2 variants local mode should score. All three by default: the
    # non-SNV, noncoding and splice panels each use more than one of them, so
    # scoring only the primary variant would leave real features imputed.
    # --evo2-models narrows it when the extra passes are not worth the time.
    raw = kwargs.get("evo2_models")
    if raw:
        variants = [v.strip() for v in str(raw).split(",") if v.strip()]
    else:
        variants = list(EVO2_DEFAULT_VARIANTS)
    flags["evo2_models"] = variants
    return flags


def _normalize_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else "chr" + chrom


# ─────────────────────── Root ───────────────────────────────
@click.group()
@click.version_option(__version__, prog_name="merge")
def cli():
    """MERGE Variant Pathogenicity Prediction CLI

    \b
    NOT FOR CLINICAL USE — research prototype, NOT a medical device.
    Not for diagnosis, treatment decisions or genetic counselling. No warranty.

    \b
    MERGE (the ensemble, which uses no AlphaGenome feature) is returned for
    every variant. Splicing and non-coding variants also get the MERGE +
    AlphaGenome average (0.5 x MERGE + 0.5 x one AlphaGenome column's
    percentile against the benchmark variants -- a lookup table, not a fitted
    model); coding and non-SNV do not, because their best
    AlphaGenome feature is near chance. The two scores are on different scales
    and the average has its own cut-points, so read its own label rather than
    comparing the number to MERGE's bands.

    \b
    Code and the four MERGE_noAlphaGenome_*.pkl ensembles: MIT. The
    AlphaGenome percentile table behind the average
    (alphagenome_reference_quantiles.json): CC BY-NC 4.0; delete it and MERGE
    still runs. Four
    upstream components are non-commercial — AlphaGenome, Nucleotide
    Transformer v1/v2, dbNSFP, ANNOVAR. See LICENSE and THIRD_PARTY_NOTICES.md,
    both shipped with this package.

    \b
    AlphaGenome needs your own API key, in both modes:
      merge alphagenome configure --api-key YOUR_KEY

    \b
    Server: https://merge.fanglab.cn  (fixed, no configuration needed)

    \b
    Quick start (remote mode):
      merge predict --chrom chr17 --pos 43092919 --ref A --alt G

    \b
    Local mode (no rate limit, GPU required):
      merge doctor                                   # Check device
      merge local env setup --model evo2             # Deploy Evo2 environment
      merge local env start --model evo2             # Start inference service
      merge local predict --chrom chr17 --pos 43092919 --ref A --alt G
    """


# ─────────────────────── merge doctor ─────────────────────────
@cli.command()
def doctor():
    """Detect local GPU and environment, provide mode recommendations."""
    from . import local_engine
    from .local_engine import detect_gpu, check_local_files, check_service_health
    import shutil

    console.print("\n[bold]═══ MERGE Environment Diagnostics ═══[/bold]\n")

    # GPU
    console.print("[bold]GPU Detection[/bold]")
    gpu_info = detect_gpu()
    if gpu_info["has_gpu"]:
        for i, (name, vram) in enumerate(zip(gpu_info["gpus"], gpu_info["vram_gb"])):
            vram_str = f"{vram} GB" if vram else "Unknown VRAM"
            console.print(f"  [green]✓[/green] GPU {i}: {name} ({vram_str})")
        console.print(
            "\n  [bold green]Recommended: Your device has a GPU. Local mode is preferred.[/bold green]\n"
            "  Local mode has no rate limit and is faster.\n"
            "  Run [bold]merge local env setup --model evo2[/bold] to start setup."
        )
    else:
        console.print("  [yellow]○[/yellow] No NVIDIA GPU detected")
        console.print(
            "\n  [bold yellow]Recommended: No GPU detected. Remote mode is preferred.[/bold yellow]\n"
            "  Running DL models on CPU is extremely slow (single prediction may take 30+ min).\n"
            "  Remote mode uses the MERGE server (https://merge.fanglab.cn) for all computation.\n"
            "  Just run [bold]merge predict[/bold] directly, no local setup needed."
        )

    # Local services
    console.print("\n[bold]Local Inference Services[/bold]")
    for m in local_engine.ALL_SERVICES:
        h = check_service_health(m)
        if h["online"]:
            console.print(f"  [green]●[/green] {m:<12} online  {h.get('latency_ms')} ms")
        else:
            console.print(f"  [dim]○[/dim] {m:<12} offline ({h.get('error', '')})")

    # Bundled models
    console.print("\n[bold]Bundled Ensemble Models (included in pip package)[/bold]")
    files_info = check_local_files()
    for label, info in files_info.items():
        if not info.get("bundled"):
            continue
        if info["exists"]:
            console.print(f"  [green]✓[/green] {label}")
        else:
            console.print(f"  [red]✗[/red] {label}  [dim](not found, reinstall: pip install --force-reinstall merge-cli)[/dim]")

    # Which ensemble module actually gets executed. Worth showing plainly: a
    # stale ensemble_predict.py from an older install used to be picked up
    # silently, and the resulting failure named neither the file nor the cause.
    from . import local_engine as _le
    try:
        _ep = _le._load_ep_module()
        console.print(f"  [green]✓[/green] Ensemble module in use  [dim]{_ep.__file__}[/dim]")
    except Exception as _e:
        console.print(f"  [red]✗[/red] Ensemble module failed to load: {_e}")

    # Large data files
    console.print("\n[bold]Pre-computed Data Files (manual download required)[/bold]")
    for label, info in files_info.items():
        if info.get("bundled"):
            continue
        if info["exists"]:
            sz = f"{info['size_gb']} GB" if info["size_gb"] else ""
            console.print(f"  [green]✓[/green] {label:<20} {sz}")
        elif label.startswith("Pre-computed"):
            console.print(f"  [dim]○[/dim] {label:<20} [dim]not downloaded (optional; the cache is "
                          f"disabled in this release)[/dim]")
        elif "optional" in label:
            console.print(f"  [dim]○[/dim] {label:<20} [dim]not downloaded (optional, local mode)[/dim]")
        else:
            console.print(f"  [dim]○[/dim] {label:<20} [dim]not downloaded (required for local mode)[/dim]")

    console.print(f"\n  Server: [bold]{FIXED_API_URL}[/bold] (default for remote mode)")
    console.print(f"  Current mode: [bold]{get_mode()}[/bold]\n")


# ─────────────────────── merge config ─────────────────────────
@cli.group()
def config():
    """Manage runtime configuration (server URL is fixed)."""


@config.command("set-mode")
@click.argument("mode", type=click.Choice(["remote", "local"]))
def config_set_mode(mode):
    """Switch runtime mode (remote / local)."""
    set_mode(mode)
    console.print(f"[green]✓[/green] Mode switched to: [bold]{mode}[/bold]")
    if mode == "local":
        console.print("  Make sure you have completed: merge local env setup --model evo2")


@config.command("show")
def config_show():
    """Show current configuration."""
    info = show_config()
    console.print(f"\n  Server URL  : [bold]{info['api_url']}[/bold]  [dim](fixed, not configurable)[/dim]")
    console.print(f"  Mode        : [bold]{info['mode']}[/bold]")
    console.print(f"  Config file : [dim]{info['config_file']}[/dim]")
    if info["mode"] == "local":
        lc = info["local"]
        console.print("\n[bold]Local Configuration:[/bold]")
        console.print(f"  Data dir       : {lc.get('data_dir')}")
        console.print(f"  Precomputed dir: {lc.get('precomputed_data_dir')}")
        console.print(f"  Model dir      : {lc.get('model_dir')}  [dim](pkl bundled, for reference only)[/dim]")
        console.print(f"  Evo2 model dir : {lc.get('evo2_model_dir') or '(not set, using default)'}")
        console.print(f"  Evo2 model name: {lc.get('evo2_model_name', 'evo2_7b')}")
        console.print(f"  ANNOVAR dir    : {lc.get('annovar_dir') or '(not set)'}")
        console.print(f"  Reference FASTA: {lc.get('genome_ref') or '(not set — needed for non-SNVs)'}")
        console.print("\n[bold]Model Services:[/bold]")
        console.print(f"  AlphaGenome    : {lc.get('alphagenome_url')}  "
                      f"[dim]api key: {'set' if lc.get('alphagenome_api_key') else 'not set'}[/dim]")
        console.print(f"  HyenaDNA       : {lc.get('hyenadna_url')}")
        console.print(f"  NT             : {lc.get('nt_url')}")
        console.print(f"  Evo2           : {lc.get('evo2_url')}")
        console.print(f"  Enformer       : {lc.get('enformer_url')}")
        console.print(f"  Carbon         : {lc.get('carbon_url')}")
        console.print(f"  GENERATOR      : {lc.get('generator_url')}")
        console.print(f"  GENERATOR-v2   : {lc.get('generator_v2_url')}")
        console.print(f"  NT-v2          : {lc.get('nt_v2_url')}")
    console.print()


# ─────────────────────── merge precomputed ─────────────────────
@cli.group()
def precomputed():
    """Manage local precomputed VCF score cache."""


@precomputed.command("configure")
@click.option("--data-dir", required=True, type=click.Path(file_okay=False),
              help="Directory containing precomputed .vcf.gz and .tbi files")
def precomputed_configure(data_dir):
    """Configure the precomputed VCF directory."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    set_local_config("precomputed_data_dir", data_dir)
    console.print(f"[green]✓[/green] Precomputed data dir set to: [bold]{data_dir}[/bold]")


@precomputed.command("status")
def precomputed_status():
    """Check precomputed VCF and tabix index availability."""
    from . import local_engine
    data_dir = local_engine.get_precomputed_data_dir()
    console.print(f"\n[bold]Precomputed VCF Status[/bold]")
    if not local_engine.PRECOMPUTED_CACHE_ENABLED:
        console.print(
            "  [yellow]⚠ The cache is currently disabled.[/yellow] The shipped tables predate the\n"
            "    bundled ensemble models — they lack the newer model features and carry an\n"
            "    ensemble score from the previous pipeline, so every variant is scored with\n"
            "    the current models instead. Re-enable with MERGE_ENABLE_PRECOMPUTED=1 once\n"
            "    regenerated tables are in place.\n")
    console.print(f"  Directory: [bold]{data_dir}[/bold]\n")
    for key, info in local_engine.precomputed_file_status(data_dir).items():
        ok = info["vcf_exists"] and info["tbi_exists"]
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(
            f"  {mark} {info['genome']} {info['source']:<8} "
            f"VCF={'yes' if info['vcf_exists'] else 'no'} "
            f"TBI={'yes' if info['tbi_exists'] else 'no'}"
        )
    console.print()


@precomputed.command("download")
@click.option("--data-dir", default=None, type=click.Path(file_okay=False),
              help="Destination directory (default: configured precomputed dir)")
@click.option("--genome", default="all", type=click.Choice(["all", "hg38", "hg19"]))
@click.option("--variant-type", "variant_type", default="all",
              type=click.Choice(["all", "coding", "splicing"]))
@click.option("--force", is_flag=True, default=False,
              help="Download even while the cache is disabled")
def precomputed_download(data_dir, genome, variant_type, force):
    """Download precomputed coding/splicing VCF files and .tbi indices."""
    from . import local_engine
    if not local_engine.PRECOMPUTED_CACHE_ENABLED and not force:
        console.print(
            "[yellow]The local precomputed VCF cache is disabled in this release.[/yellow]\n"
            "  The VCF tables served for it predate the bundled models, so they would not be\n"
            "  used; every variant is scored with the current models instead.\n"
            "  Precomputed MERGE scores for the coding and splice-region candidate sets are\n"
            "  distributed with the MERGE-benchmark archive (see the README).\n"
            "  Use --force to download the VCF tables anyway.")
        return
    target_dir = os.path.abspath(os.path.expanduser(data_dir or local_engine.get_precomputed_data_dir()))
    os.makedirs(target_dir, exist_ok=True)
    set_local_config("precomputed_data_dir", target_dir)
    _download_precomputed_vcfs(target_dir, genome, variant_type)


# ─────────────────────── merge alphagenome ─────────────────────
@cli.group()
def alphagenome():
    """Configure AlphaGenome, which runs on Google DeepMind's hosted API.

    \b
    AlphaGenome is not a local model — it needs your own API key, in BOTH
    remote and local mode:
      https://deepmind.google.com/science/alphagenome
      merge alphagenome configure --api-key YOUR_KEY

    \b
    Without a key you still get AlphaGenome values for any variant already in
    the MERGE precomputed cache. Only a cache miss needs a live call, and that
    call is made under your key — never under a MERGE-owned one.

    \b
    DeepMind's terms, which you accept when you obtain a key:
      The AlphaGenome API is provided for non-commercial use only and is
      subject to the AlphaGenome Terms of Service. Outputs generated by
      AlphaGenome should not be used for the training of other machine
      learning models.
      Predictions are for theoretical modelling and research purposes only;
      they should not be used for clinical decision-making or relied upon for
      medical or other professional advice.
    """


@alphagenome.command("configure")
@click.option("--api-key", default=None, help="Your AlphaGenome API key")
@click.option("--url", default=None, help="Local AlphaGenome client service URL")
def alphagenome_configure(api_key, url):
    """Store your AlphaGenome API key (and optionally the service URL)."""
    if not api_key and not url:
        console.print("[yellow]Nothing to do: pass --api-key and/or --url.[/yellow]")
        return
    if api_key:
        set_local_config("alphagenome_api_key", api_key)
        masked = api_key[:4] + "…" + api_key[-4:] if len(api_key) > 10 else "…"
        console.print(f"[green]✓[/green] AlphaGenome API key saved ({masked})")
        console.print("  [dim]Stored in ~/.config/merge-cli/config.json. Used in both "
                      "modes: local mode bakes it into the service, remote mode sends "
                      "it with each request (the server uses it for that call and does "
                      "not store it).[/dim]")
        console.print("  [dim]In local mode, restart the service for it to take effect: "
                      "merge local env stop --model alphagenome && "
                      "merge local env start --model alphagenome[/dim]")
        console.print("  [yellow]Non-commercial use only[/yellow][dim] — the AlphaGenome "
                      "Terms of Service apply to every call made with this key, and "
                      "AlphaGenome outputs must not be used to train other ML models.[/dim]")
    if url:
        set_local_config("alphagenome_url", url.rstrip("/"))
        console.print(f"[green]✓[/green] AlphaGenome URL set to: [bold]{url.rstrip('/')}[/bold]")


@alphagenome.command("status")
def alphagenome_status():
    """Check the AlphaGenome client service and whether a key is configured."""
    from .local_engine import check_service_health
    from .config import get_alphagenome_api_key
    has_key = bool(get_alphagenome_api_key())
    console.print(f"  API key   : {'[green]configured[/green]' if has_key else '[yellow]not set[/yellow]'}")
    if has_key:
        console.print("    [dim]Used in both modes. Live AlphaGenome calls are made "
                      "under this key, never under a MERGE-owned one.[/dim]")
    else:
        console.print("    [dim]Get one at https://deepmind.google.com/science/alphagenome, "
                      "then: merge alphagenome configure --api-key YOUR_KEY[/dim]")
        console.print("    [dim]Without a key: cached variants still return AlphaGenome "
                      "values; a cache miss returns them as unavailable.[/dim]")
    h = check_service_health("alphagenome")
    if h.get("online"):
        console.print(f"  [green]✓[/green] Service online at {h.get('url')} ({h.get('latency_ms')} ms)")
    else:
        console.print(f"  [yellow]○[/yellow] Service offline at {h.get('url')}: {h.get('error')}")


# ─────────────────────── merge enformer ────────────────────────
@cli.group()
def enformer():
    """Configure and inspect Enformer local service integration."""


@enformer.command("configure")
@click.option("--url", default=None,
              help="Enformer service URL exposing /health and /predict")
@click.option("--model-path", default=None,
              help="Enformer SavedModel directory or TF-Hub URL "
                   "(e.g. https://tfhub.dev/deepmind/enformer/1)")
@click.option("--transform-path", default=None,
              help="Path to enformer.transform.pkl — the normalisation transform the "
                   "MERGE features were computed with")
def enformer_configure(url, model_path, transform_path):
    """Configure the Enformer service URL, weights and transform."""
    if not any([url, model_path, transform_path]):
        console.print("[yellow]Nothing to do: pass --url, --model-path and/or --transform-path.[/yellow]")
        return
    if url:
        set_local_config("enformer_url", url.rstrip("/"))
        console.print(f"[green]✓[/green] Enformer URL set to: [bold]{url.rstrip('/')}[/bold]")
    if model_path:
        set_local_config("enformer_model_path", model_path)
        console.print(f"[green]✓[/green] Enformer model path: [bold]{model_path}[/bold]")
    if transform_path:
        set_local_config("enformer_transform_path", transform_path)
        console.print(f"[green]✓[/green] Enformer transform: [bold]{transform_path}[/bold]")
        if not os.path.exists(transform_path):
            console.print("  [yellow]⚠ That path does not exist yet.[/yellow]")


@enformer.command("status")
def enformer_status():
    """Check the configured Enformer service health endpoint."""
    from .local_engine import check_service_health
    h = check_service_health("enformer")
    if h.get("online"):
        console.print(f"[green]✓[/green] Enformer online at {h.get('url')} ({h.get('latency_ms')} ms)")
    else:
        console.print(f"[yellow]○[/yellow] Enformer offline at {h.get('url')}: {h.get('error')}")


# ─────────────────────── merge mcp ─────────────────────────────
@cli.group()
def mcp():
    """Register MERGE as an MCP server, so you can ask for predictions in chat.

    \b
    Needs the optional MCP dependency:
      pip install "merge-cli[mcp]"

    \b
    Then, for Claude Desktop or Claude Code:
      merge mcp install
      merge mcp status
    """


def _print_entry(entry: dict) -> None:
    cmd = " ".join([entry["command"], *entry.get("args", [])])
    console.print(f"  [dim]{cmd}[/dim]")


@mcp.command("install")
@click.option("--client", type=click.Choice(["auto", "claude-desktop", "claude-code", "codex"]),
              default="auto", help="Which MCP client to configure (default: auto-detect)")
@click.option("--scope", type=click.Choice(["user", "project"]), default="user",
              help="claude-code/codex only: this machine (user) or this directory (project)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be written, change nothing")
@click.option("--no-backup", is_flag=True, default=False,
              help="Do not keep a timestamped copy of the previous config")
def mcp_install(client, scope, dry_run, no_backup):
    """Add the MERGE server to your MCP client's config file.

    Finds the merge-mcp executable belonging to this install — the step that
    most often goes wrong when the config is written by hand — and merges it
    into the client config, leaving any other MCP servers you have untouched.
    """
    from . import mcp_install as mi

    if not mi.mcp_package_available():
        console.print("[yellow]The 'mcp' package is not installed, so the server "
                      "cannot start yet.[/yellow]")
        console.print('  Install it with: [bold]pip install "merge-cli[mcp]"[/bold]')
        console.print("  [dim]Continuing anyway — the config will be correct once "
                      "you install it.[/dim]\n")

    targets = [client]
    if client == "auto":
        targets = [c for c in mi.CLIENTS if mi.client_installed(c)]
        if not targets:
            console.print("[red]No supported MCP client detected.[/red]")
            console.print("  Pass --client explicitly if yours is installed somewhere "
                          "unusual.")
            sys.exit(1)
        console.print(f"[dim]Detected: {', '.join(targets)}[/dim]\n")

    wrote_any = False
    for target in targets:
        try:
            plan = mi.plan_install(target, scope)
        except mi.ConfigError as e:
            console.print(f"[red]✗ {target}: {e}[/red]")
            sys.exit(1)

        label = f"{target}" + (f" ({scope})" if target in ("claude-code", "codex") else "")
        console.print(f"[bold]{label}[/bold]")
        console.print(f"  config: {plan['path']}"
                      + ("" if plan["exists"] else "  [dim](will be created)[/dim]"))

        if plan["already_current"]:
            console.print("  [green]✓[/green] already registered and up to date")
            _print_entry(plan["entry"])
            console.print()
            continue

        # Codex only. A table under the hyphenated name starts and answers
        # tools/list but shows "Tools: (none)", so it has to go even when the
        # real entry is already correct — writing is what removes it.
        if plan.get("stale_entry"):
            console.print(f"  [yellow]removing a stale '{mi.SERVER_NAME}' entry[/yellow] "
                          "[dim](Codex cannot call a hyphenated server)[/dim]")

        if plan["existing"]:
            console.print("  [yellow]existing entry will be replaced:[/yellow]")
            _print_entry(plan["existing"])
        if plan["other_servers"]:
            console.print(f"  [dim]leaving alone: {', '.join(plan['other_servers'])}[/dim]")

        console.print("  will register:")
        _print_entry(plan["entry"])

        if dry_run:
            console.print("  [dim](--dry-run: nothing written)[/dim]\n")
            continue

        try:
            made_backup = mi.apply_install(plan, backup=not no_backup)
        except (mi.ConfigError, OSError) as e:
            console.print(f"  [red]✗ failed to write: {e}[/red]")
            sys.exit(1)

        wrote_any = True
        console.print("  [green]✓[/green] written")
        if made_backup:
            console.print(f"  [dim]previous config saved to {made_backup}[/dim]")
        console.print()

    from . import skill_install as si

    if not any(si.is_current(c) for c in si.CLIENTS):
        console.print("[bold]Next:[/bold] [bold]merge skill install[/bold]")
        console.print("  [dim]Registering the tools is not the same as the model "
                      "noticing them — clients that defer tool loading (Codex "
                      "does) will not show them until a skill points the way.[/dim]\n")

    if wrote_any:
        console.print("[bold]Restart your MCP client for this to take effect.[/bold]")
        console.print('  Then try asking: [dim]"Is chr1:1040819 G>GC pathogenic?"[/dim]')


@mcp.command("status")
def mcp_status():
    """Show where the MERGE server is registered and whether it still resolves."""
    from . import mcp_install as mi

    rep = mi.status()
    console.print("[bold]Server command[/bold]")
    _print_entry(rep["resolved_command"])
    ok = rep["mcp_package"]
    console.print(f"  mcp package: {'[green]installed[/green]' if ok else '[yellow]missing[/yellow]'}")
    if not ok:
        console.print('    [dim]pip install "merge-cli[mcp]"[/dim]')

    console.print("\n[bold]Client configs[/bold]")
    for info in rep["clients"]:
        label = info["client"] + (f" ({info['scope']})" if info["client"] in ("claude-code", "codex") else "")
        if info.get("error"):
            console.print(f"  [red]✗[/red] {label}: {info['error']}")
            continue
        if not info["registered"]:
            hint = "" if info["installed"] else " [dim](client not detected)[/dim]"
            console.print(f"  [yellow]○[/yellow] {label}: not registered{hint}")
            console.print(f"    [dim]{info['path']}[/dim]")
            continue
        if not info.get("command_exists"):
            console.print(f"  [red]✗[/red] {label}: registered, but its command no longer exists")
            _print_entry(info["entry"])
            console.print("    [dim]Re-run `merge mcp install` to point it at this "
                          "install.[/dim]")
        elif not info.get("current"):
            console.print(f"  [yellow]![/yellow] {label}: registered, but points elsewhere")
            _print_entry(info["entry"])
            console.print("    [dim]Harmless if that install still works; "
                          "`merge mcp install` repoints it here.[/dim]")
        else:
            console.print(f"  [green]✓[/green] {label}: registered and current")
        if info.get("stale_entry"):
            console.print(f"    [yellow]![/yellow] a stale '{mi.SERVER_NAME}' entry is also "
                          "present; Codex cannot call a hyphenated server")
            console.print("    [dim]`merge mcp install` removes it.[/dim]")
        console.print(f"    [dim]{info['path']}[/dim]")


# ─────────────────────── merge skill ───────────────────────────
@cli.group()
def skill():
    """Install the MERGE skill, so an agent finds these tools on its own.

    \b
    Clients that defer MCP tool loading (Codex does) may never show the model
    that merge-cli is there. Skill descriptions stay visible, so the skill is
    what points it at the tools. Codex needs no plugin for this — it reads
    ~/.agents/skills directly.

    \b
      merge skill install
      merge skill status
    """


@skill.command("install")
@click.option("--client", type=click.Choice(["auto", "claude", "codex"]), default="auto",
              show_default=True, help="Which agent to install for (auto: all detected)")
@click.option("--scope", type=click.Choice(["user", "project"]), default="user",
              help="This machine (user) or this directory (project)")
def skill_install_cmd(client, scope):
    """Copy the bundled skill into an agent's skills directory."""
    from . import skill_install as si

    targets = [client]
    if client == "auto":
        targets = [c for c in si.CLIENTS if si.client_installed(c)]
        console.print(f"[dim]Installing for: {', '.join(targets)}[/dim]")

    for target in targets:
        try:
            res = si.install(target, scope)
        except (FileNotFoundError, OSError) as e:
            console.print(f"[red]✗ {target}: {e}[/red]")
            sys.exit(1)
        verb = "replaced" if res["replaced"] else "installed"
        console.print(f"[green]✓[/green] {target}: skill {verb} at [bold]{res['path']}[/bold]")

        # The skill points the model at tools. Without the server registered
        # there are none, the model finds nothing and answers from memory
        # instead — silently, and looking exactly like it ignored the skill.
        if not si.mcp_registered(target):
            console.print(f"  [yellow]![/yellow] the MERGE server is not registered "
                          f"for {target}, so the skill has no tools to point at")
            console.print("    [dim]Run `merge mcp install` — the skill only tells "
                          "the model when to use them.[/dim]")

    console.print('  [dim]Restart your client, then ask "Is chr1:1040819 G>GC '
                  'pathogenic?"[/dim]')


@skill.command("status")
def skill_status():
    """Show where the skill is installed and whether it matches this version."""
    from . import skill_install as si

    rep = si.status()
    console.print(f"[bold]Skill[/bold] {rep['skill']}")
    console.print(f"  [dim]bundled: {rep['bundled']}[/dim]")
    for t in rep["targets"]:
        label = f"{t['client']} ({t['scope']})"
        if not t["installed"]:
            hint = "" if t["detected"] else " [dim](client not detected)[/dim]"
            console.print(f"  [yellow]○[/yellow] {label}: not installed{hint}")
        elif t["current"]:
            console.print(f"  [green]✓[/green] {label}: installed and current")
        else:
            console.print(f"  [yellow]![/yellow] {label}: installed, but differs from "
                          "this version [dim](`merge skill install` refreshes it)[/dim]")
        console.print(f"    [dim]{t['path']}[/dim]")


@skill.command("uninstall")
@click.option("--client", type=click.Choice(["claude", "codex"]), required=True)
@click.option("--scope", type=click.Choice(["user", "project"]), default="user")
def skill_uninstall(client, scope):
    """Remove the skill directory, leaving any other skills alone."""
    from . import skill_install as si

    res = si.uninstall(client, scope)
    if res["removed"]:
        console.print(f"[green]✓[/green] removed {res['path']}")
    else:
        console.print(f"[yellow]○[/yellow] not installed: {res['path']}")


# ─────────────────────── merge plugin ──────────────────────────
@cli.group()
def plugin():
    """Generate the Codex plugin package (manifest + skill + MCP config).

    \b
    Codex does not read the Claude skills directory, so `merge skill install`
    does nothing there. A plugin is how the same skill reaches Codex.

    \b
      merge plugin build --out ./plugins
    """


@plugin.command("build")
@click.option("--out", "out_dir", default="./plugins", show_default=True,
              type=click.Path(file_okay=False),
              help="Directory to write the <name>/ plugin tree into")
@click.option("--marketplace/--no-marketplace", default=True, show_default=True,
              help="Also register it in <out>/../.agents/plugins/marketplace.json")
# Spelled out rather than imported, to keep plugin_build out of the import path
# of every `merge` invocation. tests/test_codex_plugin.py asserts these stay in
# step with plugin_build's enums.
@click.option("--installation",
              type=click.Choice(["NOT_AVAILABLE", "AVAILABLE", "INSTALLED_BY_DEFAULT"]),
              default="AVAILABLE", show_default=True,
              help="Marketplace installation policy")
@click.option("--authentication", type=click.Choice(["ON_INSTALL", "ON_USE"]),
              default="ON_INSTALL", show_default=True,
              help="Marketplace authentication policy")
def plugin_build_cmd(out_dir, marketplace, installation, authentication):
    """Write the plugin tree, regenerating it from this install."""
    from pathlib import Path

    from . import plugin_build as pb

    try:
        res = pb.build(out_dir, marketplace=False)
        if marketplace:
            res["marketplace"] = pb.update_marketplace(
                Path(out_dir).parent, res["plugin"],
                installation=installation, authentication=authentication)
    except (pb.ConfigError, OSError, ValueError) as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    console.print(f"[green]✓[/green] plugin written: [bold]{res['plugin']}[/bold]")
    for line in ("  .codex-plugin/plugin.json", "  .mcp.json",
                 "  skills/", "  README.md"):
        console.print(f"    [dim]{line}[/dim]")

    mk = res.get("marketplace")
    if not mk:
        console.print("  [dim]No marketplace written; Codex discovers local "
                      "plugins through one.[/dim]")
        return
    console.print(f"[green]✓[/green] marketplace: [bold]{mk['path']}[/bold]")
    console.print(f"    [dim]source: {mk['entry']['source']['path']}  "
                  f"policy: {mk['entry']['policy']['installation']} / "
                  f"{mk['entry']['policy']['authentication']}[/dim]")
    if mk["other_plugins"]:
        console.print(f"    [dim]left alone: {', '.join(mk['other_plugins'])}[/dim]")
    if mk["backup"]:
        console.print(f"    [dim]previous version saved to {mk['backup']}[/dim]")


@mcp.command("uninstall")
@click.option("--client", type=click.Choice(["claude-desktop", "claude-code", "codex"]), required=True)
@click.option("--scope", type=click.Choice(["user", "project"]), default="user")
def mcp_uninstall(client, scope):
    """Remove the MERGE entry from a client config, leaving other servers alone."""
    from . import mcp_install as mi
    try:
        res = mi.uninstall(client, scope)
    except mi.ConfigError as e:
        console.print(f"[red]✗ {e}[/red]"); sys.exit(1)
    if res["removed"]:
        console.print(f"[green]✓[/green] removed from {res['path']}")
        if res.get("backup"):
            console.print(f"  [dim]previous config saved to {res['backup']}[/dim]")
        console.print("[bold]Restart your MCP client for this to take effect.[/bold]")
    else:
        console.print(f"[yellow]○[/yellow] not registered in {res['path']} — nothing to do")


# ─────────────────────── merge predict ────────────────────────
@cli.command()
@click.option("--chrom",   required=True, help="Chromosome, e.g. chr17 or 17")
@click.option("--pos",     required=True, type=int, help="Variant position (1-based)")
@click.option("--ref",     required=True, help="Reference allele")
@click.option("--alt",     required=True, help="Alternate allele")
@click.option("--genome",  default="hg38", type=click.Choice(["hg38", "hg19"]))
@click.option("--format",  "fmt", default="table",
              type=click.Choice(["table", "json", "tsv"]))
@click.option("--no-ensemble", is_flag=True, default=False, help="Skip MERGE ensemble scoring")
@click.option("--local", "use_local", is_flag=True, default=False, help="Force local mode")
@click.option("--ensemble-type",
              type=click.Choice(["coding", "noncoding", "splice", "nonsnv"]), default=None,
              help="Manually specify variant type (bypass ANNOVAR)")
@click.option("--no-shap", is_flag=True, default=False,
              help="Skip SHAP analysis plot (faster prediction)")
@click.option("--annovar-dir", default=None,
              help="Local ANNOVAR directory (overrides config; default: use remote ANNOVAR)")
@click.option("--genome-ref", default=None,
              help="Reference genome FASTA path for local ANNOVAR (e.g. /path/to/hg19.fa)")
@click.option("--no-precomputed", is_flag=True, default=False,
              help="Disable local precomputed VCF lookup (currently disabled by default "
                   "while the tables are regenerated)")
@click.option("--evo2-timeout", default=180.0, type=float, show_default=True,
              help="Remote mode: seconds to wait for an Evo2 job the server queued "
                   "instead of scoring inline (0 = do not wait; the Evo2 feature is "
                   "then median-imputed)")
@_add_model_options
def predict(chrom, pos, ref, alt, genome, fmt, no_ensemble,
            use_local, ensemble_type, no_shap, annovar_dir, genome_ref,
            no_precomputed, evo2_timeout, **kwargs):
    """Single variant pathogenicity prediction (remote / local mode).

    \b
    Examples:
      merge predict --chrom chr17 --pos 43092919 --ref A --alt G
      merge predict --chrom 17 --pos 43092919 --ref A --alt G --genome hg19
      merge predict --chrom chr17 --pos 43092919 --ref A --alt G --annovar-dir /path/to/annovar
    """
    flags    = _model_flags(**kwargs)
    chrom    = _normalize_chrom(chrom)
    is_local = use_local or (get_mode() == "local")

    if annovar_dir:
        from .config import set_local_config
        set_local_config("annovar_dir", annovar_dir)
    if genome_ref:
        from .config import set_local_config
        set_local_config("genome_ref", genome_ref)

    from . import local_engine

    resp = None
    if not no_precomputed:
        cached = local_engine.query_precomputed_local(chrom, pos, ref, alt, genome)
        # Only usable when the record covers every model asked for; otherwise
        # score for real rather than return a partial feature set.
        if cached and not local_engine._cache_missing_requested_models(cached, flags):
            resp = local_engine.precomputed_prediction_response(chrom, pos, ref, alt, genome, cached)

    if resp is None and is_local:
        with console.status("[bold cyan]Running local prediction...[/bold cyan]"):
            try:
                resp = local_engine.predict_local(chrom, pos, ref, alt, genome, ensemble_type=ensemble_type, **flags)
            except FileNotFoundError as e:
                console.print(f"[red]File missing: {e}[/red]"); sys.exit(1)
            except Exception as e:
                console.print(f"[red]Local prediction failed: {e}[/red]"); sys.exit(1)
    elif resp is None:
        with console.status(f"[bold cyan]Predicting ({FIXED_API_URL})...[/bold cyan]"):
            try:
                resp = api.predict_single(chrom, pos, ref, alt, genome, **flags)
            except Exception as e:
                console.print(f"[red]Prediction failed: {e}[/red]"); sys.exit(1)
        # The server may queue Evo2 rather than score it inline; wait for the job
        # so the feature is real rather than median-imputed.
        if resp.get("success") and isinstance(resp.get("prediction"), dict):
            from .evo2_poll import resolve_evo2
            evo2 = resp["prediction"].get("evo2")
            if isinstance(evo2, dict) and evo2.get("job_id"):
                with console.status("[bold cyan]Waiting for the queued Evo2 job...[/bold cyan]"):
                    note = resolve_evo2(resp["prediction"], evo2_timeout)
                if note:
                    click.echo(f"Note: {note}", err=True)

    if not resp.get("success"):
        console.print(f"[red]Server error: {resp.get('error')}[/red]"); sys.exit(1)

    prediction = resp["prediction"]
    prediction.setdefault("genome_build", genome)
    prediction.setdefault("precomputed_hit", bool(prediction.get("precomputed")))
    prediction.setdefault("precomputed_source", None)
    ensemble   = None

    if flags.get("use_enformer", True) and not prediction.get("precomputed_hit"):
        existing_enf = prediction.get("enformer")
        # Only call local enformer as fallback if server didn't provide valid data
        if not isinstance(existing_enf, dict) or (
            existing_enf.get("error") or
            (existing_enf.get("score") is None and existing_enf.get("delta_score") is None
             and existing_enf.get("enformer_score") is None and existing_enf.get("enformer_absmean") is None)
        ):
            enf = local_engine.call_local_service("enformer", chrom, pos, ref, alt, genome)
            prediction["enformer"] = enf
            if isinstance(enf, dict) and enf.get("error"):
                prediction.setdefault("errors", {})["enformer"] = enf["error"]

    # A cache hit supplies features, not a verdict. Both the local VCF cache and
    # the server's own cache carry an ensemble score from the previous pipeline
    # (it comes back labelled "ClinVar" rather than "MERGE_Coding"), and on real
    # variants the two disagree outright — chr17:43092919 G>A reads 0.0% cached
    # and 75.7% when scored with the current models. Drop the stale verdict and
    # keep the features.
    if prediction.pop("precomputed", None) is not None and fmt != "json":
        console.print("[dim]⚡ Feature scores came from a precomputed cache; the MERGE "
                      "score is computed with the bundled models[/dim]")

    if not no_ensemble:
        skip_shap = no_shap
        if is_local and not no_shap and fmt == "table" and sys.stdin.isatty():
            skip_shap = not click.confirm("\nGenerate SHAP analysis plot?", default=True)
        variant_info = f"{chrom}:{pos} {ref}>{alt}"
        effective_ensemble_type = ensemble_type
        if not effective_ensemble_type:
            effective_ensemble_type = prediction.get("_variant_type")
        if not effective_ensemble_type:
            with console.status("[bold cyan]Detecting variant type...[/bold cyan]"):
                try:
                    effective_ensemble_type = local_engine.get_variant_type(chrom, pos, ref, alt, genome)
                except Exception:
                    effective_ensemble_type = None
        if effective_ensemble_type:
            prediction["_variant_type"] = effective_ensemble_type

        with console.status("[bold cyan]Computing MERGE ensemble score...[/bold cyan]"):
            try:
                # Score with the bundled models in both modes. Remote mode still
                # fetches the per-model features from the server, but the final
                # MERGE score is computed here: the server's /ensemble/ endpoint
                # currently answers from the previous ensemble pipeline (its
                # results come back keyed "ClinVar" rather than "MERGE_Coding"),
                # so trusting it would give a different verdict than local mode
                # for the same variant.
                dbnsfp      = prediction.get("dbnsfp") or {}
                transcripts = dbnsfp.get("transcripts") or ([dbnsfp] if dbnsfp else [])
                ens_result, ens_err = local_engine.run_local_ensemble(
                    prediction, transcripts, ensemble_type=effective_ensemble_type,
                    skip_shap=skip_shap, variant_info=variant_info)
                if ens_err:
                    console.print(f"[yellow]Ensemble model warning: {ens_err}[/yellow]")
                ensemble = {"success": True, **ens_result}
            except Exception as e:
                console.print(f"[yellow]Ensemble score failed (raw scores unaffected): {e}[/yellow]")

    output.render_single(prediction, ensemble, fmt=fmt,
                         errors=prediction.get("errors"))


# ─────────────────────── merge batch ──────────────────────────
@cli.command()
@click.argument("vcf_file", type=click.Path(exists=True))
@click.option("--email",  required=True, help="Email for result notification")
@click.option("--genome", default="hg38", type=click.Choice(["hg38", "hg19"]))
@_add_model_options
def batch(vcf_file, email, genome, **kwargs):
    """Batch VCF file prediction (async, results sent via email).

    \b
    Example:
      merge batch variants.vcf --email you@example.com
    """
    flags = _model_flags(**kwargs)
    with console.status("[bold cyan]Uploading VCF file...[/bold cyan]"):
        try:
            resp = api.submit_batch(vcf_file, email, genome, **flags)
        except Exception as e:
            console.print(f"[red]Submission failed: {e}[/red]"); sys.exit(1)
    if not resp.get("success"):
        console.print(f"[red]Submission failed: {resp.get('error')}[/red]"); sys.exit(1)
    output.render_batch_submitted(resp["job_id"], email)


# ─────────────────────── merge status ─────────────────────────
@cli.command()
@click.argument("job_id")
@click.option("--watch", is_flag=True, default=False, help="Poll every 30 seconds")
def status(job_id, watch):
    """Check batch job status."""
    def _poll():
        try:
            data = api.get_batch_status(job_id)
        except Exception as e:
            console.print(f"[red]Query failed: {e}[/red]"); return None
        output.render_status(data)
        return data

    data = _poll()
    if not watch or not data:
        return
    while data and data.get("status") == "processing":
        console.print("\n[dim]Polling again in 30s... Ctrl+C to exit[/dim]")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            break
        data = _poll()


# ─────────────────────── merge local ──────────────────────────
@cli.group()
def local():
    """Local mode management: all computation runs on your machine (GPU recommended).

    \b
    Quick start:
      merge doctor                                   # Step 0: Check device
      merge local env setup --model evo2             # Step 1: Deploy Evo2 environment
      merge local env start --model evo2             # Step 2: Start service
      merge local download --file dbnsfp             # Step 3: View dbNSFP download guide
      merge local predict --chrom chr17 --pos 43092919 --ref A --alt G
    """


# ── merge local setup ─────────────────────────────────────────
@local.command("setup")
def local_setup():
    """Interactive wizard: configure paths and ports for local mode."""
    from pathlib import Path
    from .local_engine import detect_gpu

    console.print("\n[bold]MERGE Local Mode Setup Wizard[/bold]")
    console.print(f"  Server: [dim]{FIXED_API_URL}[/dim] (fixed)\n")

    gpu_info = detect_gpu()
    if gpu_info["has_gpu"]:
        gpu_names = ", ".join(gpu_info["gpus"])
        console.print(f"[green]✓ GPU detected: {gpu_names}[/green]")
        console.print("  [bold green]Local mode is the best choice![/bold green]\n")
    else:
        console.print("[yellow]⚠ No GPU detected[/yellow]")
        console.print(
            "  [bold yellow]Warning: Running DL models on CPU is extremely slow.[/bold yellow]\n"
            "  Consider using remote mode (just run merge predict).\n"
        )
        if not click.confirm("Continue with local mode setup?", default=False):
            console.print("[dim]Cancelled. Use merge predict for remote mode.[/dim]")
            return

    console.print("[dim]Press Enter to accept default values[/dim]\n")
    data_dir     = click.prompt("Pre-computed files directory (dbNSFP / GPN-MSA)",
                                default=str(Path.home() / ".merge-data"))
    annovar_dir  = click.prompt("ANNOVAR directory (leave empty to use remote ANNOVAR)", default="")
    genome_ref   = click.prompt("Reference genome FASTA path (leave empty if not available)", default="")
    hyenadna_url = click.prompt("HyenaDNA service URL", default="http://localhost:5001")
    nt_url       = click.prompt("NT service URL",       default="http://localhost:5002")
    evo2_url     = click.prompt("Evo2 service URL",     default="http://localhost:5003")
    evo2_model_dir = click.prompt(
        "Evo2 model weights directory",
        default=str(Path.home() / ".merge-data" / "models" / "evo2_7b"),
    )

    os.makedirs(data_dir, exist_ok=True)

    set_local_config_bulk({
        "data_dir":       data_dir,
        "annovar_dir":    annovar_dir,
        "genome_ref":     genome_ref,
        "hyenadna_url":   hyenadna_url,
        "nt_url":         nt_url,
        "evo2_url":       evo2_url,
        "evo2_model_dir": evo2_model_dir,
        "evo2_model_name": "evo2_7b",
        "evo2_model_dirs": {"evo2_7b": evo2_model_dir},
    })
    console.print("\n[green]✓ Configuration saved[/green]")

    if not genome_ref:
        console.print(
            "\n[yellow]⚠ No reference FASTA set.[/yellow] The local model services will fall back to\n"
            "  fetching sequence context from the MERGE server (needs network access).\n"
            "  Setting one is strongly recommended: it lets the services cut the exact\n"
            "  windows the models were scored with when the ensemble was trained."
        )

    if click.confirm("Switch default mode to local?", default=True):
        set_mode("local")
        console.print("[green]✓ Switched to local mode[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. [bold]merge local env setup --model evo2[/bold]  # Deploy Evo2 conda environment")
    console.print("  2. [bold]merge local env start --model evo2[/bold]  # Start inference service")
    console.print("  3. [bold]merge local download --file dbnsfp[/bold]  # View dbNSFP download guide")
    console.print("  4. [bold]merge local env status[/bold]              # Verify environment")
    console.print("  5. [bold]merge local predict ...[/bold]             # Start predicting")
    console.print(
        "\n[dim]Optional opt-in models, each its own env + service:\n"
        "  merge local env setup --model carbon        # Carbon-3B      (port 5005)\n"
        "  merge local env setup --model generator     # GENERATOR      (port 5006)\n"
        "  merge local env setup --model generator_v2  # GENERATOR-v2   (port 5007)\n"
        "  merge local env setup --model nt_v2         # NT-v2          (port 5008)\n"
        "then predict with --carbon --generator --generator-v2 --nt-v2 (or --all-models).[/dim]")


# ── merge local download ──────────────────────────────────────
@local.command("download")
@click.option("--file", "file_type",
              type=click.Choice(["all", "dbnsfp", "gpn-msa", "precomputed"]),
              default="all",
              help="Show download instructions for specific file (or auto-download precomputed VCF)")
@click.option("--genome", default="all", type=click.Choice(["all", "hg38", "hg19"]),
              help="Genome version for precomputed VCF download (default: both)")
@click.option("--variant-type", "variant_type", default="all",
              type=click.Choice(["all", "coding", "splicing"]),
              help="Variant type for precomputed VCF download (default: both)")
def local_download(file_type, genome, variant_type):
    """Show official download URLs for pre-computed data files.

    \b
    The precomputed VCF cache files are auto-downloaded from merge.fanglab.cn:
      merge local download --file precomputed              # download all 4 files
      merge local download --file precomputed --genome hg38
      merge local download --file precomputed --genome hg38 --variant-type coding
    """
    cfg = get_local_config()
    data_dir = cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    # ── Auto-download precomputed VCF files ───────────────────────
    if file_type == "precomputed":
        precomputed_dir = cfg.get("precomputed_data_dir") or os.path.join(data_dir, "precomputed")
        os.makedirs(precomputed_dir, exist_ok=True)
        set_local_config("precomputed_data_dir", precomputed_dir)
        _download_precomputed_vcfs(precomputed_dir, genome, variant_type)
        return
    # ─────────────────────────────────────────────────────────────

    INSTRUCTIONS = {
        "dbnsfp": {
            "label": "dbNSFP (ESM1b + AlphaMissense)",
            "url":   "https://www.dbnsfp.org/",
            "steps": [
                "Visit https://www.dbnsfp.org/ and register/login",
                "Download dbNSFP5.3a_grch38.gz (~110 GB) and .tbi index",
                f"Place files in data directory: {data_dir}",
                "hg19 version: dbNSFP5.3a_grch37.gz (optional)",
            ],
        },
        "gpn-msa": {
            "label": "GPN-MSA Pre-computed Scores",
            "url":   "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/tree/main",
            "steps": [
                "Visit Hugging Face dataset page (URL above)",
                "Download scores.tsv.bgz and scores.tsv.bgz.tbi",
                f"Place files in data directory: {data_dir}",
            ],
        },
    }

    console.print(
        "\n[bold green]✓ Ensemble models (pkl files)[/bold green] are bundled with pip install, "
        "no manual download needed.\n"
        "  If model files are missing, run: [bold]pip install --force-reinstall merge-cli[/bold]\n"
        "\n[bold yellow]⚡ Pre-computed VCF cache[/bold yellow] can be auto-downloaded:\n"
        "  [bold]merge local download --file precomputed[/bold]\n"
    )

    show_keys = (list(INSTRUCTIONS.keys()) if file_type == "all"
                 else [file_type] if file_type in INSTRUCTIONS else [])

    for key in show_keys:
        info = INSTRUCTIONS[key]
        console.print(f"[bold cyan]══ {info['label']} ══[/bold cyan]")
        console.print(f"  Official URL: [link={info['url']}]{info['url']}[/link]")
        console.print("  Download steps:")
        for i, step in enumerate(info["steps"], 1):
            console.print(f"    {i}. {step}")
        console.print()


def _download_precomputed_vcfs(data_dir: str, genome: str = "all", variant_type: str = "all"):
    """Auto-download pre-computed VCF files + tabix indices from merge.fanglab.cn."""
    import shutil

    # Files hosted at: https://merge.fanglab.cn/static/precomputed/<filename>
    # Adjust BASE_STATIC_URL if you move the files elsewhere (e.g. OSS / GitHub Releases)
    BASE_STATIC_URL = f"{FIXED_API_URL}/static/precomputed"

    ALL_FILES = {
        ("hg38", "coding"):   "coding_merged.vcf.gz",
        ("hg38", "splicing"): "splicing_merged.vcf.gz",
        ("hg19", "coding"):   "coding_merged_hg19.vcf.gz",
        ("hg19", "splicing"): "splicing_merged_hg19.vcf.gz",
    }

    # Filter by --genome and --variant-type
    targets = {
        (g, vt): fname for (g, vt), fname in ALL_FILES.items()
        if (genome == "all" or g == genome)
        and (variant_type == "all" or vt == variant_type)
    }

    if not targets:
        console.print("[red]No files match the specified genome/variant-type combination.[/red]")
        return

    has_tabix = bool(shutil.which("tabix"))
    if not has_tabix:
        console.print(
            "[yellow]⚠ tabix not found in PATH. VCF files will be downloaded but NOT indexed.\n"
            "  Install htslib (conda install -c bioconda htslib) then run this command again\n"
            "  or index manually: tabix -p vcf <file.vcf.gz>[/yellow]\n"
        )

    ok_count = 0
    for (g, vt), fname in targets.items():
        dest = os.path.join(data_dir, fname)
        dest_tbi = dest + ".tbi"
        url     = f"{BASE_STATIC_URL}/{fname}"
        url_tbi = f"{BASE_STATIC_URL}/{fname}.tbi"

        console.print(f"\n[bold cyan]── {g} {vt}: {fname} ──[/bold cyan]")

        # ── Download .vcf.gz ──────────────────────────────────────
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1e6
            console.print(f"  [green]✓ Already exists ({size_mb:.0f} MB), skipping.[/green]")
        else:
            console.print(f"  Downloading {url} …")
            success = _stream_download(url, dest)
            if not success:
                console.print(f"  [red]✗ Download failed. Check URL or network.[/red]")
                continue

        # ── Download .tbi (pre-built index) ──────────────────────
        if os.path.exists(dest_tbi):
            console.print(f"  [green]✓ Index already exists, skipping.[/green]")
        else:
            console.print(f"  Downloading index {url_tbi} …")
            idx_ok = _stream_download(url_tbi, dest_tbi)
            if not idx_ok:
                # Fall back to local tabix
                if has_tabix:
                    console.print("  Pre-built index not found, building with local tabix…")
                    result = subprocess.run(
                        ["tabix", "-p", "vcf", dest],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        console.print("  [green]✓ Index built.[/green]")
                    else:
                        console.print(f"  [red]✗ tabix failed: {result.stderr.strip()}[/red]")
                        continue
                else:
                    console.print(
                        "  [red]✗ No pre-built index and tabix not found.\n"
                        "     Install htslib and run: tabix -p vcf " + dest + "[/red]"
                    )
                    continue

        console.print(f"  [green]✓ Ready: {dest}[/green]")
        ok_count += 1

    console.print(f"\n[bold green]✓ Done: {ok_count}/{len(targets)} files ready in {data_dir}[/bold green]")
    from . import local_engine
    if ok_count > 0 and local_engine.PRECOMPUTED_CACHE_ENABLED:
        console.print(
            "  Pre-computed features will now be used for variants in these tables.\n"
            "  Run [bold]merge precomputed status[/bold] to verify."
        )
    elif ok_count > 0:
        console.print("  The cache is disabled (MERGE_ENABLE_PRECOMPUTED is not set), "
                      "so these files are not used.")


def _stream_download(url: str, dest: str, chunk_size: int = 1 << 20) -> bool:
    """Download url → dest with a progress bar. Returns True on success."""
    import math
    tmp = dest + ".part"
    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct  = downloaded / total * 100
                        done = int(pct / 5)
                        bar  = "█" * done + "░" * (20 - done)
                        mb   = downloaded / 1e6
                        print(f"\r    [{bar}] {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)
        print()  # newline after progress bar
        os.replace(tmp, dest)
        return True
    except Exception as exc:
        if os.path.exists(tmp):
            os.unlink(tmp)
        console.print(f"\n  [red]Download error: {exc}[/red]")
        return False


# ── merge local predict ───────────────────────────────────────
@local.command("predict")
@click.option("--chrom",   required=True)
@click.option("--pos",     required=True, type=int)
@click.option("--ref",     required=True)
@click.option("--alt",     required=True)
@click.option("--genome",  default="hg38", type=click.Choice(["hg38", "hg19"]))
@click.option("--format",  "fmt", default="table",
              type=click.Choice(["table", "json", "tsv"]))
@click.option("--no-ensemble", is_flag=True, default=False)
@click.option("--ensemble-type",
              type=click.Choice(["coding", "noncoding", "splice", "nonsnv"]),
              default=None,
              help="Manually specify variant type (bypass ANNOVAR): coding / noncoding / splice")
@click.option("--no-shap", is_flag=True, default=False,
              help="Skip SHAP analysis plot (faster prediction)")
@click.option("--annovar-dir", default=None,
              help="Local ANNOVAR directory (overrides config; default: use remote ANNOVAR)")
@click.option("--genome-ref", default=None,
              help="Reference genome FASTA path for local ANNOVAR (e.g. /path/to/hg19.fa)")
@_add_model_options
def local_predict(chrom, pos, ref, alt, genome, fmt,
                  no_ensemble, ensemble_type, no_shap, annovar_dir, genome_ref, **kwargs):
    """Run single variant prediction entirely locally (no rate limit).

    \b
    Examples:
      merge local predict --chrom chr17 --pos 43092919 --ref A --alt G
      merge local predict --chrom chr17 --pos 43092919 --ref A --alt G \\
          --ensemble-type coding --format json
      merge local predict --chrom chr17 --pos 43092919 --ref A --alt G \\
          --annovar-dir /path/to/annovar --genome-ref /path/to/hg38.fa
    """
    from . import local_engine
    chrom = _normalize_chrom(chrom)
    flags = _model_flags(**kwargs)

    # Override ANNOVAR config if specified via CLI
    if annovar_dir:
        from .config import set_local_config
        set_local_config("annovar_dir", annovar_dir)
    if genome_ref:
        from .config import set_local_config
        set_local_config("genome_ref", genome_ref)

    with console.status("[bold cyan]Running local prediction (no rate limit)...[/bold cyan]"):
        try:
            resp = local_engine.predict_local(chrom, pos, ref, alt, genome, **flags)
        except FileNotFoundError as e:
            console.print(f"\n[red]File missing: {e}[/red]")
            console.print("[dim]Run merge local env status to check which files are ready[/dim]")
            sys.exit(1)
        except ImportError as e:
            console.print(f"\n[red]Missing dependency: {e}[/red]"); sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Local prediction failed: {e}[/red]"); sys.exit(1)

    if not resp.get("success"):
        console.print(f"[red]Prediction failed: {resp.get('error')}[/red]"); sys.exit(1)

    prediction = resp["prediction"]
    # Warnings go to stderr so they cannot corrupt machine-readable output;
    # the errors are in the JSON payload anyway.
    if fmt == "json":
        for src, err in prediction.get("errors", {}).items():
            print(f"⚠ {src}: {err}", file=sys.stderr)
    else:
        for src, err in prediction.get("errors", {}).items():
            console.print(f"[yellow]⚠ {src}: {err}[/yellow]")

    ensemble = None
    skip_shap = no_shap
    if not no_ensemble:
        if not no_shap and fmt == "table" and sys.stdin.isatty():
            skip_shap = not click.confirm("\nGenerate SHAP analysis plot?", default=True)
        variant_info = f"{chrom}:{pos} {ref}>{alt}"

        # Resolve variant type: CLI flag > _variant_type from predict_local > remote ANNOVAR API
        effective_ensemble_type = ensemble_type or prediction.get("_variant_type")
        if not effective_ensemble_type:
            try:
                effective_ensemble_type = local_engine.get_variant_type(chrom, pos, ref, alt, genome)
            except Exception:
                effective_ensemble_type = None
        if effective_ensemble_type:
            prediction["_variant_type"] = effective_ensemble_type

        with console.status("[bold cyan]Running local ensemble model (bundled pkl)...[/bold cyan]"):
            try:
                dbnsfp = prediction.get("dbnsfp") or {}
                transcripts = dbnsfp.get("transcripts") or ([dbnsfp] if dbnsfp else [])
                ens_result, ens_err = local_engine.run_local_ensemble(
                    prediction, transcripts, ensemble_type=effective_ensemble_type,
                    skip_shap=skip_shap, variant_info=variant_info)
                if ens_err:
                    console.print(f"[yellow]Ensemble model warning: {ens_err}[/yellow]")
                ensemble = {"success": True, **ens_result}
            except FileNotFoundError as e:
                console.print(f"[red]Ensemble model file missing: {e}[/red]")
                console.print("[dim]Run: pip install --force-reinstall merge-cli[/dim]")
            except Exception as e:
                console.print(f"[yellow]Ensemble model failed: {e}[/yellow]")

    output.render_single(prediction, ensemble, fmt=fmt,
                         errors=prediction.get("errors"))


# ── merge local status ────────────────────────────────────────
@local.command("status")
def local_status():
    """Check local environment status: files / inference services / ANNOVAR."""
    from . import local_engine
    cfg = get_local_config()
    console.print("\n[bold]Local Environment Status[/bold]")
    console.print(f"  Server     : [dim]{FIXED_API_URL}[/dim]")
    console.print(f"  Data dir   : {cfg['data_dir']}")
    console.print(f"  Evo2 model : {local_engine._get_evo2_model_dir()}")

    # GPU
    gpu_info = local_engine.detect_gpu()
    if gpu_info["has_gpu"]:
        console.print(f"  GPU        : [green]{', '.join(gpu_info['gpus'])}[/green]")
    else:
        console.print("  GPU        : [yellow]Not detected (remote mode recommended)[/yellow]")

    # Bundled models
    console.print("\n[bold]Ensemble Model Files (bundled with pip)[/bold]")
    all_files = local_engine.check_local_files()
    for label, info in all_files.items():
        if not info.get("bundled"):
            continue
        if info["exists"]:
            console.print(f"  [green]✓[/green] {label}")
        else:
            console.print(f"  [red]✗[/red] [dim]{label}[/dim]  [dim](reinstall merge-cli)[/dim]")

    # Large data files
    console.print("\n[bold]Pre-computed Data Files[/bold]")
    for label, info in all_files.items():
        if info.get("bundled"):
            continue
        if info["exists"]:
            sz = f"{info['size_gb']} GB" if info["size_gb"] else ""
            console.print(f"  [green]✓[/green] {label:<20} {sz}")
        else:
            console.print(f"  [red]✗[/red] [dim]{label}[/dim]  [dim](not downloaded)[/dim]")

    # Inference services
    console.print("\n[bold]Local Inference Services[/bold]")
    for m in local_engine.ALL_SERVICES:
        h = local_engine.check_service_health(m)
        if h["online"]:
            actual = local_engine.serving_env(m)
            env_tag = f"[dim](env: {actual})[/dim]" if actual else ""
            console.print(
                f"  [green]●[/green] {m:<12} online  "
                f"[dim]{h.get('latency_ms')} ms  {h.get('url')}[/dim]  {env_tag}"
            )
        else:
            env_ok = local_engine._conda_env_exists(local_engine.ENV_NAMES.get(m, m))
            env_tag = ("[green]env deployed[/green]" if env_ok
                       else "[red]env not deployed[/red]")
            console.print(
                f"  [red]○[/red] {m:<12} offline  "
                f"[dim]{h.get('error','')}[/dim]  {env_tag}"
            )

    # ANNOVAR
    console.print("\n[bold]ANNOVAR[/bold]")
    annovar_dir = cfg.get("annovar_dir", "")
    if annovar_dir and os.path.exists(os.path.join(annovar_dir, "annotate_variation.pl")):
        console.print(f"  [green]✓[/green] ANNOVAR configured: {annovar_dir}")
    else:
        console.print(
            "  [yellow]⚠[/yellow] ANNOVAR not configured\n"
            "    [dim]Use --annovar-dir /path/to/annovar or add --ensemble-type coding|noncoding|splice[/dim]"
        )
    genome_ref = cfg.get("genome_ref", "")
    if genome_ref:
        console.print(f"  Reference FASTA: {genome_ref}")
    else:
        console.print("  Reference FASTA: [dim](not set)[/dim]")
    console.print()


# ── merge local batch ─────────────────────────────────────────
@local.command("batch")
@click.argument("vcf_file", type=click.Path(exists=True))
@click.option("--genome", default="hg38", type=click.Choice(["hg38", "hg19"]))
@click.option("--output", "output_path", default=None,
              help="Output CSV file path (default: generated in VCF directory)")
@click.option("--ensemble-type",
              type=click.Choice(["coding", "noncoding", "splice", "nonsnv"]), default=None,
              help="Manually specify variant type; if not set, auto-detect via ANNOVAR")
@_add_model_options
def local_batch(vcf_file, genome, output_path, ensemble_type, **kwargs):
    """Local batch VCF prediction (no rate limit).

    \b
    Examples:
      merge local batch variants.vcf
      merge local batch variants.vcf --genome hg19
      merge local batch variants.vcf --output results.csv
      merge local batch variants.vcf --ensemble-type splice --no-evo2
    """
    from . import local_engine
    flags = _model_flags(**kwargs)
    try:
        out = local_engine.predict_local_batch(
            vcf_file, genome=genome,
            output_path=output_path,
            ensemble_type=ensemble_type,
            console=console,
            **flags,
        )
        console.print(f"\n[bold green]✓ Results saved to: {out}[/bold green]")
    except FileNotFoundError as e:
        console.print(f"[red]File error: {e}[/red]"); sys.exit(1)
    except Exception as e:
        console.print(f"[red]Batch prediction failed: {e}[/red]"); sys.exit(1)


# ── merge local env ───────────────────────────────────────────
@local.group("env")
def local_env():
    """Manage local model conda environments (no Docker needed).

    \b
    Model → conda environment mapping:
      HyenaDNA    → hyenadna    (python=3.10 + transformers)
      NT          → nt          (python=3.10 + transformers)
      Evo2        → evo2        (python=3.12 + CUDA + flash-attn + evo2)
      AlphaGenome → alphagenome (python=3.10 + flask)
      Enformer    → enformer    (external service, default :5004)

    \b
    Quick start:
      merge local env setup --model evo2               # One-click deploy Evo2
      merge local env setup --model evo2 --model-size 1b_base  # Quick test
      merge local env start --model evo2               # Start service
      merge local env status                           # Check service status
      merge local env stop  --model evo2               # Stop service
    """


@local_env.command("setup")
@click.option("--model", "model_name",
              type=click.Choice(["all", "evo2", "hyenadna", "nt", "alphagenome",
                                 "carbon", "generator", "generator_v2", "nt_v2"]),
              default="all", help="Model to deploy (default: all)")
@click.option("--model-size", "model_size",
              default="7b",
              type=click.Choice(["1b_base", "7b", "7b_262k", "7b_base", "20b", "40b"]),
              help="Evo2 model size (7b≈14 GB, 1b_base≈2 GB for testing only, 20b/40b require Hopper GPU)")
@click.option("--local-model-path", default=None,
              help="Existing weights directory for the model being deployed "
                   "(skips the HuggingFace download; required on air-gapped machines)")
def local_env_setup(model_name, model_size, local_model_path):
    """One-click deploy local model conda environment.

    \b
    A dedicated environment per model is the default, not a requirement: any
    environment with the right dependencies (torch + transformers<5 + pyfaidx)
    can serve several of these models. Sharing one saves tens of GB — start the
    service with that environment's interpreter and `merge local env status`
    will show which environment is really behind each port.

    \b
    Evo2 deployment steps (following official guide):
      1. conda create -n evo2 python=3.12
      2. conda install cuda-nvcc cuda-cudart-dev
      3. (non-7b models) conda install transformer-engine-torch=2.3.0
      4. pip install psutil
      5. pip install flash-attn==2.8.0.post2 --no-build-isolation
      6. pip install evo2
      7. (optional) HuggingFace download model weights

    \b
    Examples:
      merge local env setup --model evo2
      merge local env setup --model evo2 --model-size 1b_base
      merge local env setup --model evo2 --local-model-path /path/to/models/evo2_7b
      merge local env setup --model nt   --local-model-path /path/to/nt-500m-model
    """
    from . import local_engine
    from .local_engine import setup_model_env

    if model_name in ("all", "evo2") and model_size == "1b_base":
        console.print(
            "[yellow]Note: evo2_1b_base requires FP8 / Transformer Engine and Hopper GPU, "
            "it is NOT a lightweight alternative to 7B. Only for testing installation.[/yellow]"
        )

    if local_model_path and model_name in ("all", "evo2"):
        from .config import set_local_config
        set_local_config("evo2_model_dir", local_model_path)
        set_local_config("evo2_model_name", f"evo2_{model_size}")
        console.print(f"[green]✓ Local model path recorded: {local_model_path}[/green]")

    targets = (list(local_engine.DEPLOYABLE_MODELS)
               if model_name == "all" else [model_name])

    for m in targets:
        try:
            ok = setup_model_env(
                m, console=console,
                # A weights path only makes sense when deploying one named model;
                # with --model all it would be applied to every model at once.
                local_model_path=(local_model_path if model_name != "all" else None),
                model_size=model_size if m == "evo2" else "7b",
            )
        except KeyboardInterrupt:
            # Interrupting mid-install leaves a conda env that exists but has no
            # packages, which then fails confusingly at start time. Say so.
            console.print(
                f"\n[yellow]⚠ Interrupted while setting up {m}.[/yellow]\n"
                f"  The '{m}' environment may be incomplete — starting the service will\n"
                f"  fail until setup finishes. Re-run when ready:\n"
                f"    [bold]merge local env setup --model {m}[/bold]\n"
                f"  (dependency downloads can take several minutes; leave it running)")
            sys.exit(130)
        if not ok:
            if model_name == "all":
                console.print(f"[yellow]⚠ {m} deployment failed, continuing...[/yellow]")
            else:
                console.print(f"[red]✗ {m} deployment failed, see errors above[/red]")
                sys.exit(1)


@local_env.command("start")
@click.option("--model", "model_name",
              type=click.Choice(["all", "evo2", "hyenadna", "nt", "alphagenome",
                                 "carbon", "generator", "generator_v2", "nt_v2"]),
              default="all", help="Service to start (default: all)")
@click.option("--gpu", default=None,
              help="Pin the service to specific CUDA device(s), e.g. --gpu 2 or --gpu 0,1. "
                   "Evo2 needs an FP8-capable card (compute >= 8.9).")
def local_env_start(model_name, gpu):
    """Start local inference services in the background.

    \b
    Service ports:
      HyenaDNA    →  http://localhost:5001
      NT          →  http://localhost:5002
      Evo2        →  http://localhost:5003
      AlphaGenome →  http://localhost:5000
      Enformer    →  http://localhost:5004

    \b
    View logs:
      tail -f ~/.merge-evo2.log
      tail -f ~/.merge-hyenadna.log
    """
    from . import local_engine
    from .local_engine import start_local_service

    if model_name == "all":
        console.print(
            "[yellow]Note: Starting all models simultaneously may cause OOM on low-memory machines. "
            "Consider starting individually (e.g. --model evo2).[/yellow]"
        )

    targets = (list(local_engine.DEPLOYABLE_MODELS)
               if model_name == "all" else [model_name])
    for m in targets:
        start_local_service(m, console=console, gpu=gpu)


@local_env.command("stop")
@click.option("--model", "model_name",
              type=click.Choice(["all", "evo2", "hyenadna", "nt", "alphagenome",
                                 "carbon", "generator", "generator_v2", "nt_v2"]),
              default="all", help="Service to stop (default: all)")
def local_env_stop(model_name):
    """Stop local inference services (kill processes on corresponding ports)."""
    from . import local_engine
    from .local_engine import stop_local_service

    targets = (list(local_engine.ALL_SERVICES)
               if model_name == "all" else [model_name])
    for m in targets:
        stop_local_service(m, console=console)


@local_env.command("status")
def local_env_status():
    """Check running status of all local inference services."""
    from . import local_engine
    from .local_engine import check_service_health, _conda_env_exists, ENV_NAMES

    console.print("\n[bold]Local Inference Service Status[/bold]")
    for m in local_engine.ALL_SERVICES:
        h = check_service_health(m)
        default_env = ENV_NAMES.get(m, m)
        if h["online"]:
            # Report the environment actually behind the port. Services may be
            # served from a shared environment, which is fine — flagging the
            # default env as "not deployed" while the service answers requests
            # is just confusing.
            actual = local_engine.serving_env(m)
            if actual and actual != default_env:
                env_tag = f"[green]running[/green] [dim](env: {actual})[/dim]"
            elif actual:
                env_tag = f"[green]running[/green] [dim](env: {actual})[/dim]"
            else:
                env_tag = "[green]running[/green]"
            console.print(
                f"  [green]●[/green] {m:<12} online  "
                f"[dim]{h.get('latency_ms')} ms  {h.get('url')}[/dim]  {env_tag}"
            )
        else:
            env_ok = _conda_env_exists(default_env)
            env_tag = ("[green]env deployed[/green]" if env_ok
                       else "[red]env not deployed[/red]")
            console.print(
                f"  [red]○[/red] {m:<12} offline  "
                f"[dim]{h.get('error','')}[/dim]  {env_tag}"
            )
            console.print(
                f"    [dim]→ {'merge local env start' if env_ok else 'merge local env setup'} "
                f"--model {m}[/dim]"
            )
    console.print()


# Lets the CLI run as `python -m merge_cli.cli` when the console scripts are not
# on PATH — the usual situation after a system-wide Windows install, where pip
# puts merge.exe in ...\Scripts\ and warns that it is not on PATH.
if __name__ == "__main__":
    cli()
