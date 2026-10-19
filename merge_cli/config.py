"""
merge_cli/config.py
Fixed server address: https://merge.fanglab.cn/ (no authentication, no URL configuration).
"""
import os, json
from pathlib import Path

# ── Fixed server address ───────────────────────────────────────
FIXED_API_URL = "https://merge.fanglab.cn"

_CONFIG_FILE = Path.home() / ".config" / "merge-cli" / "config.json"


def _load_file() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_file(data: dict) -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2))


def get_api_url() -> str:
    """Always returns the fixed server address (not user-configurable)."""
    return FIXED_API_URL


def get_mode() -> str:
    return _load_file().get("mode", "remote")


def set_mode(mode: str) -> None:
    if mode not in ("remote", "local"):
        raise ValueError(f"mode must be 'remote' or 'local', got: {mode}")
    cfg = _load_file()
    cfg["mode"] = mode
    _save_file(cfg)


_LOCAL_DEFAULTS = {
    "data_dir":     str(Path.home() / ".merge-data"),
    "precomputed_data_dir": str(Path.home() / ".merge-data" / "precomputed"),
    "model_dir":    str(Path.home() / ".merge-models"),
    "annovar_dir":  "",
    "genome_ref":   "",
    "alphagenome_url": "http://localhost:5000",
    "hyenadna_url": "http://localhost:5001",
    "nt_url":       "http://localhost:5002",
    "evo2_url":     "http://localhost:5003",
    "enformer_url": "http://localhost:5004",
    # Enformer is a TensorFlow SavedModel plus a normalisation transform pickle.
    "enformer_model_path": "",
    "enformer_transform_path": "",
    # Independent local DNA-LM services (one conda env + one service each).
    "carbon_url":       "http://localhost:5005",
    "generator_url":    "http://localhost:5006",
    "generator_v2_url": "http://localhost:5007",
    "nt_v2_url":        "http://localhost:5008",
    "evo2_model_dir": "",
    "evo2_model_name": "evo2_7b",
    # Per-variant Evo2 weight directories, keyed by model name. Lets local mode
    # score evo2_7b / evo2_7b_base / evo2_1b_base without reconfiguring.
    "evo2_model_dirs": {},
}


def get_local_config() -> dict:
    cfg = _load_file()
    local = cfg.get("local", {})
    # Remove legacy deprecated fields for backward compatibility
    local.pop("evo1_url", None)
    return {**_LOCAL_DEFAULTS, **local}


def get_alphagenome_api_key() -> str:
    """The user's own AlphaGenome API key, or "" if they have not set one.

    One resolver for both modes. Local mode bakes it into the service it
    generates; remote mode forwards it to the MERGE server, which uses it for
    that one request and never stores it. The environment wins over the config
    file so a key can be supplied per-invocation without touching disk.

    Callers must treat the result as a secret: never print it unmasked, never
    put it in a log line, an error message or a saved payload.
    """
    return (os.environ.get("ALPHAGENOME_API_KEY")
            or get_local_config().get("alphagenome_api_key")
            or "")


def set_local_config(key: str, value) -> None:
    cfg = _load_file()
    if "local" not in cfg:
        cfg["local"] = {}
    cfg["local"][key] = value
    _save_file(cfg)


def set_local_config_bulk(updates: dict) -> None:
    cfg = _load_file()
    if "local" not in cfg:
        cfg["local"] = {}
    cfg["local"].update(updates)
    _save_file(cfg)


def show_config() -> dict:
    local_cfg = get_local_config()
    return {
        "api_url":     FIXED_API_URL,
        "mode":        get_mode(),
        "config_file": str(_CONFIG_FILE),
        "local":       local_cfg,
    }
