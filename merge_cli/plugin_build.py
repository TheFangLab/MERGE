"""Generate the Codex plugin package for merge-cli.

A Codex plugin bundles the things that make a capability discoverable — skills,
MCP configuration, a manifest — into one directory a marketplace can install.
That matters here because Codex defers MCP tool loading: registering the server
is not enough for the model to know it exists, and Codex does not read the
Claude skills directory that `merge skill install` writes to.

The plugin is *generated*, not hand-maintained. Its skill is the same file the
package ships and `merge skill install` copies, so there is exactly one source
of truth for the text; the version comes from merge_cli.__version__. A checked-in
copy lives at plugins/merge-cli/ for a marketplace to point at, and
tests/test_codex_plugin.py fails if it drifts from what this module produces —
the alternative is two copies that quietly disagree, which is how the duplicated
ensemble_predict.py went wrong before.
"""
import json
import shutil
from pathlib import Path

from . import __version__
from .mcp_install import (CODEX_SERVER_NAME, CODEX_STARTUP_TIMEOUT_SEC,
                          CODEX_TOOL_TIMEOUT_SEC, ConfigError, read_config,
                          write_config)
from .skill_install import SKILL_NAME, bundled_skill_dir

PLUGIN_NAME = "merge-cli"          # kebab-case, per the manifest spec
HOMEPAGE = "https://merge.fanglab.cn"
REPOSITORY = "https://github.com/XiaoYangGao0401/merge-cli"
CATEGORY = "Productivity"

# Where Codex looks for locally-sourced plugins, relative to the tree root.
MARKETPLACE_REL = Path(".agents") / "plugins" / "marketplace.json"
MARKETPLACE_NAME = "merge"
MARKETPLACE_DISPLAY_NAME = "MERGE"

INSTALLATION_POLICIES = ("NOT_AVAILABLE", "AVAILABLE", "INSTALLED_BY_DEFAULT")
AUTHENTICATION_POLICIES = ("ON_INSTALL", "ON_USE")

# AVAILABLE, not INSTALLED_BY_DEFAULT: this plugin is inert without the Python
# package behind it, so installing it for someone who has not run
# `pip install "merge-cli[mcp]"` would register a server that cannot start.
DEFAULT_INSTALLATION = "AVAILABLE"
# The MERGE API takes no credentials at all, so this is inert either way; it
# follows the documented default rather than inventing a third state.
DEFAULT_AUTHENTICATION = "ON_INSTALL"


def mcp_config() -> dict:
    """The plugin's .mcp.json.

    `python -m merge_cli.mcp_server` rather than the merge-mcp console script,
    deliberately. A plugin manifest is static and shared, so it cannot carry the
    absolute path that `merge mcp install` resolves per machine — and the script
    is exactly what is missing from PATH on a stock Windows install, where pip
    puts it in ...\\Scripts\\ and warns that the directory is not on PATH.
    `python` is on PATH there; the console script is not.

    The key is the underscore form: Codex normalises hyphens in qualified tool
    names but filters the listing on the raw server key, so a hyphenated server
    starts, answers tools/list, and still shows "Tools: (none)".
    """
    return {
        "mcpServers": {
            CODEX_SERVER_NAME: {
                "command": "python",
                "args": ["-m", "merge_cli.mcp_server"],
                "startup_timeout_sec": CODEX_STARTUP_TIMEOUT_SEC,
                "tool_timeout_sec": CODEX_TOOL_TIMEOUT_SEC,
            }
        }
    }


def manifest() -> dict:
    """The .codex-plugin/plugin.json manifest.

    Asset fields (composerIcon, logo, screenshots) are omitted rather than
    pointed at placeholder files: the spec requires them to resolve to real
    files inside the package, so a reference to something that does not exist is
    worse than no reference. Same for privacyPolicyURL / termsOfServiceURL,
    which must be absolute https URLs and do not exist for this project yet.
    """
    return {
        "name": PLUGIN_NAME,
        "version": __version__,
        "description": ("Predict the pathogenicity of human genetic variants "
                        "with the MERGE ensemble."),
        "author": {"name": "MERGE Team", "url": HOMEPAGE},
        "homepage": HOMEPAGE,
        "repository": REPOSITORY,
        "license": "LicenseRef-Proprietary",
        "keywords": ["genomics", "variant", "pathogenicity", "bioinformatics",
                     "clinical-genetics", "variant-interpretation"],
        "skills": "./skills/",
        "mcpServers": "./.mcp.json",
        "interface": {
            "displayName": "MERGE Variant Pathogenicity",
            "shortDescription": "Is this genetic variant pathogenic?",
            "longDescription": (
                "Scores a human genetic variant with the MERGE ensemble, which "
                "combines 14 sequence and protein models over four variant "
                "classes (coding, splice, noncoding, non-SNV). Returns the "
                "probability together with the gene, the MANE Select "
                "transcript, the ClinVar record when one exists, and explicit "
                "caveats saying when the score should not be trusted. "
                "Requires merge-cli to be installed: pip install "
                "\"merge-cli[mcp]\"."),
            "developerName": "MERGE Team",
            "category": CATEGORY,
            "capabilities": ["Interactive"],
            "websiteURL": HOMEPAGE,
            "defaultPrompt": [
                "Is chr1:1040819 G>GC pathogenic?",
                "Score chr17:43092919 G>A with MERGE.",
                "Is this variant likely pathogenic or benign, and what does "
                "ClinVar say?",
            ],
        },
    }


def _dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def marketplace_entry(path: str,
                      installation: str = DEFAULT_INSTALLATION,
                      authentication: str = DEFAULT_AUTHENTICATION) -> dict:
    """This plugin's entry in a marketplace's `plugins` list.

    `policy.products` is left out: it gates a plugin on the product surface and
    is only meaningful when you actually want that restriction.
    """
    if installation not in INSTALLATION_POLICIES:
        raise ValueError(f"installation must be one of {INSTALLATION_POLICIES}")
    if authentication not in AUTHENTICATION_POLICIES:
        raise ValueError(f"authentication must be one of {AUTHENTICATION_POLICIES}")
    return {
        "name": PLUGIN_NAME,
        "source": {"source": "local", "path": path},
        "policy": {"installation": installation, "authentication": authentication},
        "category": CATEGORY,
    }


def _relative_source_path(root: Path, plugin_dir: Path) -> str:
    """`./plugins/merge-cli` — the plugin's location relative to the tree root.

    Falls back to an absolute path when the plugin was built outside the root,
    which is a valid thing to do and better than emitting a `../..` chain that
    only resolves from one working directory.
    """
    try:
        rel = plugin_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return plugin_dir.resolve().as_posix()
    return "./" + rel.as_posix()


def update_marketplace(root, plugin_dir, installation: str = DEFAULT_INSTALLATION,
                       authentication: str = DEFAULT_AUTHENTICATION,
                       backup: bool = True) -> dict:
    """Add or refresh this plugin's entry in `root`/.agents/plugins/marketplace.json.

    Merges rather than overwrites. A marketplace can list plugins that have
    nothing to do with merge-cli, and it may already carry a display name the
    user chose; replacing the file wholesale would silently drop both. A file
    that exists but does not parse is an error, never an empty starting point.
    """
    root = Path(root)
    path = root / MARKETPLACE_REL
    data = read_config(path)  # raises ConfigError on malformed JSON

    data.setdefault("name", MARKETPLACE_NAME)
    interface = data.setdefault("interface", {})
    if not isinstance(interface, dict):
        raise ConfigError(f'{path}: "interface" is not an object; refusing to edit it.')
    interface.setdefault("displayName", MARKETPLACE_DISPLAY_NAME)

    plugins = data.setdefault("plugins", [])
    if not isinstance(plugins, list):
        raise ConfigError(f'{path}: "plugins" is not a list; refusing to edit it.')

    entry = marketplace_entry(_relative_source_path(root, Path(plugin_dir)),
                              installation, authentication)
    for i, existing in enumerate(plugins):
        if isinstance(existing, dict) and existing.get("name") == PLUGIN_NAME:
            plugins[i] = entry
            break
    else:
        plugins.append(entry)

    made_backup = write_config(path, data, backup=backup)
    return {"path": path, "entry": entry, "backup": made_backup,
            "other_plugins": [p.get("name") for p in plugins
                              if isinstance(p, dict) and p.get("name") != PLUGIN_NAME]}


def build(out_dir, marketplace: bool = True) -> dict:
    """Write the plugin tree to `out_dir`/merge-cli, replacing any previous one.

    When `marketplace` is set, also register it in the marketplace file at the
    parent of `out_dir` — a plugin Codex cannot discover is not much use.
    """
    out_dir = Path(out_dir)
    dest = out_dir / PLUGIN_NAME
    if dest.exists():
        shutil.rmtree(dest)
    (dest / ".codex-plugin").mkdir(parents=True)

    _dump(dest / ".codex-plugin" / "plugin.json", manifest())
    _dump(dest / ".mcp.json", mcp_config())
    shutil.copytree(bundled_skill_dir(), dest / "skills" / SKILL_NAME)
    (dest / "README.md").write_text(_readme(), encoding="utf-8")

    result = {"plugin": dest, "marketplace": None}
    if marketplace:
        result["marketplace"] = update_marketplace(out_dir.parent, dest)
    return result


def _readme() -> str:
    return f"""# MERGE Variant Pathogenicity — Codex plugin

Generated by `merge plugin build`. Do not edit by hand:
`tests/test_codex_plugin.py` fails when this directory differs from what the
command produces, because the skill here and the one `merge skill install`
writes must stay the same text.

## Prerequisite

The plugin carries the skill and the MCP configuration; the server itself comes
from the Python package.

```bash
pip install "merge-cli[mcp]"
```

The MCP entry runs `python -m merge_cli.mcp_server`. A plugin manifest is shared
and static, so it cannot hold the per-machine absolute path that
`merge mcp install` resolves — and on a stock Windows install the `merge-mcp`
console script is precisely what is not on PATH. If `python` on your PATH is not
the interpreter that has merge-cli, run `merge mcp install` instead, which writes
the absolute path.

## Contents

| Path | What |
|---|---|
| `.codex-plugin/plugin.json` | Manifest |
| `.mcp.json` | Registers the MERGE MCP server (7 tools) as `{CODEX_SERVER_NAME}` |
| `skills/{SKILL_NAME}/` | Tells the model when to reach for those tools, and how to report the result |

The server name uses an underscore on purpose: Codex filters its tool listing on
the raw server key, so a hyphenated name starts, answers `tools/list`, and still
shows "Tools: (none)" (openai/codex#15832).

## Discovery

Codex finds a local plugin through a marketplace file, which `merge plugin
build` also writes (and merges into, if one is already there):

```
{MARKETPLACE_REL.as_posix()}
```

Its entry points at this directory with `installation: {DEFAULT_INSTALLATION}` —
not `INSTALLED_BY_DEFAULT`, because the plugin is inert without the Python
package behind it, and installing it for someone who has not run
`pip install "merge-cli[mcp]"` would register a server that cannot start.

## Try it

> Is chr1:1040819 G>GC pathogenic?
"""
