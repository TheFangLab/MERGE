"""Register the merge-cli MCP server in an MCP client's config file.

The hard part of installing an MCP server by hand is not the JSON — it is
knowing the absolute path to the server executable. merge-cli may live in a
venv, in a conda env, under pipx or uv, and each puts `merge-mcp` somewhere
different; the path that ends up in the config is frequently one the client
cannot execute. This module resolves that path from the running interpreter,
which by definition is the one that has merge_cli importable.

Nothing here imports `mcp`: the package only has to be present when the server
actually runs, and `merge mcp status` needs to be able to report that it is
missing rather than crash on it.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# The key this server occupies under "mcpServers". Anything else in the file
# belongs to other servers and must survive untouched.
SERVER_NAME = "merge-cli"


# ── Where each client keeps its config ──────────────────────────────

def _claude_desktop_path() -> Path:
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "Claude"
                / "claude_desktop_config.json")
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def config_path(client: str, scope: str = "user") -> Path:
    """Absolute config path for a client. `scope` applies to claude-code/codex."""
    if client == "claude-desktop":
        return _claude_desktop_path()
    if client == "claude-code":
        # Project scope registers the server for one directory only, which is
        # what you want in a shared repo; user scope makes it available in every
        # session on this machine.
        return (Path.cwd() / ".mcp.json") if scope == "project" else (Path.home() / ".mcp.json")
    if client == "codex":
        # Codex keeps MCP servers in its main TOML config, shared by the CLI,
        # the IDE extension and the ChatGPT desktop app.
        return ((Path.cwd() / ".codex" / "config.toml") if scope == "project"
                else (Path.home() / ".codex" / "config.toml"))
    raise ValueError(f"unknown client: {client}")


CLIENTS = ("claude-desktop", "claude-code", "codex")

# Codex config is TOML, not JSON, and the rest of that file holds unrelated
# Codex settings the user cares about. Python has no TOML writer in the standard
# library, and a parse-then-rewrite round trip would silently drop their comments
# and formatting — so the codex path edits only its own table, as text.
TOML_CLIENTS = ("codex",)


def client_installed(client: str) -> bool:
    """Whether the client looks present, judged by its config directory.

    For claude-code the config is a plain file in the home or project directory,
    so there is nothing to detect — treat it as always available.
    """
    if client == "claude-code":
        return True
    if client == "codex":
        return (Path.home() / ".codex").is_dir()
    return _claude_desktop_path().parent.is_dir()


# ── What to write ───────────────────────────────────────────────────

def script_dirs() -> list:
    """Directories that may hold this install's console scripts, best first.

    sysconfig knows where the running interpreter puts them, which is the only
    reliable answer across layouts: in a venv (either OS) the scripts sit beside
    python itself, but a system-wide Windows install puts python.exe at the root
    and the scripts in a Scripts\\ subdirectory. Looking only beside
    sys.executable finds nothing there.
    """
    dirs = []
    try:
        import sysconfig
        for key in ("scripts", "purelib"):
            p = sysconfig.get_path(key)
            if key == "scripts" and p:
                dirs.append(Path(p))
    except Exception:
        pass
    exe_dir = Path(sys.executable).parent
    for candidate in (exe_dir, exe_dir / "Scripts", exe_dir / "bin"):
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


def resolve_server_command() -> dict:
    """The command an MCP client should run to start this server.

    Prefers the `merge-mcp` console script belonging to this interpreter, which
    is the readable form and what the docs show. Falls back to
    `python -m merge_cli.mcp_server`, which works even when console scripts were
    never generated, live outside PATH, or were installed somewhere this cannot
    guess — clients launch by absolute path, so PATH never has to be correct.
    """
    for d in script_dirs():
        for name in ("merge-mcp.exe", "merge-mcp"):
            candidate = d / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return {"command": str(candidate), "args": []}

    found = shutil.which("merge-mcp")
    if found:
        return {"command": found, "args": []}

    return {"command": sys.executable, "args": ["-m", "merge_cli.mcp_server"]}


def server_entry() -> dict:
    return {**resolve_server_command(), "env": {}}


def mcp_package_available() -> bool:
    """Whether the optional `mcp` dependency is importable (not imported here)."""
    import importlib.util
    return importlib.util.find_spec("mcp") is not None


# ── Reading and writing the config, without losing what is there ────

class ConfigError(RuntimeError):
    pass


def read_config(path: Path) -> dict:
    """Parse a client config, or return {} when it does not exist yet.

    A file that exists but does not parse is an error, never an empty dict:
    overwriting it would silently discard every other MCP server the user has
    registered.
    """
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from None
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{path} is not valid JSON (line {e.lineno}, column {e.colno}: {e.msg}). "
            "Refusing to touch it — fix or move the file, then retry.") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a JSON object at the top level.")
    return data


def backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")


def write_config(path: Path, data: dict, backup: bool = True) -> Optional[Path]:
    """Write the config atomically, backing up any previous version first."""
    made_backup = None
    if backup and path.exists():
        made_backup = backup_path(path)
        shutil.copy2(path, made_backup)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Replace in one step so a crash mid-write cannot leave a truncated config.
    os.replace(tmp, path)
    return made_backup


# ── TOML clients (Codex) ────────────────────────────────────────────

# Codex must not be given a hyphenated server name. From Codex v0.116.0 a tool
# name normalisation replaced hyphens with underscores in qualified names, while
# the listing still filtered on the raw server key — so a server registered as
# "merge-cli" starts, answers tools/list correctly, and still shows
# "Tools: (none)" with the model unable to call anything.
# See https://github.com/openai/codex/issues/15832.
# JSON clients keep the hyphenated name: they handle it fine, and renaming there
# would orphan every existing registration.
CODEX_SERVER_NAME = SERVER_NAME.replace("-", "_")

# Codex applies its own timeouts to an MCP server and discards whatever is
# slower, so the defaults have to be widened here or predict_variant's answer is
# thrown away rather than shown — its per-call limit cut a scoring call off at
# 300s in practice.
#
# 240, lowered from 600. The outer limit should sit just above what the server
# can actually take, not far above: while predict_variant's budget did not reach
# the remote request, a stuck call ran to the outer limit, and every abandoned
# attempt left another merge-mcp process behind. The server now bounds itself
# at time_budget (150s for answer_variant_pathogenicity), so 240 is headroom
# rather than a place to hang.
#
# The startup allowance covers a cold Windows launch, where a freshly written
# .exe is scanned by antivirus before it may run — OpenAI's own bundled server
# ships the same 120s for that reason.
# Only Codex gets these: they are keys in its schema, not the JSON clients'.
CODEX_STARTUP_TIMEOUT_SEC = 120
CODEX_TOOL_TIMEOUT_SEC = 240


def server_key(client: str) -> str:
    return CODEX_SERVER_NAME if client in TOML_CLIENTS else SERVER_NAME


def _toml_header(key: str) -> str:
    return f"[mcp_servers.{key}]"


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f'"{_toml_escape(str(v))}"'


def render_toml_table(entry: dict, key: str = CODEX_SERVER_NAME) -> str:
    """The `[mcp_servers.<key>]` block for a Codex config.

    command and args come first because that is how Codex writes its own
    entries; anything else in `entry` (the timeouts) follows in order. `env` is
    skipped — Codex's stdio schema has no such key.
    """
    args = ", ".join(f'"{_toml_escape(a)}"' for a in entry.get("args", []))
    lines = [_toml_header(key),
             f'command = "{_toml_escape(entry["command"])}"',
             f"args = [{args}]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in entry.items()
              if k not in ("command", "args", "env")]
    return "\n".join(lines) + "\n"


def _split_toml_table(text: str, key: str = CODEX_SERVER_NAME) -> tuple:
    """Split a TOML document around our table: (before, table, after).

    Our table runs from its header to the next top-level header or end of file.
    Everything outside it — other tables, comments, blank lines — is returned
    untouched, so writing it back preserves the rest of the user's Codex config
    exactly as they wrote it.
    """
    header = _toml_header(key)
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        return text, None, ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("[") and not stripped.startswith("[["):
            end = j
            break
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def read_toml_entry(path: Path, key: str = CODEX_SERVER_NAME) -> Optional[dict]:
    """The registered command/args for this server, or None. Validates the file."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e}") from None
    try:
        import tomllib
        data = tomllib.loads(text)
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        data = None
    except Exception as e:
        raise ConfigError(
            f"{path} is not valid TOML ({e}). Refusing to touch it — fix or move "
            "the file, then retry.") from None
    if data is None:
        return None
    entry = (data.get("mcp_servers") or {}).get(key)
    return entry if isinstance(entry, dict) else None


def write_toml_entry(path: Path, entry: dict, backup: bool = True,
                     key: str = CODEX_SERVER_NAME) -> Optional[Path]:
    """Replace or append only our table, leaving the rest of the file verbatim."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    # Drop a table left by an earlier install under the hyphenated name, which
    # Codex accepts but cannot surface tools from. Without this the user ends up
    # with two entries, one of them permanently silent.
    if key != SERVER_NAME:
        stale_before, stale, stale_after = _split_toml_table(text, SERVER_NAME)
        if stale is not None:
            text = stale_before + stale_after

    before, existing, after = _split_toml_table(text, key)
    block = render_toml_table(entry, key)
    if existing is None:
        joiner = "" if (not before or before.endswith("\n\n")) else (
            "\n" if before.endswith("\n") else "\n\n")
        new_text = before + joiner + block
    else:
        new_text = before + block + ("" if after.startswith("\n") or not after else "\n") + after

    made_backup = None
    if backup and path.exists():
        made_backup = backup_path(path)
        shutil.copy2(path, made_backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)
    return made_backup


# ── The operations the CLI exposes ──────────────────────────────────

def plan_install(client: str, scope: str = "user") -> dict:
    """Work out what installing would change, without changing anything."""
    path = config_path(client, scope)
    entry = server_entry()

    if client in TOML_CLIENTS:
        # Codex has no "env" key in its stdio schema; send command/args plus the
        # timeouts its defaults are too tight for.
        entry = {k: v for k, v in entry.items() if k != "env"}
        entry["startup_timeout_sec"] = CODEX_STARTUP_TIMEOUT_SEC
        entry["tool_timeout_sec"] = CODEX_TOOL_TIMEOUT_SEC
        existing = read_toml_entry(path, server_key(client))
        # A table left under the hyphenated name registers, starts and answers
        # tools/list, yet shows "Tools: (none)" and cannot be called. It has to
        # count as work to do: if the current entry alone decided this, an
        # up-to-date install would report "already current" and leave the silent
        # duplicate in place forever, which is exactly what happened in the wild.
        stale = read_toml_entry(path, SERVER_NAME)
        return {
            "client": client, "scope": scope, "path": path,
            "exists": path.exists(), "entry": entry, "existing": existing,
            "already_current": existing == entry and stale is None,
            "stale_entry": stale, "other_servers": [],
            "format": "toml",
        }

    data = read_config(path)
    servers = data.get("mcpServers")
    if servers is not None and not isinstance(servers, dict):
        raise ConfigError(f'{path}: "mcpServers" is not an object; refusing to edit it.')
    servers = servers or {}

    existing = servers.get(SERVER_NAME)
    return {
        "client": client,
        "scope": scope,
        "path": path,
        "exists": path.exists(),
        "entry": entry,
        "existing": existing,
        "already_current": existing == entry,
        "other_servers": sorted(k for k in servers if k != SERVER_NAME),
        "format": "json",
    }


def apply_install(plan: dict, backup: bool = True) -> Optional[Path]:
    """Merge the server entry into the config named by `plan`."""
    path = plan["path"]
    if plan.get("format") == "toml":
        return write_toml_entry(path, plan["entry"], backup=backup,
                                key=server_key(plan["client"]))
    data = read_config(path)
    data.setdefault("mcpServers", {})[SERVER_NAME] = plan["entry"]
    return write_config(path, data, backup=backup)


def uninstall(client: str, scope: str = "user") -> dict:
    """Remove only this server's entry, leaving every other one in place."""
    path = config_path(client, scope)

    if client in TOML_CLIENTS:
        key = server_key(client)
        if read_toml_entry(path, key) is None:
            return {"path": path, "removed": False}
        text = path.read_text(encoding="utf-8")
        before, _, after = _split_toml_table(text, key)
        made_backup = backup_path(path)
        shutil.copy2(path, made_backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(before + after, encoding="utf-8")
        os.replace(tmp, path)
        return {"path": path, "removed": True, "backup": made_backup}

    data = read_config(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return {"path": path, "removed": False}
    servers.pop(SERVER_NAME)
    made_backup = write_config(path, data)
    return {"path": path, "removed": True, "backup": made_backup}


def status() -> dict:
    """Where this server is registered, and whether those entries still work."""
    current = server_entry()
    report = {
        "resolved_command": current,
        "mcp_package": mcp_package_available(),
        "clients": [],
    }
    for client in CLIENTS:
        for scope in (("user", "project") if client in ("claude-code", "codex") else ("user",)):
            path = config_path(client, scope)
            info = {"client": client, "scope": scope, "path": path,
                    "installed": client_installed(client), "registered": False}
            try:
                if client in TOML_CLIENTS:
                    entry = read_toml_entry(path, server_key(client))
                    # Surface the silent duplicate rather than making the user
                    # discover it by wondering why the tools never show up.
                    info["stale_entry"] = read_toml_entry(path, SERVER_NAME) is not None
                else:
                    entry = (read_config(path).get("mcpServers") or {}).get(SERVER_NAME)
            except ConfigError as e:
                info["error"] = str(e)
                report["clients"].append(info)
                continue
            if entry:
                info["registered"] = True
                info["entry"] = entry
                # Codex's stdio schema has no env key and does carry timeouts,
                # so compare against the shape actually written for that client
                # rather than the JSON one — otherwise every Codex entry reads
                # as out of date.
                expected = current
                if client in TOML_CLIENTS:
                    expected = {k: v for k, v in current.items() if k != "env"}
                    expected["startup_timeout_sec"] = CODEX_STARTUP_TIMEOUT_SEC
                    expected["tool_timeout_sec"] = CODEX_TOOL_TIMEOUT_SEC
                info["current"] = entry == expected
                # A registered entry whose command no longer exists is the usual
                # symptom of merge-cli having been reinstalled elsewhere.
                cmd = entry.get("command")
                info["command_exists"] = bool(cmd) and (
                    Path(cmd).is_file() or shutil.which(str(cmd)) is not None)
            report["clients"].append(info)
    return report
