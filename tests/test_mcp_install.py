"""Regression guard for `merge mcp install` (merge_cli/mcp_install.py).

This code edits a file the user did not create and may share with other MCP
servers, so the tests here are mostly about what it must NOT do: never drop
another server's entry, never drop unrelated top-level keys, and never write
over a config it could not parse — a config that fails to parse is far more
likely to be a file worth saving than one worth replacing.

Entirely offline: temporary directories, no network, no client required.

Run:  ./.venv/bin/python tests/test_mcp_install.py
"""
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path

from merge_cli import mcp_install as mi

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Resolving the server command ────────────────────────────────────
print("resolve_server_command")

cmd = mi.resolve_server_command()
check("returns a command", bool(cmd.get("command")), f"— {cmd}")
runnable = Path(cmd["command"]).is_file() or shutil.which(cmd["command"]) is not None
check("the command actually exists on disk", runnable, f"— {cmd['command']}")
# Whichever form it picked, it must be able to start the server: either the
# console script, or this interpreter with -m.
if cmd["args"]:
    check("fallback form targets the mcp_server module",
          cmd["args"] == ["-m", "merge_cli.mcp_server"], f"— {cmd['args']}")
else:
    check("console-script form has an empty args list", cmd["args"] == [])

check("server_entry carries an env dict", isinstance(mi.server_entry().get("env"), dict))

# A system-wide Windows install puts python.exe at the root and the console
# scripts in a Scripts\ subdirectory, so looking only beside sys.executable
# finds nothing. sysconfig knows the real location; the Scripts/bin siblings are
# belt and braces.
dirs = mi.script_dirs()
check("script_dirs includes sysconfig's scripts directory",
      Path(sysconfig.get_path("scripts")) in dirs, f"— {dirs[0]}")
check("script_dirs also probes a Scripts/ subdirectory",
      any(d.name in ("Scripts", "bin") for d in dirs))
check("script_dirs has no duplicates", len(dirs) == len(set(dirs)))

# With no console script anywhere, the command must still be able to start the
# server — clients launch by absolute path, so PATH is never required.
_real_dirs, _real_which = mi.script_dirs, shutil.which
try:
    mi.script_dirs = lambda: [Path(tempfile.gettempdir()) / "merge-nonexistent-dir"]
    mi.shutil.which = lambda _n: None
    fb = mi.resolve_server_command()
    check("falls back to python -m when no script is found",
          fb["command"] == sys.executable
          and fb["args"] == ["-m", "merge_cli.mcp_server"], f"— {fb}")
finally:
    mi.script_dirs, mi.shutil.which = _real_dirs, _real_which


# ── Reading configs ─────────────────────────────────────────────────
print("\nread_config")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    check("missing file reads as empty", mi.read_config(td / "nope.json") == {})

    empty = td / "empty.json"
    empty.write_text("   \n")
    check("whitespace-only file reads as empty", mi.read_config(empty) == {})

    broken = td / "broken.json"
    broken.write_text('{"mcpServers": {,,, BROKEN')
    original = broken.read_text()
    try:
        mi.read_config(broken)
        check("malformed JSON raises", False)
    except mi.ConfigError as e:
        check("malformed JSON raises ConfigError", True)
        check("the error names the file and the position",
              str(broken) in str(e) and "line" in str(e))
    check("malformed file is left byte-for-byte alone",
          broken.read_text() == original)

    notobj = td / "list.json"
    notobj.write_text("[1, 2, 3]")
    try:
        mi.read_config(notobj)
        check("top-level non-object raises", False)
    except mi.ConfigError:
        check("top-level non-object raises ConfigError", True)


# ── Installing without collateral damage ────────────────────────────
print("\nplan_install / apply_install")

with tempfile.TemporaryDirectory() as td:
    home = Path(td) / "home"
    home.mkdir()
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    try:
        cfg = home / ".mcp.json"

        # A config that looks like a real user's: other servers, an outdated
        # merge-cli entry, and an unrelated top-level key.
        write_json(cfg, {
            "mcpServers": {
                "github": {"command": "/usr/bin/gh-mcp", "args": ["--stdio"]},
                "merge-cli": {"command": "/old/dead/path/merge-mcp", "args": [], "env": {}},
            },
            "unrelatedTopLevelKey": {"keep": "me"},
        })

        plan = mi.plan_install("claude-code", "user")
        check("plan targets the user config", plan["path"] == cfg, f"— {plan['path']}")
        check("plan sees the stale entry", plan["existing"]["command"] == "/old/dead/path/merge-mcp")
        check("plan knows it is not current", plan["already_current"] is False)
        check("plan lists the other servers", plan["other_servers"] == ["github"])

        before = cfg.read_text()
        mi.plan_install("claude-code", "user")
        check("planning writes nothing", cfg.read_text() == before)

        backup = mi.apply_install(plan)
        after = json.loads(cfg.read_text())

        check("other server survives untouched",
              after["mcpServers"]["github"] == {"command": "/usr/bin/gh-mcp", "args": ["--stdio"]})
        check("unrelated top-level key survives",
              after["unrelatedTopLevelKey"] == {"keep": "me"})
        check("merge-cli entry was replaced",
              after["mcpServers"]["merge-cli"] == mi.server_entry())
        check("a backup was written", backup is not None and Path(backup).exists())
        check("the backup holds the previous content",
              json.loads(Path(backup).read_text())["mcpServers"]["merge-cli"]["command"]
              == "/old/dead/path/merge-mcp")
        check("no temp file left behind", not list(home.glob("*.tmp")))

        # Running it again must be a no-op the caller can detect.
        check("second plan reports already current",
              mi.plan_install("claude-code", "user")["already_current"] is True)

        # ── status ──
        rep = mi.status()
        entry = next(c for c in rep["clients"]
                     if c["client"] == "claude-code" and c["scope"] == "user")
        check("status sees the registration", entry["registered"] is True)
        check("status marks it current", entry.get("current") is True)
        check("status confirms the command exists", entry.get("command_exists") is True)

        # A stale path must be reported, not silently accepted.
        data = json.loads(cfg.read_text())
        data["mcpServers"]["merge-cli"]["command"] = "/definitely/not/here/merge-mcp"
        write_json(cfg, data)
        entry = next(c for c in mi.status()["clients"]
                     if c["client"] == "claude-code" and c["scope"] == "user")
        check("status flags a command that no longer exists",
              entry.get("command_exists") is False)

        # ── uninstall ──
        res = mi.uninstall("claude-code", "user")
        after = json.loads(cfg.read_text())
        check("uninstall reports removal", res["removed"] is True)
        check("uninstall removed merge-cli", "merge-cli" not in after["mcpServers"])
        check("uninstall kept the other server", "github" in after["mcpServers"])
        check("uninstall kept unrelated keys", "unrelatedTopLevelKey" in after)
        check("uninstalling again is a no-op",
              mi.uninstall("claude-code", "user")["removed"] is False)

        # ── a config that does not exist yet ──
        fresh = home / ".mcp.json"
        fresh.unlink()
        plan = mi.plan_install("claude-code", "user")
        check("plan handles a missing config", plan["exists"] is False)
        mi.apply_install(plan)
        check("install creates the config",
              json.loads(fresh.read_text())["mcpServers"]["merge-cli"] == mi.server_entry())
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


# ── Codex (TOML) ────────────────────────────────────────────────────
print("\ncodex / TOML")

CODEX_CONFIG = '''# my codex config, do not lose this
model = "gpt-5-codex"

[mcp_servers.tooluniverse]
command = "uvx"
args = ["--refresh", "tooluniverse"]

[sandbox_workspace_write]
network_access = true
'''

with tempfile.TemporaryDirectory() as td:
    home = Path(td) / "home"
    (home / ".codex").mkdir(parents=True)
    cfg = home / ".codex" / "config.toml"
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    try:
        import tomllib

        cfg.write_text(CODEX_CONFIG, encoding="utf-8")
        plan = mi.plan_install("codex", "user")
        check("codex plans against config.toml", plan["path"] == cfg)
        check("codex uses the toml format", plan.get("format") == "toml")
        check("codex entry omits env (not in Codex's stdio schema)",
              "env" not in plan["entry"])

        mi.apply_install(plan)
        text = cfg.read_text()
        data = tomllib.loads(text)

        check("result is still valid TOML", "mcp_servers" in data)
        check("the user's comment survives", "do not lose this" in text)
        check("an unrelated top-level key survives", data.get("model") == "gpt-5-codex")
        check("another server survives", data["mcp_servers"]["tooluniverse"]["command"] == "uvx")
        check("another table survives", data["sandbox_workspace_write"]["network_access"] is True)
        check("registered under the underscore name Codex can surface",
              data["mcp_servers"]["merge_cli"]["command"] == mi.server_entry()["command"])
        check("the hyphenated name is not used for codex",
              "merge-cli" not in data["mcp_servers"])
        check("installing again is a no-op",
              mi.plan_install("codex", "user")["already_current"] is True)

        # Codex discards a tool result that outruns its own limits, so the entry
        # has to widen them or predict_variant's answer is thrown away unseen.
        entry = data["mcp_servers"]["merge_cli"]
        check("codex entry carries a startup timeout",
              entry.get("startup_timeout_sec") == mi.CODEX_STARTUP_TIMEOUT_SEC)
        check("codex entry carries a tool-call timeout",
              entry.get("tool_timeout_sec") == mi.CODEX_TOOL_TIMEOUT_SEC)
        check("the timeouts are TOML integers, not strings",
              isinstance(entry.get("tool_timeout_sec"), int))

        # A hyphenated leftover starts and answers tools/list but shows
        # "Tools: (none)". The cleanup lives in the write path, so an already
        # up-to-date install used to skip it and leave the duplicate forever.
        cfg.write_text(cfg.read_text(encoding="utf-8")
                       + '\n[mcp_servers.merge-cli]\ncommand = "old.exe"\nargs = []\n',
                       encoding="utf-8")
        stale_plan = mi.plan_install("codex", "user")
        check("a stale hyphenated table makes the install not-current",
              stale_plan["already_current"] is False)
        check("the stale table is reported in the plan",
              stale_plan.get("stale_entry", {}).get("command") == "old.exe")
        mi.apply_install(stale_plan)
        data = tomllib.loads(cfg.read_text(encoding="utf-8"))
        check("applying it removes the stale hyphenated table",
              "merge-cli" not in data["mcp_servers"])
        check("and keeps the working underscore one",
              data["mcp_servers"]["merge_cli"]["command"] == mi.server_entry()["command"])
        check("now it is current again",
              mi.plan_install("codex", "user")["already_current"] is True)

        # Replacing a stale block that sits in the middle of the file is the
        # case a naive append would corrupt.
        cfg.write_text(
            'model = "gpt-5-codex"\n\n'
            '[mcp_servers.merge_cli]\n'
            'command = "C:\\\\old\\\\merge-mcp.exe"\n'
            'args = ["--stale"]\n\n'
            '[sandbox_workspace_write]\n'
            'network_access = true\n', encoding="utf-8")
        check("a stale windows-style entry is read back",
              mi.read_toml_entry(cfg)["command"] == "C:\\old\\merge-mcp.exe")
        mi.apply_install(mi.plan_install("codex", "user"))
        data = tomllib.loads(cfg.read_text())
        check("mid-file block replaced, not duplicated",
              cfg.read_text().count("[mcp_servers.merge_cli]") == 1)
        check("the table after it survives replacement",
              data["sandbox_workspace_write"]["network_access"] is True)
        check("args were replaced too", data["mcp_servers"]["merge_cli"]["args"] == [])

        res = mi.uninstall("codex", "user")
        data = tomllib.loads(cfg.read_text())
        check("codex uninstall reports removal", res["removed"] is True)
        check("codex uninstall removed only our table",
              "mcp_servers" not in data or "merge_cli" not in data.get("mcp_servers", {}))
        check("codex uninstall left the rest valid",
              data.get("model") == "gpt-5-codex"
              and data["sandbox_workspace_write"]["network_access"] is True)
        check("codex uninstalling again is a no-op",
              mi.uninstall("codex", "user")["removed"] is False)

        # Windows paths must survive the TOML round trip.
        rendered = mi.render_toml_table({"command": r"C:\Users\x\.venv\Scripts\merge-mcp.exe",
                                         "args": []})
        parsed = tomllib.loads(rendered)
        check("a Windows path round-trips through TOML escaping",
              parsed["mcp_servers"]["merge_cli"]["command"]
              == r"C:\Users\x\.venv\Scripts\merge-mcp.exe", f"— {rendered.splitlines()[1]}")

        # Migration: an entry left by an earlier install under the hyphenated
        # name must be removed, not left alongside the new one — Codex accepts
        # it but can never surface its tools (openai/codex#15832).
        cfg.write_text(
            'model = "gpt-5-codex"\n\n'
            '[mcp_servers.node_repl]\n'
            'command = "node_repl.exe"\n'
            'startup_timeout_sec = 120\n\n'
            '[mcp_servers.node_repl.env]\n'
            'CODEX_HOME = "C:\\\\x\\\\.codex"\n\n'
            '[mcp_servers.merge-cli]\n'
            'command = "C:\\\\old\\\\merge-mcp.exe"\n'
            'args = []\n', encoding="utf-8")
        mi.apply_install(mi.plan_install("codex", "user"))
        data = tomllib.loads(cfg.read_text())
        check("the hyphenated entry is migrated away",
              "merge-cli" not in data["mcp_servers"])
        check("the underscore entry replaces it",
              data["mcp_servers"]["merge_cli"]["command"] == mi.server_entry()["command"])
        check("a sibling server's sub-table survives migration",
              data["mcp_servers"]["node_repl"]["env"]["CODEX_HOME"] == "C:\\x\\.codex"
              and data["mcp_servers"]["node_repl"]["startup_timeout_sec"] == 120)
        check("server_key maps codex to the underscore name",
              mi.server_key("codex") == "merge_cli" and mi.server_key("claude-code") == "merge-cli")

        broken = home / ".codex" / "bad.toml"
        broken.write_text("model = = =\n")
        try:
            mi.read_toml_entry(broken)
            check("malformed TOML raises", False)
        except mi.ConfigError:
            check("malformed TOML raises ConfigError", True)
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home


# ── Config paths ────────────────────────────────────────────────────
print("\nconfig_path")

check("claude-code user scope is in the home directory",
      mi.config_path("claude-code", "user") == Path.home() / ".mcp.json")
check("claude-code project scope is in the working directory",
      mi.config_path("claude-code", "project") == Path.cwd() / ".mcp.json")
desktop = mi.config_path("claude-desktop")
check("claude-desktop path is absolute and named correctly",
      desktop.is_absolute() and desktop.name == "claude_desktop_config.json", f"— {desktop}")
try:
    mi.config_path("emacs")
    check("an unknown client raises", False)
except ValueError:
    check("an unknown client raises ValueError", True)

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
else:
    print("All mcp_install checks passed.")

raise SystemExit(1 if failures else 0)
