"""Regression guard for the Codex plugin package.

Why this exists
---------------
The plugin is a second home for text that already lives in the Python package:
the skill, the version, the Codex server name and its timeouts. Two copies of
the same thing is how the duplicated ensemble_predict.py went wrong — the copies
diverge, and nothing announces it. So the checked-in plugins/merge-cli/ is
generated, and this test fails when it no longer matches what `merge plugin
build` produces from the current source.

The manifest checks encode the rules the spec states, because a manifest that
violates them fails at install time in Codex rather than here — a slow, remote
way to find out that a path was absolute or a version was not semver.

Entirely offline.

Run:  ./.venv/bin/python tests/test_codex_plugin.py
"""
import json
import re
import tempfile
from pathlib import Path

import merge_cli
from merge_cli import mcp_install as mi, plugin_build as pb, skill_install as si

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


REPO = Path(__file__).resolve().parent.parent
CHECKED_IN = REPO / "plugins" / pb.PLUGIN_NAME

# ── The manifest follows the spec ───────────────────────────────────
print("plugin.json")

m = pb.manifest()
check("name is kebab-case with no spaces",
      re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", m["name"]) is not None, f"— {m['name']}")
check("version is strict semver",
      re.fullmatch(r"\d+\.\d+\.\d+", m["version"]) is not None, f"— {m['version']}")
check("version tracks the package, not a hand-edited copy",
      m["version"] == merge_cli.__version__)

for key in ("skills", "mcpServers"):
    check(f"{key} is a relative ./ path",
          isinstance(m[key], str) and m[key].startswith("./"), f"— {m[key]}")

# apps/mcpServers may only be declared when the file is actually there.
check("apps is not declared (there is no .app.json)", "apps" not in m)

iface = m["interface"]
for key in ("websiteURL", "privacyPolicyURL", "termsOfServiceURL"):
    if key in iface:
        check(f"{key} is an absolute https URL", iface[key].startswith("https://"),
              f"— {iface[key]}")
check("websiteURL is present and https", iface["websiteURL"].startswith("https://"))

# Asset fields must resolve to real files inside the package. Referencing a
# placeholder that does not exist is worse than omitting the field.
for key in ("composerIcon", "logo"):
    if key in iface:
        check(f"{key} points at a file that exists",
              (CHECKED_IN / iface[key]).is_file(), f"— {iface[key]}")
for shot in iface.get("screenshots", []):
    check("screenshot exists and is under ./assets/",
          shot.startswith("./assets/") and (CHECKED_IN / shot).is_file(), f"— {shot}")
check("defaultPrompt gives Codex something concrete to offer",
      len(iface["defaultPrompt"]) >= 1
      and all(isinstance(p, str) and p for p in iface["defaultPrompt"]))

# ── The MCP entry ───────────────────────────────────────────────────
print("\n.mcp.json")

servers = pb.mcp_config()["mcpServers"]
check("exactly one server is declared", len(servers) == 1)
name, entry = next(iter(servers.items()))
# Codex normalises hyphens in qualified tool names but filters the listing on
# the raw key, so a hyphenated server shows "Tools: (none)" (openai/codex#15832).
check("the server key is the underscore form Codex can surface",
      name == mi.CODEX_SERVER_NAME and "-" not in name, f"— {name}")
check("it launches via python -m, not the console script",
      entry["command"] == "python" and entry["args"] == ["-m", "merge_cli.mcp_server"],
      f"— {entry['command']} {entry['args']}")
check("the startup timeout matches what mcp install writes",
      entry["startup_timeout_sec"] == mi.CODEX_STARTUP_TIMEOUT_SEC)
check("the tool timeout matches what mcp install writes",
      entry["tool_timeout_sec"] == mi.CODEX_TOOL_TIMEOUT_SEC)

# ── One skill, not two ──────────────────────────────────────────────
print("\nthe skill is not duplicated")

with tempfile.TemporaryDirectory() as tmp:
    built = pb.build(Path(tmp) / "plugins")["plugin"]

    packaged = (si.bundled_skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    in_plugin = (built / "skills" / si.SKILL_NAME / "SKILL.md").read_text(encoding="utf-8")
    check("the plugin ships the same skill text the package does",
          in_plugin == packaged)

    # removeprefix, not lstrip: lstrip takes a character set, so "./.mcp.json"
    # would lose the dotfile's own leading dot and resolve to "mcp.json".
    check("the manifest declares the directory the skill is actually in",
          (built / m["skills"].removeprefix("./")).is_dir(), f"— {m['skills']}")
    check("the manifest declares the mcp file that was actually written",
          (built / m["mcpServers"].removeprefix("./")).is_file(), f"— {m['mcpServers']}")

    # ── The checked-in copy is current ──────────────────────────────
    print("\nplugins/ in the repo is up to date")

    if not CHECKED_IN.is_dir():
        check("the plugin is checked in", False, f"— missing {CHECKED_IN}")
    else:
        built_files = {p.relative_to(built): p.read_bytes()
                       for p in built.rglob("*") if p.is_file()}
        repo_files = {p.relative_to(CHECKED_IN): p.read_bytes()
                      for p in CHECKED_IN.rglob("*") if p.is_file()}
        check("the same set of files is checked in",
              set(built_files) == set(repo_files),
              f"— only in repo: {sorted(map(str, set(repo_files) - set(built_files)))}, "
              f"only generated: {sorted(map(str, set(built_files) - set(repo_files)))}")
        stale = sorted(str(k) for k in set(built_files) & set(repo_files)
                       if built_files[k] != repo_files[k])
        check("every checked-in file matches what the build produces", not stale,
              f"— stale: {stale}. Re-run `merge plugin build --out ./plugins`."
              if stale else "")

    # Rebuilding must be idempotent, or "is it current?" is unanswerable.
    again = pb.build(Path(tmp) / "plugins")["plugin"]
    check("rebuilding produces identical bytes",
          {p.relative_to(again): p.read_bytes() for p in again.rglob("*") if p.is_file()}
          == built_files)


# ── The marketplace ─────────────────────────────────────────────────
print("\nmarketplace.json")

entry = pb.marketplace_entry("./plugins/merge-cli")
check("source is declared local", entry["source"]["source"] == "local")
check("the path is relative and ./-prefixed",
      entry["source"]["path"].startswith("./"), f"— {entry['source']['path']}")
check("the plugin name matches the manifest", entry["name"] == m["name"])
check("category matches the manifest's", entry["category"] == m["interface"]["category"])
check("installation is a documented policy",
      entry["policy"]["installation"] in pb.INSTALLATION_POLICIES)
check("authentication is a documented policy",
      entry["policy"]["authentication"] in pb.AUTHENTICATION_POLICIES)
# Installing by default would register a server that cannot start for anyone
# who has not installed the Python package.
check("installation is AVAILABLE, not INSTALLED_BY_DEFAULT",
      entry["policy"]["installation"] == "AVAILABLE")
check("products is omitted (it gates the plugin; nothing here needs that)",
      "products" not in entry["policy"])

for bad in [{"installation": "SOMETIMES"}, {"authentication": "NEVER"}]:
    try:
        pb.marketplace_entry("./x", **bad)
        check(f"an invalid {list(bad)[0]} is rejected", False)
    except ValueError:
        check(f"an invalid {list(bad)[0]} is rejected", True)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    res = pb.build(root / "plugins")
    mk_path = root / pb.MARKETPLACE_REL
    check("build writes the marketplace file", mk_path.is_file(), f"— {mk_path}")

    data = json.loads(mk_path.read_text(encoding="utf-8"))
    check("it has the documented top-level shape",
          set(data) >= {"name", "interface", "plugins"} and "displayName" in data["interface"])
    check("our plugin is listed once",
          [p["name"] for p in data["plugins"]].count(pb.PLUGIN_NAME) == 1)
    listed = data["plugins"][0]["source"]["path"]
    check("the source path resolves to the plugin that was built",
          (root / listed).resolve() == res["plugin"].resolve(), f"— {listed}")

    # A marketplace can list other people's plugins and carry a name its owner
    # chose. Overwriting the file would drop both without saying so.
    data["name"] = "personal"
    data["interface"]["displayName"] = "Personal"
    data["plugins"].insert(0, {"name": "someone-elses", "source": {
        "source": "local", "path": "./plugins/someone-elses"}})
    mk_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    pb.build(root / "plugins")
    data = json.loads(mk_path.read_text(encoding="utf-8"))
    names = [p["name"] for p in data["plugins"]]
    check("rebuilding keeps another plugin's entry", "someone-elses" in names)
    check("and does not duplicate ours", names.count(pb.PLUGIN_NAME) == 1)
    check("the owner's marketplace name survives", data["name"] == "personal")
    check("so does their display name", data["interface"]["displayName"] == "Personal")
    check("a backup of the previous version is kept",
          any(p.name.startswith("marketplace.json.bak-") for p in mk_path.parent.iterdir()))

    # Refusing to touch a file that does not parse matters more than writing:
    # replacing it would discard every plugin the user had registered.
    mk_path.write_text("{ not json", encoding="utf-8")
    try:
        pb.build(root / "plugins")
        check("a malformed marketplace is refused, not replaced", False)
    except pb.ConfigError:
        check("a malformed marketplace is refused, not replaced", True)
    check("and the file is left as it was",
          mk_path.read_text(encoding="utf-8") == "{ not json")

    check("--no-marketplace skips it",
          pb.build(root / "plugins", marketplace=False)["marketplace"] is None)

print("\nthe CLI's policy choices match the module's")
cli_src = (REPO / "merge_cli" / "cli.py").read_text(encoding="utf-8")
for policy in pb.INSTALLATION_POLICIES + pb.AUTHENTICATION_POLICIES:
    check(f"cli.py offers {policy}", f'"{policy}"' in cli_src)

print("\nthe manifest is valid JSON on disk")
if (CHECKED_IN / ".codex-plugin" / "plugin.json").is_file():
    on_disk = json.loads((CHECKED_IN / ".codex-plugin" / "plugin.json")
                         .read_text(encoding="utf-8"))
    check("it parses and carries this version",
          on_disk.get("version") == merge_cli.__version__,
          f"— {on_disk.get('version')}")

print("\n" + "=" * 50)
print(f"FAIL {len(failures)}" + (f" -> {failures}" if failures else "  (all passed)"))
raise SystemExit(1 if failures else 0)
