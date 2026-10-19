"""Install the bundled MERGE skill into an agent's skills directory.

A skill is what makes the MCP tools findable. Clients that defer tool loading —
Codex does, to keep the context small — do not put every server's tools in front
of the model, so a question like "is this variant pathogenic?" can be answered
from memory while merge-cli sits unused one search away. Skill names and
descriptions stay visible, so the skill is what tells the model to go looking.

The file itself lives in the package (merge_cli/data/skills/), so it ships and
versions with the tools it describes; this module only copies it out.
"""
import importlib.resources as resources
import shutil
from pathlib import Path

SKILL_NAME = "merge-variant-pathogenicity"

CLIENTS = ("claude", "codex")

# Claude Code and Claude Desktop read personal skills from here.
CLAUDE_SKILLS_DIR = Path.home() / ".claude" / "skills"
# Codex reads them from .agents/skills, searching the working directory, the
# repo root, then $HOME. It needs no plugin and no marketplace for this: a
# folder in the right place is the whole mechanism, and it picks up changes on
# its own (restart if one does not appear).
CODEX_SKILLS_DIR = Path.home() / ".agents" / "skills"

# Per client: (user-level directory, the project-level directory's parts).
_LAYOUT = {
    "claude": (lambda: CLAUDE_SKILLS_DIR, (".claude", "skills")),
    "codex": (lambda: CODEX_SKILLS_DIR, (".agents", "skills")),
}


def bundled_skill_dir() -> Path:
    """Where the packaged skill lives inside this install."""
    return Path(str(resources.files("merge_cli") / "data" / "skills" / SKILL_NAME))


# Which MCP client configs correspond to a skill target. The skill tells a model
# to reach for tools; those tools only exist if the server is registered in the
# same agent's config.
_MCP_CLIENTS = {
    "claude": ("claude-code", "claude-desktop"),
    "codex": ("codex",),
}


def mcp_registered(client: str) -> bool:
    """Whether the MCP server is registered for the agent this skill targets.

    A skill without it is inert in the worst way: it directs the model to tools
    that do not exist, the search comes back empty, and the model quietly falls
    back to answering from memory. Nothing errors, and the symptom is
    indistinguishable from the model simply choosing not to use the tool.
    """
    from . import mcp_install as mi

    wanted = _MCP_CLIENTS.get(client, ())
    try:
        return any(info.get("registered") for info in mi.status()["clients"]
                   if info["client"] in wanted)
    except Exception:
        return True  # can't tell — say nothing rather than warn wrongly


def client_installed(client: str) -> bool:
    """Whether the client looks present, judged by its home directory.

    Claude has no single marker directory worth testing, so it is always
    treated as available; Codex keeps $CODEX_HOME at ~/.codex.
    """
    if client == "codex":
        return (Path.home() / ".codex").is_dir()
    return True


def skills_dir(client: str = "claude", scope: str = "user") -> Path:
    try:
        user_dir, project_parts = _LAYOUT[client]
    except KeyError:
        raise ValueError(f"unknown client: {client}") from None
    return Path.cwd().joinpath(*project_parts) if scope == "project" else user_dir()


def target_dir(client: str = "claude", scope: str = "user") -> Path:
    return skills_dir(client, scope) / SKILL_NAME


def installed_version(client: str = "claude", scope: str = "user"):
    """The installed copy's text, or None. Used to report drift from the bundled one."""
    f = target_dir(client, scope) / "SKILL.md"
    try:
        return f.read_text(encoding="utf-8")
    except OSError:
        return None


def is_current(client: str = "claude", scope: str = "user") -> bool:
    installed = installed_version(client, scope)
    if installed is None:
        return False
    return installed == (bundled_skill_dir() / "SKILL.md").read_text(encoding="utf-8")


def install(client: str = "claude", scope: str = "user") -> dict:
    """Copy the bundled skill into place, replacing an older copy of itself.

    Only ever touches this one skill's directory — everything else the user has
    under skills/ is left alone.
    """
    src, dest = bundled_skill_dir(), target_dir(client, scope)
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"the bundled skill is missing from this install ({src}). "
            "Reinstall merge-cli.")
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if existed:
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return {"path": dest, "replaced": existed}


def uninstall(client: str = "claude", scope: str = "user") -> dict:
    dest = target_dir(client, scope)
    if not dest.exists():
        return {"path": dest, "removed": False}
    shutil.rmtree(dest)
    return {"path": dest, "removed": True}


def status() -> dict:
    report = {"skill": SKILL_NAME, "bundled": bundled_skill_dir(), "targets": []}
    for client in CLIENTS:
        for scope in ("user", "project"):
            dest = target_dir(client, scope)
            report["targets"].append({
                "client": client, "scope": scope, "path": dest,
                "detected": client_installed(client),
                "installed": (dest / "SKILL.md").is_file(),
                "current": is_current(client, scope),
            })
    return report
