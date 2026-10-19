"""Regression guard for answer_variant_pathogenicity and the bundled skill.

Why this exists
---------------
This tool is the one an agent reaches for without being told, so its failure
modes are the quiet kind: a coordinate parsed into the wrong locus, a score
reported without the caveat that says it is untrustworthy, a computed number
presented as if it were a ClinVar classification. None of those raise.

The parsing tests care most about what must be *refused*. Accepting an rsID or
an HGVS string by guessing at a coordinate would score a different variant and
answer confidently about it, which is worse than any error message.

Entirely offline: predict_variant is stubbed, no network.

Run:  ./.venv/bin/python tests/test_answer_variant.py
"""
import tempfile
from pathlib import Path

from merge_cli import mcp_install as mi, mcp_server as ms, skill_install as si

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


# ── Parsing ─────────────────────────────────────────────────────────
print("_parse_variant accepts the shapes people paste")

for text, want in [
    ("chr1:1040819 G>GC", ("chr1", 1040819, "G", "GC")),
    ("1-1040819-G-GC", ("chr1", 1040819, "G", "GC")),
    ("chr1 1040819 G GC", ("chr1", 1040819, "G", "GC")),
    ("chr17:43092919:G>A", ("chr17", 43092919, "G", "A")),
    ("CHR1:1,040,819 G/GC", ("chr1", 1040819, "G", "GC")),
    ("chrx:12345 a>t", ("chrX", 12345, "A", "T")),
    ("MT:100 T>C", ("chrM", 100, "T", "C")),
    ("  chr2:1000 A>-  ", ("chr2", 1000, "A", "-")),
]:
    got, why = ms._parse_variant(text)
    check(f"{text!r}", got == want, f"— {got or why}")

print("\n_parse_variant refuses rather than guesses")

got, why = ms._parse_variant("rs799917")
check("an rsID is refused", got is None and "rsid" in (why or "").lower())
check("and the refusal says how to fix it", "coordinates" in (why or "").lower())

for hgvs in ["NM_198576.4(AGRN):c.667dup", "ENST00000357654.9:c.2612C>T",
             "NP_009225.1:p.Pro871Leu", "g.43092919G>A"]:
    got, why = ms._parse_variant(hgvs)
    check(f"HGVS {hgvs[:28]!r} is refused", got is None and "hgvs" in (why or "").lower())

# chr23 is some pipelines' name for X. Reading it that way would silently score
# a different chromosome, so it has to be an error rather than an assumption.
got, why = ms._parse_variant("23:100 A>T")
check("chr23 is refused, not assumed to mean X",
      got is None and "does not exist" in (why or ""))
for bad in ["", "not a variant", "chr1:1040819 G>GC extra", "chr1:abc G>A"]:
    got, why = ms._parse_variant(bad)
    check(f"{bad[:26]!r} is refused", got is None and bool(why))


# ── The answer ──────────────────────────────────────────────────────
print("\nanswer_variant_pathogenicity")

CLINVAR_BENIGN = {
    "genename": "BRCA1;BRCA1", "rs_dbSNP": "rs799917",
    "MANE": ".;Select", "HGVSp_VEP": "p.Pro575Leu;p.Pro871Leu",
    "HGVSc_VEP": "c.1724C>T;c.2612C>T",
    "Ensembl_transcriptid": "ENST00000497488;ENST00000357654",
    "clinvar_clnsig": "Benign", "clinvar_review": "reviewed_by_expert_panel",
    "clinvar_id": "41812",
}


def stub_result(score=0.0079, label="Benign", missing=None,
                annotations=CLINVAR_BENIGN, vtype="coding", n_features=6):
    return {
        "success": True,
        "prediction": {"_variant_type": vtype,
                       "dbnsfp": {"annotations": dict(annotations)} if annotations else None},
        "ensemble": {"MERGE_Coding": {
            "score": score, "interpretation": {"label": label},
            "model_used": "MERGE_Coding", "n_features": n_features,
            "missing_features": missing or []}},
    }


def call(result, variant="chr17:43092919 G>A"):
    calls = []
    saved = ms.predict_variant

    def fake(chrom, pos, ref, alt, **kw):
        calls.append((chrom, pos, ref, alt, kw))
        return result(len(calls)) if callable(result) else result

    ms.predict_variant = fake
    try:
        return ms.answer_variant_pathogenicity(variant), calls
    finally:
        ms.predict_variant = saved


out, calls = call(stub_result())
check("the parsed coordinate is what gets scored",
      calls[0][:4] == ("chr17", 43092919, "G", "A"), f"— {calls[0][:4]}")
check("one call is enough (no separate type lookup)", len(calls) == 1)
check("score and label are surfaced",
      out["score"] == 0.0079 and out["label"] == "Benign")
check("the MANE Select transcript is the one named, not the first",
      out["hgvs_p"] == "p.Pro871Leu" and out["transcript"] == "ENST00000357654",
      f"— {out.get('hgvs_p')} / {out.get('transcript')}")
check("gene and rsid come through", out["gene"] == "BRCA1" and out["rsid"] == "rs799917")
check("the real ClinVar record is returned, not recalled",
      out["clinvar"]["significance"] == "Benign"
      and out["clinvar"]["review"] == "reviewed_by_expert_panel")
check("an expert-panel assertion is flagged as outranking the score",
      any("outrank" in c for c in out["caveats"]), f"— {out['caveats']}")
check("elapsed_seconds is reported", "elapsed_seconds" in out)

out, _ = call(stub_result(missing=["AlphaMissense_score", "evo2_7b_score"]))
check("imputed features produce a caveat",
      any("median-imputed" in c for c in out["caveats"]))
check("the caveat says imputation biases toward pathogenic",
      any("pathogenic side" in c for c in out["caveats"]))
check("missing protein evidence is called out specifically",
      any("unreliable" in c for c in out["caveats"]), f"— {out['caveats']}")

# The case that matters most: a confident score that contradicts a curated
# assertion must not be reported as though it agreed.
out, _ = call(stub_result(score=0.93, label="Pathogenic"))
check("MERGE/ClinVar disagreement is surfaced",
      any("disagree" in c for c in out["caveats"]), f"— {out['caveats']}")

out, _ = call(stub_result(annotations=None, vtype="nonsnv"))
check("a variant with no dbNSFP record still scores",
      out["success"] is True and out["score"] == 0.0079)
check("and simply carries no clinvar block", "clinvar" not in out)

print("\nfailure handling")

out, calls = call({"success": False,
                   "error": "reference mismatch against hg38 at this position"})
check("a reference mismatch is passed through as a refusal", out["success"] is False)
check("and is NOT retried — it is a verdict, not a hiccup", len(calls) == 1)

out, calls = call(lambda n: ({"success": False, "error": "transport blew up"}
                             if n == 1 else stub_result()))
check("a transient failure is retried once", len(calls) == 2)
check("the retry stops waiting on evo2", calls[1][4].get("evo2_timeout") == 0.0)
check("and the retry's answer is returned", out["success"] is True)
check("with a caveat that it was degraded",
      any("retry" in c.lower() for c in out["caveats"]), f"— {out['caveats']}")


# ── The skill ───────────────────────────────────────────────────────
print("\nbundled skill")

bundled = si.bundled_skill_dir() / "SKILL.md"
check("the skill ships inside the package", bundled.is_file(), f"— {bundled}")
text = bundled.read_text(encoding="utf-8")
check("it has YAML frontmatter", text.startswith("---\n"))
check("its name matches the directory", f"name: {si.SKILL_NAME}" in text)
# The description is the only thing a deferring client sees before deciding
# whether to look for the tools, so the trigger words have to be in it.
head = text.split("---")[1]
for word in ["pathogenic", "benign", "VUS", "variant effect", "致病性", "ClinVar"]:
    check(f"the description can be matched on {word!r}", word in head)
check("it names the one-shot tool", "answer_variant_pathogenicity" in text)
check("it tells the model to search for the tools first", "tool_search" in text)
check("it forbids answering from memory", "from memory" in text)

with tempfile.TemporaryDirectory() as tmp:
    saved = (si.CLAUDE_SKILLS_DIR, si.CODEX_SKILLS_DIR)
    si.CLAUDE_SKILLS_DIR = Path(tmp) / ".claude" / "skills"
    si.CODEX_SKILLS_DIR = Path(tmp) / ".agents" / "skills"
    try:
        # Codex reads .agents/skills directly — no plugin, no marketplace. This
        # is the path that was missing when a registered server still went
        # unused, so the exact directory matters.
        check("codex installs into .agents/skills",
              si.skills_dir("codex").parts[-2:] == (".agents", "skills"),
              f"— {si.skills_dir('codex')}")
        check("claude installs into .claude/skills",
              si.skills_dir("claude").parts[-2:] == (".claude", "skills"))
        check("codex project scope is .agents/skills under the cwd",
              si.skills_dir("codex", "project") == Path.cwd() / ".agents" / "skills")
        try:
            si.skills_dir("emacs")
            check("an unknown client raises", False)
        except ValueError:
            check("an unknown client raises ValueError", True)

        codex_res = si.install("codex")
        check("the codex skill lands where Codex looks",
              (codex_res["path"] / "SKILL.md").is_file(), f"— {codex_res['path']}")
        check("and is reported current", si.is_current("codex") is True)
        check("installing for codex does not touch claude",
              si.is_current("claude") is False)

        check("nothing is installed yet", si.is_current() is False)
        res = si.install()
        check("install writes the skill", (res["path"] / "SKILL.md").is_file())
        check("and reports it as new", res["replaced"] is False)
        check("status sees it as current", si.is_current() is True)
        clients = {t["client"] for t in si.status()["targets"]}
        check("status covers both clients", clients == set(si.CLIENTS), f"— {clients}")

        # The skill is an aid to the MCP server, not a replacement: it directs
        # the model at tools that only exist once the server is registered.
        saved_status = mi.status
        try:
            mi.status = lambda: {"clients": [
                {"client": "codex", "registered": True},
                {"client": "claude-code", "registered": False},
                {"client": "claude-desktop", "registered": False}]}
            check("an unregistered server is reported for claude",
                  si.mcp_registered("claude") is False)
            check("a registered one is not warned about",
                  si.mcp_registered("codex") is True)
            mi.status = lambda: (_ for _ in ()).throw(RuntimeError("unreadable"))
            check("an unreadable config warns about nothing rather than wrongly",
                  si.mcp_registered("claude") is True)
        finally:
            mi.status = saved_status

        # A stale copy from an older merge-cli must be replaced, not merged with.
        (res["path"] / "SKILL.md").write_text("outdated", encoding="utf-8")
        check("drift is detected", si.is_current() is False)
        check("reinstalling reports the replacement", si.install()["replaced"] is True)
        check("and restores the current text", si.is_current() is True)

        # Other people's skills are not ours to touch.
        other = si.CLAUDE_SKILLS_DIR / "someone-elses-skill"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text("keep me", encoding="utf-8")
        si.install()
        check("an unrelated skill survives install",
              (other / "SKILL.md").read_text() == "keep me")
        check("uninstall removes ours", si.uninstall()["removed"] is True)
        check("and leaves theirs", (other / "SKILL.md").is_file())
        check("uninstalling again is a no-op", si.uninstall()["removed"] is False)
    finally:
        si.CLAUDE_SKILLS_DIR, si.CODEX_SKILLS_DIR = saved

print("\n" + "=" * 50)
print(f"FAIL {len(failures)}" + (f" -> {failures}" if failures else "  (all passed)"))
raise SystemExit(1 if failures else 0)
