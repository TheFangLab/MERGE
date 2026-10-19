---
name: merge-variant-pathogenicity
description: >-
  Determine whether a genomic variant is pathogenic, deleterious, damaging,
  disease-causing, benign, likely pathogenic, likely benign, or a VUS, using the
  MERGE ensemble. Use whenever a question involves variant effect prediction,
  variant interpretation, variant classification, clinical significance, or
  ClinVar-style assessment — for genomic coordinates (chr1:1040819 G>GC,
  1-1040819-G-GC), SNVs, indels, non-SNVs, delins, VCF rows, HGVS or rsIDs.
  Also covers 变异致病性, 致病性预测, 变异解读, 这个变异有害吗, 良性还是致病.
  Trigger this before searching the web or answering from memory.
---

# Variant pathogenicity with MERGE

MERGE scores a variant by combining 14 sequence and protein models into one
ensemble probability, using a different feature panel for each of four variant
classes (coding, splice, noncoding, non-SNV). It is served through the
`merge-cli` MCP server (registered as `merge_cli` in Codex).

## The rule

**Never answer a pathogenicity question from memory or from a web search alone.**
Recalled ClinVar classifications are frequently stale or wrong, and they cannot
cover variants that have never been submitted. Call the tool. It returns both a
freshly computed score and the real ClinVar record when one exists, so the two
can be compared rather than confused.

## How to call it

If the MCP tools are not already visible in this session, search for them first
(`tool_search` or the client's equivalent) with a query like
`merge_cli variant pathogenicity`.

Then, for essentially every question, one call is enough:

```
answer_variant_pathogenicity(variant="chr1:1040819 G>GC", genome="hg38")
```

It parses the variant, picks the right panel, handles slow or unavailable
models, and returns an interpreted answer. Do **not** call `get_variant_type`
first — it is a pre-check, not a prerequisite.

Reach for the lower-level tools only when you specifically need what they add:

| Tool | Use it when |
|---|---|
| `answer_variant_pathogenicity` | Default. "Is this variant pathogenic?" |
| `predict_variant` | You need per-model raw scores, a forced `ensemble_type`, SHAP, or custom timeouts |
| `get_variant_type` | You only want the class (coding/splice/noncoding/nonsnv), not a score |
| `submit_batch_vcf` + `check_batch_status` | More than a handful of variants — a VCF, not a loop |
| `check_health` | Calls keep failing and you need to tell a server problem from a broken install |
| `list_models` | The question is "which models does MERGE use?" |

## Input rules

- Genomic coordinates only: `chr1:1040819 G>GC`, `1-1040819-G-GC`,
  `chr1 1040819 G GC`. Indels use ANNOVAR-style alleles, `-` for an empty side.
- **rsIDs and HGVS are refused on purpose.** Resolving them needs a transcript
  database the server does not have, and a guessed coordinate would silently
  score the wrong locus. If the user gives `rs799917` or
  `NM_198576.4(AGRN):c.667dup`, convert it first (VariantValidator, Mutalyzer,
  Ensembl VEP, dbSNP) and confirm the coordinate with the user before scoring.
- Default assembly is hg38. If the user's source is hg19/GRCh37, pass
  `genome="hg19"` — do not silently score hg19 coordinates against hg38.

## Reading the answer

```json
{"success": true, "variant": "chr17:43092919 G>A", "variant_type": "coding",
 "score": 0.0207, "label": "Benign",
 "merge_alphagenome_average": null,
 "merge_alphagenome_average_unavailable_reason":
   "the MERGE + AlphaGenome average is only offered for splicing and non-coding variants; ...",
 "gene": "BRCA1", "hgvs_p": "p.Pro871Leu", "transcript": "ENST00000357654",
 "rsid": "rs799917",
 "clinvar": {"significance": "Benign", "review": "reviewed_by_expert_panel"},
 "caveats": ["..."]}
```

When you report this:

1. **Always state the `caveats`.** They are not boilerplate — they say when the
   score is untrustworthy. A caveat about median-imputed features means real
   evidence was missing, and imputation biases the score toward pathogenic. A
   score with missing AlphaMissense/ESM-1b evidence should not be presented as a
   conclusion.
2. **Keep the two sources distinct.** Say which number is the MERGE prediction
   and which is the ClinVar assertion. Never present a computed score as if it
   were a database classification.
3. **ClinVar reviewed by an expert panel or a practice guideline outranks the
   score.** Lead with it; use MERGE as supporting evidence.
4. **If they disagree, say so.** Do not quietly pick the one that sounds better.
5. `score` is a probability from 0 to 1. Higher means more likely pathogenic.
6. **`score` is always there; a second number sometimes is, and they are not on
   the same scale.** `score` is the MERGE ensemble, which uses no AlphaGenome
   feature, and it comes back for every variant class. On **splicing and
   non-coding** variants only, `merge_alphagenome_average` is that score
   averaged 1:1 with the percentile of one AlphaGenome column. Lead with
   `score`. Quote the average with `merge_alphagenome_average_label`, never by
   comparing the number to the ensemble's 0.8/0.6/0.4/0.2 bands — it has its own
   per-task cut-points and 0.65 means different things on the two scales. When
   it is null, `merge_alphagenome_average_unavailable_reason` says why; report
   the ensemble score alone rather than presenting one as the other.
7. **On coding and non-SNV variants there is no second score, and that is by
   design — not a failure to report.** The best single AlphaGenome feature
   scores about 0.54–0.57 AUROC on held-out data for those classes, so the
   average is not offered: the status reads
   `not_offered_for_this_variant_type`. Do not describe it as missing, degraded,
   or worth retrying — say the MERGE ensemble score is the answer for that
   variant class. (AlphaGenome itself is still run and still reported in
   `prediction["alphagenome"]`.)

## When something goes wrong

- `success: false` with a **reference mismatch** — the ref allele does not match
  that position in that assembly. The error names the real base. This is a
  refusal, not a failure: the variant as written does not exist. Fix the allele
  or the assembly and retry; never report a score for it.
- **Slow calls.** The tool bounds its own runtime and degrades rather than
  hanging, so a timeout means the client cut it off, not the server. Retry with
  a smaller `time_budget`, or use `predict_variant(..., evo2_timeout=0)` to skip
  the slowest optional model — and then say that Evo2 was imputed.
- **Many variants.** Use `submit_batch_vcf`. Looping
  `answer_variant_pathogenicity` costs ~20s per variant.

## Interpreting responsibly

MERGE is a research prediction tool. It is **not a diagnostic device**, and a
score is not a clinical classification. It must not be used for diagnosis,
treatment decisions or genetic counselling. Zygosity, inheritance mode,
phenotype fit, segregation and population frequency all matter and none of them
are in the score. Present results as computational evidence, and say so.

If a result is missing its AlphaGenome features, say so rather than presenting
the score as complete: AlphaGenome needs the user's own API key, and without one
only variants already in the MERGE precomputed cache carry those features. Tell
them they can add a key with `merge alphagenome configure --api-key ...`
(free, non-commercial use only, from Google DeepMind).
