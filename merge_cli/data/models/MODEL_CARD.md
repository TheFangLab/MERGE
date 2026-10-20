# MERGE models

The same files ship with merge-cli (`merge_cli/data/models/`) and serve the web server
(https://merge.fanglab.cn). Source: MERGE-benchmark (`05_results/trained_models/`,
`05_results_alphagenome_average/`).

| File | What it is |
|---|---|
| `MERGE_noAlphaGenome_<task>.pkl` | The MERGE model for one variant class. Its candidate pool excluded every `alphagenome_*` column, so no AlphaGenome output was used to train it. |
| `alphagenome_reference_quantiles.json` | The AlphaGenome half of the MERGE + AlphaGenome average: every distinct AlphaGenome score among the benchmark variants with its mid-rank percentile. Not a model: no labels, nothing fitted. Splicing and noncoding only. |
| `merge_final_thresholds.json` | Interpretation cut-points for the average (the file name is the one the tools load). |

```python
import json, joblib, numpy as np, pandas as pd
m   = joblib.load('MERGE_noAlphaGenome_splicing.pkl')
ref = json.load(open('alphagenome_reference_quantiles.json'))['tasks']['splicing']
df  = pd.read_csv('your_variants_scored.csv')      # m['features'] and ref['feature']

score = m['pipeline'].predict_proba(df[m['features']].to_numpy())[:, 1]
pct   = np.interp(df[ref['feature']], ref['scores'], ref['percentiles'])
merge_alphagenome_average = 0.5 * score + 0.5 * pct   # undefined where the AlphaGenome value is missing
```

## The two scores

`score` is the MERGE probability, returned for all four variant classes.

    merge_alphagenome_average = 0.5 x score + 0.5 x percentile(AlphaGenome column)

a fixed 1:1 mean. The AlphaGenome column is fixed by variant class:
`alphagenome_splicing` for splicing and `alphagenome_raw_score_max` for noncoding
variants. On coding and non-SNV variants AlphaGenome's summaries are near chance
(best held-out AUROC 0.571 and 0.512), so the average is not offered there
(`merge_alphagenome_average: null`, status `not_offered_for_this_variant_type`).
It is also null when AlphaGenome has no value for the variant; the AlphaGenome value
is never imputed at prediction time, and the average is never replaced by the MERGE
score.

## Models

Refitted on every variant of the benchmark table, with the feature-selection strategy
chosen in the chromosome-held-out benchmark. These are deployment models: quote the
benchmark, not their fit. Output is the probability of the positive class (pathogenic,
or splice-altering for splicing). Median imputation and standardization are inside each
pipeline, so missing features are filled with training medians.

| Task | Feature selection | Algorithm | Features | n_train | Positive rate |
|---|---|---|---:|---:|---:|
| coding | beam search | LR_L2 | 6 | 4,289 | 0.6789 |
| splicing | LASSO | CatBoost | 12 | 35,015 | 0.6354 |
| noncoding | beam search | MLP | 6 | 86,829 | 0.0154 |
| nonsnv | LASSO | XGBoost | 12 | 36,862 | 0.8144 |

Panels:

- **coding**: AlphaMissense_score, ESM1b_score, carbon_score, enformer_top5_absmean, enformer_std, heyenadna_score
- **splicing**: carbon_score, enformer_std, enformer_max_abs, GPN_MSA_score, enformer_absmean, enformer_neg_sum, generator_score, enformer_top5_absmean, enformer_mean, evo2_7b_base_score, enformer_pos_sum, evo2_1b_base_score
- **noncoding**: evo2_7b_score, carbon_score, enformer_absmean, GPN_MSA_score, evo2_1b_base_score, NT_score
- **nonsnv**: evo2_7b_score, carbon_score, enformer_std, enformer_top5_absmean, generator_score, evo2_1b_base_score, enformer_absmean, generator_v2_score, evo2_7b_base_score, enformer_neg_sum, hyenadna_score, enformer_pos_sum

GENERATOR scores are float32, non-SNV HyenaDNA scores are computed one variant at a
time in float32, and the seven Enformer summaries cover all 5,313 human tracks (see
MERGE-benchmark `02_feature_extraction/README.md`); compute them the same way when
scoring new variants.

## Interpretation bands

MERGE score: ≥ 0.8 Pathogenic, ≥ 0.6 Likely Pathogenic, ≥ 0.4 Uncertain Significance,
≥ 0.2 Likely Benign, otherwise Benign.

The average has its own cut-points, matched to the MERGE bands by the fraction of
benchmark variants (with an AlphaGenome score) in each band; no labels are used.

| Task | Pathogenic | Likely Pathogenic | Uncertain Significance | Likely Benign | Agreement with the MERGE band |
|---|---:|---:|---:|---:|---:|
| splicing | ≥ 0.6727 | ≥ 0.5339 | ≥ 0.3926 | ≥ 0.2523 | 0.812 |
| noncoding | ≥ 0.8150 | ≥ 0.6827 | ≥ 0.5764 | ≥ 0.4975 | 0.876 |

The labels name score ranges. They are not ACMG/AMP classifications or calibrated
probabilities, and the models are research tools, not medical devices.

## Benchmark performance

Chromosome-held-out AUROC / AUPRC, mean over three folds (MERGE-benchmark, manuscript
Table 2):

| Task | n | MERGE | MERGE + AlphaGenome average | AlphaGenome |
|---|---:|---:|---:|---:|
| coding | 4,289 | 0.9519 / 0.9729 | not offered | — |
| splicing | 35,015 | 0.9269 / 0.9561 | 0.9567 / 0.9740 | 0.9563 / 0.9710 |
| noncoding | 86,829 | 0.9472 / 0.6841 | 0.9781 / 0.7702 | 0.9774 / 0.7953 (raw maximum) |
| nonsnv | 36,862 | 0.9820 / 0.9943 | not offered | — |

In the benchmark, missing AlphaGenome scores were set to the training-fold median.

## Licence

The four `MERGE_noAlphaGenome_*.pkl` models are MIT. `alphagenome_reference_quantiles.json`
is CC BY-NC 4.0: it summarizes AlphaGenome output, whose API is offered for
non-commercial use. Without it MERGE runs unchanged and `merge_alphagenome_average`
is null. Several features come from resources with their own terms (dbNSFP for
AlphaMissense and ESM-1b, Nucleotide Transformer, AlphaGenome); see LICENSE and
THIRD_PARTY_NOTICES.md of merge-cli.

## Provenance

| Model | Training table | SHA-256 of the table |
|---|---|---|
| coding | `coding_features.csv` | `6b10de865b13190c…` |
| splicing | `splicing_features.csv` | `a737bf76a8954abc…` |
| noncoding | `noncoding_features.csv` | `34cc0fa06fb0449b…` |
| nonsnv | `non_snv_features.csv` at benchmark commit `3cc3f8d` | `ecbf4a499a5546ac…` |

The non-SNV bundle records the original training-table hash above. The current
benchmark table has SHA-256 `51b91ea7229d801f…`: its six `alphagenome_*` columns
were updated after the model was fitted. The identifiers, labels and every
non-AlphaGenome feature are unchanged; these updated columns are excluded from
the MERGE candidate pool and do not affect this bundle's inputs or predictions.

Fitted with scikit-learn 1.9.0 and CatBoost 1.2.10 (full software versions are stored in
each file under `software_versions`).
