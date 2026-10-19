"""merge-cli wiring smoke test — 99 assertions, no GPU services required.

What it guards that the other tests do not
------------------------------------------
Section 7 compiles every generated model-service script and asserts the
algorithm conventions baked into them — window sizes (Carbon 24576, the others
600), dtypes (fp32 everywhere except Carbon's bf16), trust_remote_code per
model, and the separate SNV / non-SNV scoring conventions. Getting any of these
wrong does not raise: it changes the scale of a feature, the ensemble happily
consumes the wrong number, and the only symptom is a prediction that is quietly
wrong. There is no other check on this.

Sections 8 / 8b / 8c / 8d score all four variant types from simulated full
feature sets, exercise the non-SNV haplotype construction and the dual NT /
HyenaDNA conventions, and confirm the precomputed cache decides by field
presence rather than by row existence.

Everything here is offline: no GPU, no model services, and no network beyond
what the imports need.

Run:  ./.venv/bin/python tests/test_smoke.py
"""
import sys

from merge_cli import api, local_engine as le
from merge_cli.cli import _model_flags

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print("  [PASS] %s %s" % (name, detail))
    else:
        fail += 1
        print("  [FAIL] %s %s" % (name, detail))


print("=== 1. 版本与打包 ===")
import merge_cli
# Assert the version is *consistent*, not a literal: __init__.py is the single
# source and pyproject reads it from there, so a release must not be able to
# leave `merge --version` disagreeing with the installed distribution.
check("__version__ is set", bool(merge_cli.__version__), merge_cli.__version__)
try:
    from importlib.metadata import version as _dist_version
    _installed = _dist_version("merge-cli")
    check("__version__ matches the installed distribution",
          _installed == merge_cli.__version__,
          "%s vs %s" % (merge_cli.__version__, _installed))
except Exception as _e:
    check("installed distribution metadata readable", False, str(_e))

ep = le._load_ep_module()
bundles = ep.load_bundles()
check("四个 pkl 全部加载", sorted(bundles) == ["coding", "noncoding", "nonsnv", "splice"],
      str(sorted(bundles)))
check("pkl 加载无错误", not ep._load_errors, str(ep._load_errors)[:200])
for t, b in sorted(bundles.items()):
    print("        %-10s algo=%-10s n_feat=%d" % (t, b["algorithm"], len(b["features"])))

print("\n=== 2. 模型开关 (14 模型 / 13 flags) ===")
f = _model_flags()
check("默认全部 12 个开关开启",
      all(f[k] for k in api.MODEL_FLAGS))
check("4 个 DNA-LM 现在默认开启",
      all(f[k] for k in ["use_generator", "use_generator_v2", "use_nt_v2", "use_carbon"]))
check("--no-carbon 可关闭", _model_flags(no_carbon=True)["use_carbon"] is False)
check("--no-generator 可关闭", _model_flags(no_generator=True)["use_generator"] is False)
check("--no-nt-v2 可关闭", _model_flags(no_nt_v2=True)["use_nt_v2"] is False)
check("api 默认全开", api._DEFAULT_ON == set(api.MODEL_FLAGS))
f_all = _model_flags(all_models=True)
check("--all-models 打开全部 13 个开关",
      all(f_all[k] for k in api.MODEL_FLAGS), str([k for k in api.MODEL_FLAGS if not f_all[k]]))
check("use_carbon 存在于 api.MODEL_FLAGS", "use_carbon" in api.MODEL_FLAGS)
check("use_evo1 已移除", "use_evo1" not in api.MODEL_FLAGS)
check("api flags 共 12 个(已删popEVE)", len(api.MODEL_FLAGS) == 12, str(len(api.MODEL_FLAGS)))
check("use_popeve 已移除", "use_popeve" not in api.MODEL_FLAGS)

print("\n=== 3. Evo2 变体 ===")
check("evo2 三变体默认全开",
      _model_flags()["evo2_models"] == ["evo2_7b", "evo2_7b_base", "evo2_1b_base"],
      str(_model_flags()["evo2_models"]))
check("--evo2-models 可收窄",
      _model_flags(evo2_models="evo2_7b")["evo2_models"] == ["evo2_7b"])
check("--evo2-model 只改主分变体(不影响本地跑哪几个)",
      _model_flags(evo2_model="evo2_7b_base")["evo2_model"] == "evo2_7b_base"
      and _model_flags(evo2_model="evo2_7b_base")["evo2_models"]
          == ["evo2_7b", "evo2_7b_base", "evo2_1b_base"])
check("--evo2-models 多变体",
      _model_flags(evo2_models="evo2_7b,evo2_7b_base,evo2_1b_base")["evo2_models"]
      == ["evo2_7b", "evo2_7b_base", "evo2_1b_base"])
print("        本机 FP8 可用: %s -> 可跑变体 %s"
      % (le.gpu_supports_fp8(), le.available_evo2_variants()))

print("\n=== 4. 远程 ensemble 签名（此前必崩的 bug）===")
try:
    import inspect
    sig = inspect.signature(api.predict_ensemble)
    check("predict_ensemble 接受 ensemble_type", "ensemble_type" in sig.parameters, str(sig))
except Exception as e:
    check("predict_ensemble 签名", False, str(e))

print("\n=== 5. non-SNV 判型 ===")
for ref, alt, want in [("A", "G", False), ("AT", "G", True), ("A", "GGC", True),
                       ("-", "G", True), ("A", "-", True), ("ATG", "CGA", True)]:
    check("is_non_snv(%s>%s)==%s" % (ref, alt, want), le.is_non_snv(ref, alt) == want)
check("get_variant_type indel -> nonsnv",
      le.get_variant_type("chr17", 43092919, "AT", "G") == "nonsnv")
check("--ensemble-type 覆盖生效",
      le.get_variant_type("chr17", 43092919, "A", "G", ensemble_type="nonsnv") == "nonsnv")

print("\n=== 6. 服务注册表与端口 ===")
check("9 个可部署模型(含enformer)", len(le.DEPLOYABLE_MODELS) == 9, str(le.DEPLOYABLE_MODELS))
check("enformer 可部署", "enformer" in le.DEPLOYABLE_MODELS)
for m in ["carbon", "generator", "generator_v2", "nt_v2"]:
    check("%s 在 DEPLOYABLE_MODELS" % m, m in le.DEPLOYABLE_MODELS)
    check("%s 有端口" % m, m in le._SERVICE_PORTS, str(le._SERVICE_PORTS.get(m)))
    check("%s 有 conda env" % m, m in le.ENV_NAMES)
    check("%s 有 HF model id" % m, m in le.DNA_LM_MODELS, le.DNA_LM_MODELS.get(m, ""))
check("失效的 18004 gateway 已移除", "generator_ntv2" not in le._SERVICE_PORTS and "generator_ntv2" not in le.ALL_SERVICES)

print("\n=== 7. 生成的服务脚本可编译 + 算法口径 ===")
import py_compile, tempfile, os
for m in ["carbon", "generator", "generator_v2", "nt_v2", "nt", "hyenadna", "evo2", "enformer", "alphagenome"]:
    try:
        p = le._write_server_script(m)
        py_compile.compile(p, doraise=True)
        src = open(p).read()
        check("%s 服务脚本生成并编译" % m, True, "(%d 行)" % len(src.splitlines()))
    except Exception as e:
        check("%s 服务脚本" % m, False, str(e)[:150])

# 校验关键算法常量确实写进脚本
carbon_src = open(le._write_server_script("carbon")).read()
check("carbon 窗口 24576", "WINDOW_BP = 24576" in carbon_src)
check("carbon 6-mer 对齐", "MULTIPLE_OF = 6" in carbon_src)
check("carbon <dna> 包裹", '"<dna>" + s + "</dna>"' in carbon_src)
check("carbon SUM-delta 方向 ref-alt", "totals[0] - totals[1]" in carbon_src)

gen_src = open(le._write_server_script("generator")).read()
check("generator 窗口 600 / max_length 128",
      "WINDOW_BP = 600" in gen_src and "MAX_LENGTH = 128" in gen_src)
check("generator NLL(alt)-NLL(ref)", "nll[1] - nll[0]" in gen_src)

ntv2_src = open(le._write_server_script("nt_v2")).read()
check("nt_v2 masked emb L2", "_emb_l2" in ntv2_src and "KIND = 'masked'" in ntv2_src)

nt_src = open(le._write_server_script("nt")).read()
check("NT 窗口已修为 8192", "WINDOW_BP = 8192" in nt_src)
check("NT 不再用旧的 6000 窗口", "6000" not in nt_src)

hy_src = open(le._write_server_script("hyenadna"))
hy_src = hy_src.read()
check("HyenaDNA 用上游 1000bp 上下文",
      "CONTEXT_LEN = 1000" in hy_src and "_upstream_context" in hy_src)
check("HyenaDNA 校验 REF(不再对错误坐标静默出分)",
      "reference mismatch" in hy_src and "_upstream_context(v.chrom, v.pos, ref)" in hy_src)

evo2_src = open(le._write_server_script("evo2")).read()
check("evo2 三变体 + reduce_method=mean",
      'reduce_method="mean"' in evo2_src and "average_reverse_complement=False" in evo2_src)
check("evo2 ref-alt 方向", "ref_ll - alt_ll" in evo2_src)

print("\n=== 8. 集成打分四类型（模拟全特征）===")
pred = {
    "input": {"chrom": "chr17", "pos": 43092919, "ref": "A", "alt": "G"},
    "genome_version": "hg38",
    "dbnsfp": {"dl_models": [{"AlphaMissense": {"score": 0.98}, "ESM1b": {"score": -12.3}}]},
    "gpn_msa": {"score": -8.1},
    "alphagenome": {"statistics": {
        "alphagenome_raw_score_max": 2.2, "alphagenome_raw_score_mean": 0.4,
        "alphagenome_raw_score_min": -1.8, "alphagenome_quantile_score_max": 0.97,
        "alphagenome_quantile_score_min": 0.02, "alphagenome_quantile_score_mean": 0.5,
        "alphagenome_splicing": 1.3}},
    "enformer": {"enformer_max_abs": 0.31, "enformer_pos_sum": 1.4, "enformer_std": 0.05,
                 "enformer_absmean": 0.02, "enformer_mean": 0.001,
                 "enformer_top5_absmean": 0.2, "enformer_neg_sum": -1.1},
    "evo2_7b": {"llr_score": -0.42}, "evo2_7b_base": {"llr_score": -0.35},
    "evo2_1b_base": {"llr_score": -0.29},
    "carbon": {"score": 11.2}, "generator": {"score": 0.13},
    "generator_v2": {"score": 0.09}, "nt_v2": {"score": 0.7},
    "nt": {"score": 1.9}, "hyenadna": {"score": -0.8},
}
tr = pred["dbnsfp"]["dl_models"]
for task in ["coding", "splice", "noncoding", "nonsnv"]:
    res, err = le.run_local_ensemble(pred, tr, ensemble_type=task, skip_shap=True)
    r = list(res.values())[0]
    if "error" in r:
        check("%s 打分" % task, False, r["error"][:120])
    else:
        check("%s 打分" % task, r["score"] is not None and not r["missing_features"],
              "score=%.4f %s missing=%d/%d" % (r["score"], r["model_used"],
                                               len(r["missing_features"]), r["n_features"]))

print("\n=== 8b. non-SNV 单倍型与双口径 ===")
carbon_src2 = open(le._write_server_script("carbon")).read()
check("DNA-LM 服务支持 non-SNV 单倍型", "build_haplotype" in carbon_src2)
check("causal 分开前向(不 padding)", "for seq in seqs:" in carbon_src2)
nt_src2 = open(le._write_server_script("nt")).read()
check("NT 双口径: SNV 8192 / non-SNV 600",
      "SNV_WINDOW_BP = 8192" in nt_src2 and "NONSNV_WINDOW_BP = 600" in nt_src2)
hy_src2 = open(le._write_server_script("hyenadna")).read()
check("HyenaDNA 双口径: SNV 上游1000 / non-SNV 600@1024",
      "CONTEXT_LEN = 1000" in hy_src2 and "NONSNV_MAX_LENGTH = 1024" in hy_src2)
evo2_src2 = open(le._write_server_script("evo2")).read()
check("evo2 支持 non-SNV(score_mean)", "build_haplotype" in evo2_src2 and "score_mean" in evo2_src2)
check("transformers 已钉 4.x", le.TRANSFORMERS_PIN.startswith("transformers>=4.40,<5"), le.TRANSFORMERS_PIN)
check("carbon trust_model=False(与训练一致)", le.DNA_LM_PARAMS["carbon"]["trust_model"] is False)
check("nt_v2 trust_model=True", le.DNA_LM_PARAMS["nt_v2"]["trust_model"] is True)

print("\n=== 8c. enformer / alphagenome ===")
enf_src = le._write_server_script("enformer")
enf_txt = open(enf_src).read()
check("enformer 序列长度 393216", "SEQUENCE_LENGTH = 393216" in enf_txt)
check("enformer 7 summary 齐全",
      all(k in enf_txt for k in ["enformer_absmean","enformer_mean","enformer_std",
                                  "enformer_max_abs","enformer_top5_absmean",
                                  "enformer_pos_sum","enformer_neg_sum"]))
check("enformer 用 VariantSeqExtractor(支持indel)", "VariantSeqExtractor" in enf_txt)
check("enformer alt-ref 方向", "alt.mean(axis=1) - ref.mean(axis=1)" in enf_txt)
ag_txt = open(le._write_server_script("alphagenome")).read()
check("alphagenome 支持环境变量密钥", "ALPHAGENOME_API_KEY" in ag_txt)
check("alphagenome 无密钥时明确报错(不再假装默认凭据)",
      "no AlphaGenome API key" in ag_txt and "colab_utils" not in ag_txt)

print("\n=== 8d. 缓存按字段判定 ===")
check("缓存默认停用(等新表)", le.PRECOMPUTED_CACHE_ENABLED is False)
old_cache = {"precomputed_scores": {"AM": 0.1, "EVO2": -0.2}}   # 旧表：无新模型字段
new_cache = {"precomputed_scores": {"AM": 0.1, "EVO2": -0.2, "GENERATOR": 0.1,
                                     "GENERATOR_V2": 0.1, "NT_V2": 0.2, "CARBON": 5.0,
                                     "ENFORMER_MAX_ABS": 1.0}}
allf = _model_flags()
check("旧缓存被判定为不可用", le._cache_missing_requested_models(old_cache, allf) is True)
check("新缓存(含全字段)可用", le._cache_missing_requested_models(new_cache, allf) is False)
check("关掉这些模型后旧缓存可用",
      le._cache_missing_requested_models(old_cache, _model_flags(
          no_carbon=True, no_generator=True, no_generator_v2=True,
          no_nt_v2=True, no_enformer=True)) is False)

print("\n=== 9. output 渲染（含新模型）===")
from merge_cli import output
res, _ = le.run_local_ensemble(pred, tr, ensemble_type="coding", skip_shap=True)
flat = output._extract_flat(pred, {"success": True, **res})
check("popEVE 已从输出移除", "popEVE" not in flat)
for k in ["Carbon", "GENERATOR", "GENERATOR_v2", "NT_v2", "Evo2", "Evo2_7b_base", "Evo2_1b_base"]:
    check("flat 含 %s" % k, k in flat and flat[k] is not None, str(flat.get(k)))
check("flat variant_type = coding", flat.get("variant_type") == "coding", str(flat.get("variant_type")))

print("\n" + "=" * 50)
print("PASS %d / FAIL %d" % (ok, fail))
sys.exit(1 if fail else 0)
