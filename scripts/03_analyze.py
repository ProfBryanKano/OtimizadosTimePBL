"""Merge raw results and compute every number reported in the paper.

Writes
  results/raw/all_modules.csv       one row per cohort x policy x module
  results/raw/all_curriculum.csv    one row per cohort x policy (8-module summary)
  results/summary/cohort_summary.csv
  results/summary/policy_summary.csv
  results/summary/paper_numbers.md  numbers quoted in the text, with 95% bootstrap CIs
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SUM = ROOT / "results" / "summary"
rng = np.random.default_rng(5)


def ci(x, B=3000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)


def fmt(t, k=3):
    return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"


def load():
    parts = sorted(glob.glob(str(RAW / "parts" / "modules_*.csv")))
    if parts:
        M = pd.concat([pd.read_csv(f) for f in parts]).drop_duplicates(["n", "m", "rep", "policy", "module"])
        C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RAW / "parts" / "curriculum_*.csv")))])
        C = C.drop_duplicates(["n", "m", "rep", "policy"])
        M.to_csv(RAW / "all_modules.csv", index=False)
        C.to_csv(RAW / "all_curriculum.csv", index=False)
    return pd.read_csv(RAW / "all_modules.csv"), pd.read_csv(RAW / "all_curriculum.csv")


def main():
    SUM.mkdir(parents=True, exist_ok=True)
    M, C = load()
    A = M.groupby(["n", "m", "rep", "policy"]).agg(
        sd_skill=("sd_skill", "mean"), sd_eng=("sd_eng", "mean"), iso=("iso_any", "mean"),
        rep_share=("repeat_share", "mean"), sec=("sec", "median")).reset_index()
    base = A[A.policy == "RND"].set_index(["n", "m", "rep"])
    A = A.join(base[["sd_skill", "sd_eng"]].rename(columns=lambda c: c + "_rnd"), on=["n", "m", "rep"])
    A["balance_gain_skill"] = 1 - A.sd_skill / A.sd_skill_rnd
    A["balance_gain_eng"] = 1 - A.sd_eng / A.sd_eng_rnd
    A = A.merge(C, on=["n", "m", "rep", "policy"])
    A["deficit_protected"] = A.distinct_nonprot - A.distinct_prot
    A.to_csv(SUM / "cohort_summary.csv", index=False)
    cols = ["balance_gain_skill", "balance_gain_eng", "iso", "rep_share", "distinct_mean", "distinct_min",
            "deficit_protected", "ingroup_ratio", "modularity", "ever_iso", "sec"]
    A.groupby(["n", "m", "policy"])[cols].mean().round(4).to_csv(SUM / "policy_summary.csv")

    L = ["# Numbers reported in the paper", "", "Means with 95% bootstrap confidence intervals.", ""]
    # one module (module 1, teams of four)
    M1 = M[(M.module == 0) & (M.m == 4)]
    L.append("## One module (first module, teams of 4)")
    for n in (40, 120, 400):
        r = M1[(M1.n == n) & (M1.policy == "RND")]
        L.append(f"- n={n}: random isolates {r.iso_any.mean():.1%} of under-represented students "
                 f"(gender {r.iso_gender.mean():.1%}, disability {r.iso_disability.mean():.1%}, low income {r.iso_low_imd.mean():.1%})")
    p = C[(C.m == 4) & (C.n == 120) & (C.policy == "RND")][["pred_gender", "pred_dis", "pred_imd"]].mean()
    L.append(f"- hypergeometric prediction at n=120: {p.pred_gender:.1%}, {p.pred_dis:.1%}, {p.pred_imd:.1%}")
    g4 = A[A.m == 4]
    for n in (40, 120, 400):
        s = g4[g4.n == n].groupby("policy")
        sa, sae = s.balance_gain_skill.mean()["SA"], s.balance_gain_skill.mean()["SA-E"]
        L.append(f"- n={n}: balance kept SA {sa:.1%}, SA-E {sae:.1%}, price of fairness {100 * (sa - sae):.1f} pp; "
                 f"SNK skill {s.balance_gain_skill.mean()['SNK']:.1%}, engagement {s.balance_gain_eng.mean()['SNK']:.1%}")
    L += ["", "## Curriculum (8 modules, teams of 4)"]
    for n in (40, 120, 400):
        s = g4[g4.n == n]
        w = s.pivot_table(index="rep", columns="policy", values="rep_share")
        L.append(f"- n={n}: repeated pairs RND {fmt(ci(w.RND))}, SA-E {fmt(ci(w['SA-E']))}, SA-EM {fmt(ci(w['SA-EM']))}; "
                 f"SA-EM < RND in {(w['SA-EM'] < w.RND).mean():.0%} of cohorts (Wilcoxon p={wilcoxon(w['SA-EM'], w.RND).pvalue:.1e})")
        q = s.groupby("policy")
        L.append(f"  distinct teammates RND {q.distinct_mean.mean()['RND']:.1f} (min {q.distinct_min.mean()['RND']:.1f}), "
                 f"SA-E {q.distinct_mean.mean()['SA-E']:.1f} (min {q.distinct_min.mean()['SA-E']:.1f}), "
                 f"SA-EM {q.distinct_mean.mean()['SA-EM']:.1f} (min {q.distinct_min.mean()['SA-EM']:.1f}); "
                 f"cost of memory {100 * (q.balance_gain_skill.mean()['SA-E'] - q.balance_gain_skill.mean()['SA-EM']):.1f} pp; "
                 f"SA-EM isolation {q.iso.mean()['SA-EM']:.1%}; median time SA-EM {1000 * q.sec.median()['SA-EM']:.1f} ms/module")
        w = s.pivot_table(index="rep", columns="policy", values="deficit_protected")
        L.append(f"  protected deficit SA-EM {fmt(ci(w['SA-EM']), 2)}, ADAPT {fmt(ci(w['SA-EM-ADAPT']), 2)}, "
                 f"ALT {fmt(ci(w['SA-EM-ALT']), 2)}; in-group tie ratio SA-EM {q.ingroup_ratio.mean()['SA-EM']:.2f}; "
                 f"ever isolated SA-EM {q.ever_iso.mean()['SA-EM']:.1%}, ALT {q.ever_iso.mean()['SA-EM-ALT']:.1%}, ADAPT {q.ever_iso.mean()['SA-EM-ADAPT']:.1%}")
    s40 = g4[g4.n == 40].groupby("policy")
    L.append(f"- n=40 small-group members: share of protected {s40.small_share.mean()['SA-EM']:.1%}; distinct teammates "
             f"SA-E {s40.distinct_small.mean()['SA-E']:.1f}, SA-EM {s40.distinct_small.mean()['SA-EM']:.1f}, ADAPT {s40.distinct_small.mean()['SA-EM-ADAPT']:.1f}")
    L += ["", "## Team-size robustness (n=120)"]
    for m in (3, 5):
        s = A[(A.n == 120) & (A.m == m)].groupby("policy")
        L.append(f"- teams of {m}: repeated pairs RND {s.rep_share.mean()['RND']:.1%}, SA-E {s.rep_share.mean()['SA-E']:.1%}, "
                 f"SA-EM {s.rep_share.mean()['SA-EM']:.1%}; distinct SA-EM {s.distinct_mean.mean()['SA-EM']:.1f} of {min(8 * (m - 1), 119)}")
    (SUM / "paper_numbers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
