"""Running team-formation policies over a multi-module curriculum."""
import time
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from scipy.special import comb

from .algorithms import random_assignment, serpentine, anneal
from .data import sample_cohort
from .metrics import module_metrics, repeat_share, add_pairs

# Policy codes (names used in the paper in brackets)
POLICIES = {
    "RND": "random draw each module [Sorteio]",
    "SNK": "serpentine by skill [Serpentina]",
    "SA": "annealing, balance only, re-optimised each module [Annealing (equilibrio)]",
    "SA-E": "annealing + no-isolation penalty [Annealing + equidade]",
    "SA-EM": "annealing + no-isolation + memory of past pairs [Annealing + equidade + memoria]",
    "SA-EM-ALT": "as SA-EM, fairness only in alternate modules [equidade alternada]",
    "SA-EM-ADAPT": "as SA-EM, fairness off for groups with 2-3 members [equidade adaptativa]",
}
REDUCED_POLICIES = ["RND", "SA-E", "SA-EM"]  # used in the team-size robustness runs


def _iters(n):
    return max(400 * n, 100000)


def run_policy(c, m, T, pol, seed, mu=0.1):
    """Run one policy for T modules on cohort c. Returns per-module rows and a curriculum summary."""
    rng = np.random.default_rng(seed)
    n = c["skill"].size
    k = n // m
    W = np.zeros((n, n))
    G = c["G"]
    cnt = G.sum(0)
    Gadapt = G[:, cnt >= 4] if (cnt >= 4).any() else np.zeros((n, 1), np.int64)
    mods = []
    ever = np.zeros(n, bool)
    for r in range(T):
        t0 = time.perf_counter()
        if pol == "RND":
            t = random_assignment(n, m, rng)
        elif pol == "SNK":
            t = serpentine(c["skill"], m)
        else:
            lam = {"SA": 0.0, "SA-E": 1.0, "SA-EM": 1.0, "SA-EM-ALT": 1.0 if r % 2 == 0 else 0.0,
                   "SA-EM-ADAPT": 1.0}[pol]
            muu = 0.0 if pol in ("SA", "SA-E") else mu
            Gp = np.ascontiguousarray(Gadapt) if pol == "SA-EM-ADAPT" else G
            t = anneal(random_assignment(n, m, rng), c["skill"], c["eng"], Gp, W, k, m, lam, muu,
                       _iters(n), seed * 100 + r)
        sec = time.perf_counter() - t0
        mm = module_metrics(t, c, m)
        mm["repeat_share"] = repeat_share(W, t) if r > 0 else np.nan
        mm["module"] = r
        mm["sec"] = sec
        for a in range(G.shape[1]):
            cn = np.bincount(t, G[:, a], k)
            ever |= (G[:, a] == 1) & (cn[t] == 1)
        mods.append(mm)
        add_pairs(W, t)
    deg = (W > 0).sum(1)
    Gp = G.any(1)
    g = nx.from_numpy_array((W > 0).astype(int))
    Q = modularity(g, greedy_modularity_communities(g)) if g.number_of_edges() > 0 else np.nan
    ing, ex = [], []
    for a in range(G.shape[1]):
        mem = np.where(G[:, a] == 1)[0]
        if len(mem) < 2:
            continue
        ties = W[mem] > 0
        ing.append(ties[:, mem].sum() / max(ties.sum(), 1))
        ex.append((len(mem) - 1) / (n - 1))
    small = np.zeros(n, bool)
    for a in range(G.shape[1]):
        if 2 <= cnt[a] <= 3:
            small |= G[:, a] == 1
    cum = dict(distinct_mean=deg.mean(), distinct_min=deg.min(), distinct_prot=deg[Gp].mean(),
               distinct_nonprot=deg[~Gp].mean(), ingroup_ratio=np.mean(ing) / np.mean(ex), modularity=Q,
               ever_iso=ever[Gp].mean(), small_share=small[Gp].mean(),
               distinct_small=deg[small].mean() if small.any() else np.nan,
               ever_iso_small=ever[small].mean() if small.any() else np.nan)
    return mods, cum


def cohort_seed(n, m, rep):
    return 70000 + 1000 * n + 10 * m + rep


def run_cohort(d, n, m, rep, policies, T=8):
    """Sample one cohort (deterministic seed) and run every policy on it."""
    c = sample_cohort(d, n, np.random.default_rng(cohort_seed(n, m, rep)))
    G = c["G"]
    pred = [comb(n - K, m - 1) / comb(n - 1, m - 1) for K in G.sum(0)]
    rows, cums = [], []
    for p in policies:
        mods, cum = run_policy(c, m, T, p, rep)
        for x in mods:
            rows.append(dict(x, policy=p, n=n, m=m, rep=rep))
        cum.update(policy=p, n=n, m=m, rep=rep, pred_gender=pred[0], pred_dis=pred[1], pred_imd=pred[2])
        cums.append(cum)
    return rows, cums
