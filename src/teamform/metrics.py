"""Per-module metrics and collaboration-history helpers."""
import numpy as np

GROUPS = ["gender", "disability", "low_imd"]


def module_metrics(t, c, m):
    """Balance, isolation and team-quality metrics for one assignment."""
    k = t.max() + 1
    skill, eng, G = c["skill"], c["eng"], c["G"]
    ts = np.bincount(t, skill, k) / m
    te = np.bincount(t, eng, k) / m
    out = dict(sd_skill=ts.std(), sd_eng=te.std())
    anyiso = np.zeros(t.size, bool)
    anymem = G.any(1)
    names = GROUPS if G.shape[1] == len(GROUPS) else [f"group{a}" for a in range(G.shape[1])]
    for a, nm in enumerate(names):
        cnt = np.bincount(t, G[:, a], k)
        mem = G[:, a] == 1
        iso = mem & (cnt[t] == 1)
        out["iso_" + nm] = iso.sum() / max(mem.sum(), 1)
        anyiso |= iso
        out["prev_" + nm] = mem.mean()
    out["iso_any"] = anyiso.sum() / max(anymem.sum(), 1)
    mn = np.full(k, np.inf)
    mx = np.full(k, -np.inf)
    np.minimum.at(mn, t, skill)
    np.maximum.at(mx, t, skill)
    for nm, q in (("add", ts), ("conj", mn), ("disj", mx)):
        out[nm + "_sd"] = q.std()
        out[nm + "_p10"] = np.quantile(q, 0.1)
    return out


def add_pairs(W, t):
    """Record that every pair of teammates in t has worked together once more."""
    for r in np.unique(t):
        idx = np.where(t == r)[0]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                W[idx[x], idx[y]] += 1
                W[idx[y], idx[x]] += 1


def repeat_share(W, t):
    """Share of teammate pairs in t who had already worked together."""
    rep = 0
    tot = 0
    for r in np.unique(t):
        idx = np.where(t == r)[0]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                tot += 1
                rep += W[idx[x], idx[y]] > 0
    return rep / tot
