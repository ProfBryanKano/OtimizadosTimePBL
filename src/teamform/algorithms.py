"""Team-assignment algorithms.

Teams are encoded as an integer vector t of length n, with t[i] the team of student i.
All teams have the same size m, so n must be a multiple of m.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def perm_to_t(p, m):
    t = np.empty(p.size, np.int64)
    for r in range(p.size):
        t[p[r]] = r // m
    return t


def random_assignment(n, m, rng):
    """Uniform random partition into teams of size m."""
    return perm_to_t(rng.permutation(n), m)


def serpentine(skill, m):
    """Snake draft by skill (deterministic: always returns the same teams)."""
    n = skill.size
    k = n // m
    o = np.argsort(-skill)
    t = np.empty(n, np.int64)
    for r, i in enumerate(o):
        rd, pos = divmod(r, k)
        t[i] = pos if rd % 2 == 0 else k - 1 - pos
    return t


@njit(cache=True)
def anneal(t0, skill, eng, G, W, k, m, lam, mu, iters, seed):
    """Simulated annealing with pairwise swaps.

    Objective (minimised):
      sum over teams of squared deviations of team mean skill and engagement from the class mean
      + lam * number of (team, group) pairs with exactly one member of the group  (fairness)
      + mu  * sum over teammate pairs of W[i, j], the number of times they already worked together (memory)
    """
    np.random.seed(seed)
    n = t0.size
    A = G.shape[1]
    t = t0.copy()
    mem = np.empty((k, m), np.int64)
    pos = np.empty(n, np.int64)
    fill = np.zeros(k, np.int64)
    for i in range(n):
        r = t[i]
        mem[r, fill[r]] = i
        pos[i] = fill[r]
        fill[r] += 1
    S = np.zeros(k)
    E = np.zeros(k)
    C = np.zeros((k, A), np.int64)
    for i in range(n):
        S[t[i]] += skill[i]
        E[t[i]] += eng[i]
        for a in range(A):
            C[t[i], a] += G[i, a]
    ms = skill.mean()
    me = eng.mean()
    cur = 0.0
    for r in range(k):
        cur += (S[r] / m - ms) ** 2 + (E[r] / m - me) ** 2
        for a in range(A):
            if C[r, a] == 1:
                cur += lam
        for x in range(m):
            for y in range(x + 1, m):
                cur += mu * W[mem[r, x], mem[r, y]]
    tot = 0.0
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(n)
        tot += abs((skill[i] - skill[j]) / m) * 0.5 + abs((eng[i] - eng[j]) / m) * 0.5
    T0 = tot / 200 * 0.5 + 1e-9
    T1 = T0 * 1e-5
    best = t.copy()
    bestE = cur
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / iters)
        i = np.random.randint(n)
        j = np.random.randint(n)
        a_ = t[i]
        b_ = t[j]
        if a_ == b_:
            continue
        ds = skill[j] - skill[i]
        de = eng[j] - eng[i]
        d = ((S[a_] + ds) / m - ms) ** 2 + ((E[a_] + de) / m - me) ** 2 + ((S[b_] - ds) / m - ms) ** 2 + ((E[b_] - de) / m - me) ** 2 \
            - ((S[a_] / m - ms) ** 2 + (E[a_] / m - me) ** 2 + (S[b_] / m - ms) ** 2 + (E[b_] / m - me) ** 2)
        if lam > 0:
            for a in range(A):
                gi = G[i, a]
                gj = G[j, a]
                if gi != gj:
                    ca = C[a_, a]
                    cb = C[b_, a]
                    ca2 = ca - gi + gj
                    cb2 = cb - gj + gi
                    d += lam * ((ca2 == 1) + (cb2 == 1) - (ca == 1) - (cb == 1))
        if mu > 0:
            rep = 0.0
            for x in range(m):
                u = mem[a_, x]
                v = mem[b_, x]
                if u != i:
                    rep += W[j, u] - W[i, u]
                if v != j:
                    rep += W[i, v] - W[j, v]
            d += mu * rep
        if d < 0 or np.random.random() < np.exp(-d / T):
            pi = pos[i]
            pj = pos[j]
            mem[a_, pi] = j
            mem[b_, pj] = i
            pos[i] = pj
            pos[j] = pi
            t[i] = b_
            t[j] = a_
            S[a_] += ds
            S[b_] -= ds
            E[a_] += de
            E[b_] -= de
            for a in range(A):
                C[a_, a] += G[j, a] - G[i, a]
                C[b_, a] += G[i, a] - G[j, a]
            cur += d
            if cur < bestE - 1e-15:
                bestE = cur
                best[:] = t
    return best
