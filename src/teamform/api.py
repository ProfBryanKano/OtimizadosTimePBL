"""High-level function to form teams for a real class roster."""
import numpy as np

from .algorithms import random_assignment, anneal
from .metrics import add_pairs


def _z(x):
    x = np.asarray(x, float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def history_matrix(n, previous_assignments):
    """Build the co-membership count matrix W from a list of previous team vectors.

    Each element is an array of length n with the team of each student in a past module
    (use -1 for students who were absent from that module).
    """
    W = np.zeros((n, n))
    for t in previous_assignments:
        t = np.asarray(t)
        for r in np.unique(t[t >= 0]):
            idx = np.where(t == r)[0]
            W[np.ix_(idx, idx)] += 1
        np.fill_diagonal(W, 0)
    return W


def form_teams(skill, engagement=None, groups=None, team_size=4, previous_assignments=None,
               fairness=True, memory=True, mu=0.1, lam=1.0, seed=0, iters=None):
    """Return a team id (0..k-1) for each student.

    skill, engagement : arrays of length n (any scale, standardised internally)
    groups            : array (n, g) of 0/1 flags for under-represented groups; with fairness=True
                        the algorithm avoids teams with exactly one member of a group
    previous_assignments : list of team vectors from earlier modules; with memory=True the
                        algorithm avoids repeating pairs who already worked together
    If n is not a multiple of team_size, placeholder seats are added and removed afterwards,
    so a few teams end up with team_size - 1 students.
    """
    skill = np.asarray(skill, float)
    n = skill.size
    eng = np.zeros(n) if engagement is None else np.asarray(engagement, float)
    G = np.zeros((n, 1), np.int64) if groups is None else np.asarray(groups, np.int64).reshape(n, -1)
    W = np.zeros((n, n)) if previous_assignments is None else history_matrix(n, previous_assignments)
    m = int(team_size)
    pad = (-n) % m
    s, e = _z(skill), _z(eng)
    if pad:  # placeholders with average profile, no group, no history
        s = np.r_[s, np.zeros(pad)]
        e = np.r_[e, np.zeros(pad)]
        G = np.r_[G, np.zeros((pad, G.shape[1]), np.int64)]
        W2 = np.zeros((n + pad, n + pad))
        W2[:n, :n] = W
        W = W2
    N = n + pad
    k = N // m
    rng = np.random.default_rng(seed)
    t = anneal(random_assignment(N, m, rng), s, e, np.ascontiguousarray(G), W, k, m,
               lam if fairness else 0.0, mu if memory else 0.0, iters or max(400 * N, 100000), seed)
    return t[:n]
