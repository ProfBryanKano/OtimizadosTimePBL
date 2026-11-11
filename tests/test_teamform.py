import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teamform import (load_students, sample_cohort, random_assignment, serpentine, anneal,  # noqa: E402
                      module_metrics, add_pairs, repeat_share, run_cohort, form_teams)


def synthetic(n=40, seed=0):
    rng = np.random.default_rng(seed)
    G = np.zeros((n, 1), np.int64)
    G[rng.choice(n, 6, replace=False), 0] = 1
    return dict(skill=rng.normal(size=n), eng=rng.normal(size=n), G=G)


def test_random_assignment_has_equal_teams():
    t = random_assignment(40, 4, np.random.default_rng(0))
    assert np.bincount(t).tolist() == [4] * 10


def test_serpentine_is_deterministic_and_balanced():
    c = synthetic()
    t1, t2 = serpentine(c["skill"], 4), serpentine(c["skill"], 4)
    assert (t1 == t2).all() and np.bincount(t1).tolist() == [4] * 10


def test_anneal_beats_random_on_balance():
    c = synthetic()
    rng = np.random.default_rng(1)
    t0 = random_assignment(40, 4, rng)
    W = np.zeros((40, 40))
    t = anneal(t0, c["skill"], c["eng"], c["G"], W, 10, 4, 0.0, 0.0, 100000, 1)
    assert module_metrics(t, c, 4)["sd_skill"] < 0.5 * module_metrics(t0, c, 4)["sd_skill"]


def test_fairness_penalty_removes_isolation():
    c = synthetic()
    t0 = random_assignment(40, 4, np.random.default_rng(2))
    W = np.zeros((40, 40))
    t = anneal(t0, c["skill"], c["eng"], c["G"], W, 10, 4, 1.0, 0.0, 100000, 2)
    assert module_metrics(t, c, 4)["iso_any"] == 0.0


def test_memory_reduces_repeated_pairs():
    c = synthetic()
    W = np.zeros((40, 40))
    first = anneal(random_assignment(40, 4, np.random.default_rng(3)), c["skill"], c["eng"], c["G"], W, 10, 4, 1.0, 0.0, 100000, 3)
    add_pairs(W, first)
    no_mem = anneal(random_assignment(40, 4, np.random.default_rng(4)), c["skill"], c["eng"], c["G"], W, 10, 4, 1.0, 0.0, 100000, 4)
    mem = anneal(random_assignment(40, 4, np.random.default_rng(4)), c["skill"], c["eng"], c["G"], W, 10, 4, 1.0, 0.1, 100000, 4)
    assert repeat_share(W, mem) < repeat_share(W, no_mem)


def test_repeat_share_counts_pairs():
    W = np.zeros((4, 4))
    add_pairs(W, np.array([0, 0, 1, 1]))
    assert repeat_share(W, np.array([0, 0, 1, 1])) == 1.0
    assert repeat_share(W, np.array([0, 1, 0, 1])) == 0.0


def test_form_teams_handles_uneven_class_and_history():
    rng = np.random.default_rng(5)
    n = 42
    groups = (rng.random((n, 1)) < 0.2).astype(int)
    t1 = form_teams(rng.normal(size=n), groups=groups, team_size=4)
    t2 = form_teams(rng.normal(size=n), groups=groups, team_size=4, previous_assignments=[t1])
    assert len(t2) == n and set(np.bincount(t2)) <= {3, 4}


@pytest.mark.skipif(not (ROOT / "data" / "processed" / "students.csv").exists(), reason="processed data missing")
def test_data_and_exact_reproduction_of_published_results():
    d = load_students()
    assert len(d) == 25354 and d.pres.nunique() == 22
    rows, _ = run_cohort(d, 40, 4, 0, ["SA-EM"])
    got = pd.DataFrame(rows)
    ref = pd.read_csv(ROOT / "results" / "raw" / "all_modules.csv")
    ref = ref[(ref.n == 40) & (ref.m == 4) & (ref.rep == 0) & (ref.policy == "SA-EM")].sort_values("module")
    assert np.allclose(got.sd_skill.values, ref.sd_skill.values)
    assert np.allclose(got.repeat_share.fillna(-1).values, ref.repeat_share.fillna(-1).values)
