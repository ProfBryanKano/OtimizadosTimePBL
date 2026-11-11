"""Loading the processed OULAD student table and sampling class cohorts."""
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "students.csv"
LOW_IMD_BANDS = ["0-10%", "10-20"]  # two most deprived deciles (OULAD writes "10-20" without %)


def load_students(path=None):
    """Return the student table used in all experiments.

    Only students with a score on their module's first assessment are kept.
    Row order matters for exact reproducibility of the sampled cohorts.
    """
    path = Path(path) if path else DEFAULT_PROCESSED
    d = pd.read_csv(path, dtype={"code_module": str, "code_presentation": str, "gender": str,
                                 "imd_band": str, "final_result": str})
    d = d[d.score.notna()].copy()
    d["pres"] = d.code_module + "-" + d.code_presentation
    d["low_imd"] = d.imd_band.isin(LOW_IMD_BANDS)
    d["dis"] = d.disability.astype(str) == "True"
    d["logc"] = np.log1p(d.clicks28)
    return d.reset_index(drop=True)


def sample_cohort(d, n, rng, pres=None):
    """Draw a class of n students from one module presentation.

    Returns standardised skill and engagement and a binary matrix G with the
    three under-represented groups (minority gender, disability, low-income area).
    """
    sizes = d.groupby("pres").size()
    ok = sizes[sizes >= n].index.values
    p = pres if pres is not None else rng.choice(ok)
    sub = d[d.pres == p]
    s = sub.iloc[rng.choice(len(sub), n, replace=False)]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    skill = z(s.score.values.astype(float))
    eng = z(s.logc.values)
    fem = s.gender.values == "F"
    minority = fem if fem.mean() <= 0.5 else ~fem
    G = np.stack([minority, s.dis.values, s.low_imd.values], 1).astype(np.int64)
    return dict(pres=p, skill=skill, eng=eng, G=G, final=s.final_result.values)
