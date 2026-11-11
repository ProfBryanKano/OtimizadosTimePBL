"""Run the curriculum experiments of the paper.

Default (no arguments) reproduces everything:
  * classes of 40, 120 and 400 students, teams of 4, 8 modules, 50 cohorts each, all 7 policies
  * classes of 120 with teams of 3 and 5, 30 cohorts each, policies RND, SA-E, SA-EM

Runs can be split into chunks (useful on slow machines) with --only and --reps;
each chunk writes its own file to results/raw/parts/ and 03_analyze.py merges them.

Examples
  python scripts/02_run_experiments.py                      # full paper run
  python scripts/02_run_experiments.py --only 40x4 --reps 0 10
  python scripts/02_run_experiments.py --quick              # 3 cohorts per setting, smoke test
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teamform import load_students, POLICIES, run_cohort  # noqa: E402
from teamform.curriculum import REDUCED_POLICIES  # noqa: E402

SETTINGS = {  # name: (class size, team size, cohorts, policies)
    "40x4": (40, 4, 50, list(POLICIES)),
    "120x4": (120, 4, 50, list(POLICIES)),
    "400x4": (400, 4, 50, list(POLICIES)),
    "120x3": (120, 3, 30, REDUCED_POLICIES),
    "120x5": (120, 5, 30, REDUCED_POLICIES),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SETTINGS), help="settings to run (default: all)")
    ap.add_argument("--reps", nargs=2, type=int, metavar=("START", "END"), help="cohort range [START, END)")
    ap.add_argument("--quick", action="store_true", help="3 cohorts per setting")
    ap.add_argument("--modules", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "parts"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_students()
    for name in args.only or SETTINGS:
        n, m, R, pols = SETTINGS[name]
        r0, r1 = args.reps if args.reps else (0, 3 if args.quick else R)
        t0 = time.time()
        rows, cums = [], []
        for rep in range(r0, r1):
            a, b = run_cohort(d, n, m, rep, pols, T=args.modules)
            rows += a
            cums += b
        pd.DataFrame(rows).to_csv(out / f"modules_{name}_{r0}-{r1}.csv", index=False)
        pd.DataFrame(cums).to_csv(out / f"curriculum_{name}_{r0}-{r1}.csv", index=False)
        print(f"{name}: cohorts {r0}-{r1 - 1} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
