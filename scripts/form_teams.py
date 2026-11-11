"""Form teams for your own class from a CSV roster.

Roster: one row per student, with an id column, a skill column (e.g. a first grade),
optionally an engagement column and 0/1 columns for under-represented groups.
History: optional CSV files from earlier modules with columns <id-col> and team.

Example
  python scripts/form_teams.py --roster examples/roster_example.csv --id-col student \
      --skill-col grade --engagement-col activity --group-cols woman disability \
      --team-size 4 --history examples/module1_teams.csv --out teams_module2.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teamform.api import form_teams  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--skill-col", required=True)
    ap.add_argument("--engagement-col")
    ap.add_argument("--group-cols", nargs="*", default=[])
    ap.add_argument("--team-size", type=int, default=4)
    ap.add_argument("--history", nargs="*", default=[], help="team files from previous modules")
    ap.add_argument("--no-fairness", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="teams.csv")
    a = ap.parse_args()
    r = pd.read_csv(a.roster)
    ids = r[a.id_col].values
    prev = []
    for h in a.history:
        hdf = pd.read_csv(h).set_index(a.id_col)["team"]
        prev.append(np.array([hdf.get(i, -1) for i in ids]))
    t = form_teams(r[a.skill_col].values,
                   r[a.engagement_col].values if a.engagement_col else None,
                   r[a.group_cols].values if a.group_cols else None,
                   team_size=a.team_size, previous_assignments=prev or None,
                   fairness=not a.no_fairness, memory=not a.no_memory, seed=a.seed)
    out = pd.DataFrame({a.id_col: ids, "team": t})
    out.to_csv(a.out, index=False)
    print(f"{len(out)} students in {out.team.nunique()} teams -> {a.out}")
    for g in a.group_cols:
        cnt = r.assign(team=t).groupby("team")[g].transform("sum")
        print(f"  {g}: {int(((r[g] == 1) & (cnt == 1)).sum())} students alone in their team")


if __name__ == "__main__":
    main()
