"""Build data/processed/students.csv from the Open University Learning Analytics Dataset (OULAD).

Two sources are supported:
  --source github  downloads the R-data mirror maintained by the dataset authors (needs pyreadr)
  --source csv     reads the official CSV files (studentInfo.csv, assessments.csv,
                   studentAssessment.csv, studentVle.csv) from --csv-dir

The repository already ships data/processed/students.csv, so this step is only needed
to rebuild it from scratch.
"""
import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "students.csv"
GITHUB_ZIP = "https://codeload.github.com/jakubkuzilek/oulad/zip/refs/heads/master"
TEXT_COLS = ["code_module", "code_presentation", "assessment_type", "gender", "region",
             "highest_education", "imd_band", "age_band", "final_result"]


def load_github():
    import pyreadr
    folder = RAW / "oulad-master" / "data"
    if not folder.exists():
        print("downloading", GITHUB_ZIP)
        RAW.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(urlopen(GITHUB_ZIP).read())).extractall(RAW)
    rd = lambda name: pyreadr.read_r(str(folder / f"{name}.rda"))[name]
    return rd("student"), rd("assessment"), rd("student_assessment"), rd("student_vle")


def load_csv(csv_dir):
    p = Path(csv_dir)
    st = pd.read_csv(p / "studentInfo.csv")
    st["disability"] = st.disability.map({"Y": True, "N": False})
    return (st, pd.read_csv(p / "assessments.csv"), pd.read_csv(p / "studentAssessment.csv"),
            pd.read_csv(p / "studentVle.csv"))


def build(st, asm, sa, vle):
    for df in (st, asm):
        for c in TEXT_COLS:
            if c in df:
                df[c] = df[c].astype(str).replace({"nan": pd.NA, "<NA>": pd.NA})
    asm["date"] = pd.to_numeric(asm["date"], errors="coerce")
    # first non-exam assessment of each module presentation
    a = asm[asm.assessment_type != "Exam"].dropna(subset=["date"]).sort_values("date", kind="stable")
    first = a.groupby(["code_module", "code_presentation"]).head(1)[["code_module", "code_presentation", "id_assessment"]]
    sa = sa.copy()
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    f = first.merge(sa, on="id_assessment")
    keys = ["code_module", "code_presentation", "id_student"]
    df = st.merge(f[keys + ["score"]], on=keys, how="left")
    vle = vle.copy()
    for c in ("code_module", "code_presentation"):
        vle[c] = vle[c].astype(str)
    clicks = (vle[(vle.date >= 0) & (vle.date < 28)].groupby(keys)["sum_click"].sum()
              .rename("clicks28").reset_index())
    df = df.merge(clicks, how="left", on=keys)
    df["clicks28"] = df.clicks28.fillna(0)
    df = df[df.score.notna()]
    cols = keys + ["gender", "disability", "imd_band", "final_result", "score", "clicks28"]
    return df[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["github", "csv"], default="github")
    ap.add_argument("--csv-dir", default=str(RAW / "oulad_csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    tables = load_github() if args.source == "github" else load_csv(args.csv_dir)
    df = build(*tables)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} students, {df.groupby(['code_module','code_presentation']).ngroups} presentations")


if __name__ == "__main__":
    main()
