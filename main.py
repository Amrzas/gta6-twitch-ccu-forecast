"""Entry point: forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads every <id>.in in the input directory (the grader puts one case there) and
writes <id>.out beside the other outputs: 23 rows, hours 2..24, five quantiles
of that hour's mean concurrent viewers in the "Grand Theft Auto VI" category.

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Standard library only; no network, no model calls, deterministic: two runs on
the same case write the same bytes.
"""

import json
import os
import sys
from pathlib import Path

# The grader runs `python3 -P -s`, which keeps this file's directory off
# sys.path: put it back so `import model` finds model.py beside main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def hour1_print(case):
    """The case's hour-1 count: the mean of the first hour's viewers."""
    for row in case.get("known") or ():
        if row.get("hour") == 1:
            return float(row.get("viewers") or 0)
    raise ValueError("case has no hour-1 print")


def forecast_case(case):
    rows = model.forecast(
        hour1_print(case),
        case.get("notes") or "",
        case.get("comparables") or [],
    )
    hours = [{"hour": h, "q": q} for h, q in ((r["hour"], r["q"]) for r in rows)]
    wanted = [row["hour"] for row in case.get("forecast") or ()]
    if wanted and wanted != [r["hour"] for r in hours]:
        hours = [r for r in hours if r["hour"] in wanted]
    if len(hours) != 23:
        raise ValueError(f"wrote {len(hours)} rows, not 23")
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        body = json.dumps(out, separators=(",", ":"))
        if len(body.encode("utf-8")) > 65536:
            raise ValueError(f"{path.stem}.out is over the 65,536-byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(body + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
