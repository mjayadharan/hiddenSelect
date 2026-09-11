"""verify_gates.py — the gate harness. Copy to <report>/analysis/verify_<id>.py.

A gate is a machine-checkable assertion about the report's own outputs. The
report quotes gate results; the gates are what make the quoting honest.
Every gate writes into results/gates_summary.json, and a FAILING gate is
reported as failing — a failed gate that is itself the finding stays failed.

The gate families below are the ones that have earned their place. Keep the
ones that apply, delete the rest, and add report-specific ones.

Run: python verify_<id>.py     (exit code 1 if any gate fails)
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEPS = HERE.parent / "deps"
for _p in (str(DEPS), str(HERE)):          # deps/ FIRST: frozen code wins
    if _p not in sys.path:
        sys.path.insert(0, _p)

RES = HERE / "results"
GATES = {}


def gate(name, ok, detail=""):
    GATES[name] = {"pass": bool(ok), "detail": str(detail)}
    print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")


def main():
    # --- G1  INHERITED STAGE REPRODUCES ITS SOURCE, BYTE FOR BYTE ----------
    # Recompute anything inherited from an earlier report and compare whole
    # files. This is the strongest staleness clearance available; prefer it
    # over assuming a persisted result still holds.
    #
    # ok = (RES / f).read_bytes() == (REF / f).read_bytes()
    # gate("G1_inherited_matches_source", ok, "; ".join(details))

    # --- G2  A REPRODUCED ARM MATCHES ITS PERSISTED VERSION ---------------
    # Keyed comparison: join on the identifying columns, then require exact
    # equality of the compared fields. Report it as KEYED VALUE EQUALITY,
    # not byte identity, and say which columns were outside the join.

    # --- G3  SOLVER IDENTITY ----------------------------------------------
    # A new implementation must reproduce the frozen one on a shared special
    # case (e.g. ridge at lambda = 0 == the frozen least-squares refit) to a
    # stated relative tolerance.

    # --- G4  MATHEMATICAL IDENTITY ----------------------------------------
    # Recompute an identity the report leans on (a projection, a
    # factorisation, a null-space relation) directly and bound the error.

    # --- G5  JOIN COMPLETENESS --------------------------------------------
    # Expected row count, no unexpected NaN, and every permitted NaN
    # ENUMERATED in the detail string with the reason it is permitted.

    # --- G6  PROTOCOL INVARIANTS ------------------------------------------
    # Whatever the pipeline promises about its own output (budget respected,
    # flags mutually consistent, fallbacks implying their precondition).

    # --- G7  CONTROL: A METRIC THAT MUST *NOT* MOVE -----------------------
    # If a metric is arm-independent by construction, assert zero spread
    # across arms. This catches leakage between an arm comparison and a
    # quantity that cannot depend on the arm.

    # --- G8  INSTRUMENTATION MEASURES WHAT IT CLAIMS ----------------------
    # If two recorded quantities are meant to differ (a first-pass count vs a
    # post-loop count), assert they are identical where they must be and
    # differ somewhere where they must.

    # --- G9  DERIVED CLAIMS ARE INTERNALLY CONSISTENT ---------------------
    # Assert the arithmetic the report prints: decomposition identities,
    # every confidence interval bracketing its own point estimate, expected
    # row counts of derived tables.

    # --- G10 COMPLETENESS + HASH LEDGER -----------------------------------
    ok, det = True, []
    hashes = {}
    for p in sorted(RES.glob("*.csv")):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        det.append(f"{p.name}:{sum(1 for _ in p.open()) - 1}")
    ok &= len(hashes) > 0
    (RES / "results_hashes.json").write_text(json.dumps(hashes, indent=1))
    gate("G10_completeness_hashes", ok, "; ".join(det))

    (RES / "gates_summary.json").write_text(json.dumps(GATES, indent=1))
    n_fail = sum(1 for g in GATES.values() if not g["pass"])
    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATES FAIL'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
