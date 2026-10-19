# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Run every validation script and print a summary table.

Usage:  python validation/run_all.py
Exit status is non-zero if any script reports a failed check.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = [
    ("Lemma 2.1", "v01_lemma_2_1_pohlig_hellman.py"),
    ("Theorem 2.2", "v02_theorem_2_2_smoothness_invariance.py"),
    ("Proposition 2.3", "v03_proposition_2_3_quotient_sectors.py"),
    ("Lemma 3.1", "v04_lemma_3_1_carry_expansion.py"),
    ("Theorem 3.2 / Conj. 3.3", "v05_theorem_3_2_higher_digit_rank.py"),
    ("Lemma 4.1", "v06_lemma_4_1_valuation_additivity.py"),
    ("Theorem 4.2", "v07_theorem_4_2_valuation_clock.py"),
]


def main() -> int:
    rows, failed = [], False
    for label, script in SCRIPTS:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE,
                              capture_output=True, text=True)
        dt = time.time() - t0
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        last = [ln for ln in proc.stdout.splitlines() if "checks passed" in ln]
        summary = last[-1] if last else "no summary (crashed?)"
        ok = proc.returncode == 0
        failed |= not ok
        rows.append((label, script, "PASS" if ok else "FAIL", summary, dt))
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    for label, script, status, summary, dt in rows:
        print(f"{status:4}  {label:24} {summary:28} {dt:6.1f}s  {script}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
