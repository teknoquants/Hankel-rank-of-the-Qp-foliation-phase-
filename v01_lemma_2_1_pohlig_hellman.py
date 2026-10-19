# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Lemma 2.1 (Pohlig--Hellman character reduction).

Statement.  Let p be prime with p - 1 = prod q_i^{e_i}, every
q_i <= B.  For y in F_p^*, the discrete logarithm x = log_g y, and hence the
value chi(y) = chi(g)^x of any multiplicative character chi, is computable
with O( sum_i e_i (log p + q_i) ) multiplications in F_p, which is
O( log p * (log p + B) ).

Checks
------
1. Pohlig--Hellman returns the true discrete log (exhaustive for p < 2000,
   random sample for larger primes).
2. chi_k(y) := e(k x / (p-1)) built from the PH logarithm is a homomorphism:
   chi_k(y y') = chi_k(y) chi_k(y') on random pairs.
3. The instrumented multiplication count obeys
       mults <= C * sum_i e_i (log2 p + q_i)          (C = 2)
   and also the coarser O(log p (log p + B)) bound.
"""
from __future__ import annotations

import cmath
import math
import random

from common import (Counter, Report, dlog_table, factor, pohlig_hellman,
                    prim_root, smoothness)

PRIMES = [97, 193, 257, 769, 1153, 7681, 12289, 40961, 65537, 786433]
C_CONST = 2


def main() -> None:
    rep = Report("Lemma 2.1 -- Pohlig--Hellman character reduction")
    rng = random.Random(2101)
    for p in PRIMES:
        g = prim_root(p)
        fac = factor(p - 1)
        B = smoothness(p - 1)
        bound_fine = sum(e * (math.log2(p) + q) for q, e in fac.items())
        bound_coarse = math.log2(p) * (math.log2(p) + B)
        if p < 2000:
            ys = range(1, p)
            table = dlog_table(p, g)
        else:
            ys = [rng.randrange(1, p) for _ in range(300)]
            table = None
        all_ok, worst, max_mults = True, 0.0, 0
        for y in ys:
            c = Counter()
            x = pohlig_hellman(y, g, p, c)
            ok = pow(g, x, p) == y and (table is None or table[y] == x)
            # character values via the PH logarithm are multiplicative:
            # chi_k(y * y2) = chi_k(y) chi_k(y2) for an independent y2
            y2 = rng.randrange(1, p)
            x2 = pohlig_hellman(y2, g, p, Counter())
            x12 = pohlig_hellman(y * y2 % p, g, p, Counter())
            for k in (1, (p - 1) // 2, (p - 1) // B, p - 2):
                chi = lambda z: cmath.exp(2j * math.pi * k * z / (p - 1))
                ok &= abs(chi(x12) - chi(x) * chi(x2)) < 1e-9
            all_ok &= ok
            worst = max(worst, c.mults / bound_fine)
            max_mults = max(max_mults, c.mults)
        rep.check(f"p={p}: dlog/char correct (p-1={'*'.join(f'{q}^{e}' for q,e in fac.items())}, B={B})",
                  all_ok, f"{len(list(ys)) if p < 2000 else 300} inputs")
        rep.check(f"p={p}: mults <= {C_CONST}*sum e_i(log2 p + q_i)",
                  worst <= C_CONST,
                  f"max mults={max_mults}, max ratio={worst:.2f}, coarse bound={bound_coarse:.0f}")
        rep.check(f"p={p}: mults <= {C_CONST}*log2 p (log2 p + B)",
                  max_mults <= C_CONST * bound_coarse)
    rep.finish()


if __name__ == "__main__":
    main()
