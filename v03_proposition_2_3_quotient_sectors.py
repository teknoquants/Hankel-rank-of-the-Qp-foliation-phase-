# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Proposition 2.3 (what smoothness does buy: quotient sectors).

Statement.  Let Q | p - 1, H_Q = (F_p^*)^Q (index Q), zeta = g^{(p-1)/Q} and
lambda_Q(x) = log_zeta( x^{(p-1)/Q} ) in Z/Q (a homomorphism F_p^* -> Z/Q).
Define the quotient-sector sum

    S_n^[Q] = sum_{chi : chi^Q = 1} G(chi-bar)/(p-1) * Tr(B_chi^n).

Then
    S_n^[Q] = sum_{c in Z/Q} N_n(c) * eta_Q(c),
    eta_Q(c) = (Q/(p-1)) * sum_{t : lambda_Q(t) = c} psi(t)   (Gaussian periods),

where N_n(c) counts rooted closed NB walks of length n with lambda_Q(a-bar) = c.
N_n is computed by a transfer matrix of dimension d*Q whose entries need
lambda_Q(a(e,e')), obtainable by Pohlig--Hellman inside mu_Q in
O(log p + sum_{q^f || Q} f*(q + log Q)) multiplications.  The walk-dependent cost is
therefore poly(n, d, Q, B, log p); the Q Gaussian periods are a
walk-independent precomputation (O(p) by direct summation).  For Q = p - 1
the identity reduces to S_n itself.

Checks
------
1. Identity S^[Q] (character side) == coset transfer-matrix side, several Q.
2. Q = p - 1 reproduces the full brute-force S_n.
3. Subgroup Pohlig--Hellman labels agree with brute force and respect the
   operation-count bound 2 log2 p + 3 sum f (q + log2 Q).
"""
from __future__ import annotations

import cmath
import math
import random

import numpy as np

from common import (K4, Counter, Report, cyclic_product, factor, pohlig_hellman,
                    powmod, prim_root, psi, random_digits)


def label_PH(x, g, p, Q, c: Counter) -> int:
    """lambda_Q(x) via exponentiation + Pohlig--Hellman in the order-Q subgroup."""
    zeta = pow(g, (p - 1) // Q, p)
    h = powmod(x, (p - 1) // Q, p, c)
    comps = []
    for q, f in factor(Q).items():
        gam = pow(zeta, Q // q, p)  # order q
        zinv = pow(zeta, -1, p)
        xq = 0
        for j in range(f):
            z = h * powmod(zinv, xq, p, c) % p
            c.mults += 1
            hh = powmod(z, Q // q ** (j + 1), p, c)
            acc, dgt = 1, 0
            while acc != hh:
                acc = acc * gam % p
                c.mults += 1
                dgt += 1
            xq += dgt * q ** j
        comps.append((xq, q ** f))
    from common import crt
    return crt(comps) if comps else 0


def quotient_via_characters(core, a, p, g, Q, n, logs):
    total = 0
    step = (p - 1) // Q
    for k in range(0, p - 1, step):  # exactly the chi with chi^Q = 1
        B = np.zeros((core.d, core.d), dtype=complex)
        for (e, f), x in a.items():
            B[e, f] = cmath.exp(2j * math.pi * k * logs[x] / (p - 1))
        kbar = (-k) % (p - 1)
        G = sum(cmath.exp(2j * math.pi * kbar * logs[t] / (p - 1)) * psi(t, p) for t in range(1, p))
        total += G / (p - 1) * np.trace(np.linalg.matrix_power(B, n))
    return total


def quotient_via_transfer(core, a, p, g, Q, n, labels_all):
    d = core.d
    T = np.zeros((d * Q, d * Q))
    for (e, f), x in a.items():
        lx = labels_all[x]
        for c in range(Q):
            T[e * Q + c, f * Q + (c + lx) % Q] = 1
    Tn = np.linalg.matrix_power(T, n)
    N = np.array([sum(Tn[e * Q, e * Q + c] for e in range(d)) for c in range(Q)])
    eta = np.zeros(Q, dtype=complex)
    for t in range(1, p):
        eta[labels_all[t]] += psi(t, p)
    eta *= Q / (p - 1)
    return float(N.sum()), complex((N * eta).sum())


def main() -> None:
    rep = Report("Proposition 2.3 -- quotient sectors under smooth p - 1")
    rng = random.Random(2303)
    core = K4()
    for p, Qs in ((97, [2, 3, 4, 6, 8, 12, 16, 32, 48, 96]),
                  (7681, [2, 3, 5, 8, 15, 30, 64])):
        g = prim_root(p)
        logs = {x: pohlig_hellman(x, g, p, Counter()) for x in range(1, p)}
        a = random_digits(core, p, rng)
        for Q in Qs:
            labels = {x: logs[x] % Q for x in range(1, p)}
            # 3. PH labels and op count
            okL, worst = True, 0
            for x in list(a.values()) + [rng.randrange(1, p) for _ in range(50)]:
                c = Counter()
                lx = label_PH(x, g, p, Q, c)
                okL &= lx == labels[x]
                bound = 2 * math.log2(p) + 3 * sum(f * (q + math.log2(Q)) for q, f in factor(Q).items())
                worst = max(worst, c.mults / bound)
            rep.check(f"p={p}, Q={Q}: lambda_Q via subgroup PH correct, mults <= 2 log2 p + 3 sum f(q + log2 Q)",
                      okL and worst <= 1, f"max ratio={worst:.2f}")
            for n in (4, 6):
                lhs = quotient_via_characters(core, a, p, g, Q, n, logs)
                count, rhs = quotient_via_transfer(core, a, p, g, Q, n, labels)
                rep.check(f"p={p}, Q={Q}, n={n}: S^[Q] characters == dQ-transfer (dim {core.d * Q})",
                          abs(lhs - rhs) < 1e-6 * max(1, abs(lhs)), f"|diff|={abs(lhs - rhs):.1e}")
        if p == 97:
            Q = p - 1
            labels = {x: logs[x] % Q for x in range(1, p)}
            for n in (4, 6):
                brute = sum(psi(cyclic_product(w, a, p), p) for w in core.closed_words(n))
                _, rhs = quotient_via_transfer(core, a, p, g, Q, n, labels)
                rep.check(f"p=97, Q=p-1, n={n}: reduces to full S_n", abs(brute - rhs) < 1e-6 * max(1, abs(brute)))
    rep.finish()


if __name__ == "__main__":
    main()
