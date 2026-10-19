# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Lemma 4.1 (valuation additivity) and its hypothesis.

Statement.  For any 2x2 matrices over Z_p of the form
nabla_e = u_e v_e^T + mu(e) R_e,

    det(nabla_e) = mu(e) * c_e + mu(e)^2 * det(R_e),
    c_e := tr( adj(u_e v_e^T) R_e ),

so val(det nabla_e) >= t_e in general, with equality iff c_e + mu(e) det R_e
is a unit.  Under the spectral normalisation (Axiom III'): v_e^T u_e = 1 and
R_e = I - u_e v_e^T, one has det(nabla_e) = mu(e) exactly, hence for every walk

    det gamma_delta = prod_k mu(e_k),   val(det gamma_delta) = sum_k t_{e_k}.

Checks
------
1. The general determinant identity on random integral data (exact mod p^K).
2. Spectral case: det nabla_e = mu(e) and walk additivity for random walks
   with non-uniform t_e.
3. Necessity of the hypothesis: explicit non-spectral R_e with
   val(det nabla_e) > t_e (so additivity in the stated form fails).
"""
from __future__ import annotations

import random

from common import (K4, Report, det, gamma, mat_add, mat_scale, outer,
                    spectral_edges, tr, val)

K = 14


def adj(A, N):
    return ((A[1][1] % N, -A[0][1] % N), (-A[1][0] % N, A[0][0] % N))


def trmul(A, B, N):
    return (A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[1][0] * B[0][1] + A[1][1] * B[1][1]) % N


def main() -> None:
    rep = Report("Lemma 4.1 -- valuation additivity")
    rng = random.Random(4101)
    core = K4()
    for p in (3, 5, 7, 11):
        N = p ** K
        # 1. general identity
        ok = True
        for _ in range(500):
            u = (rng.randrange(N), rng.randrange(N))
            v = (rng.randrange(N), rng.randrange(N))
            R = ((rng.randrange(N), rng.randrange(N)), (rng.randrange(N), rng.randrange(N)))
            t = rng.randrange(1, 4)
            mu = p ** t * rng.randrange(1, N) % N
            P = outer(u, v, N)
            nab = mat_add(P, mat_scale(mu, R, N), N)
            c = trmul(adj(P, N), R, N)
            ok &= det(nab, N) == (mu * c + mu * mu * det(R, N)) % N
            ok &= val(det(nab, N), p, K) >= min(t, K)
        rep.check(f"p={p}: det(uv^T + mu R) = mu tr(adj(uv^T) R) + mu^2 det R; val >= t", ok)

        # 2. spectral normalisation -> exact additivity
        t = [rng.choice((1, 2, 3, 4)) for _ in range(core.d)]
        data = spectral_edges(core, p, K, t, rng)
        ok_e = all(det(d.nabla, N) == d.mu for d in data)
        ok_w = True
        for n in (3, 4, 5, 6):
            for w in rng.sample(list(core.nb_words(n)), 40):
                G = gamma(w, data, N)
                pm = 1
                for e in w:
                    pm = pm * data[e].mu % N
                T = sum(t[e] for e in w)
                ok_w &= det(G, N) == pm and val(det(G, N), p, K) == min(T, K)
        rep.check(f"p={p}: spectral form => det nabla_e = mu(e) for all 12 edges", ok_e)
        rep.check(f"p={p}: spectral form => val(det gamma_delta) = sum t_e (non-uniform t)", ok_w)

        # 3. counterexample without the normalisation
        found = None
        for _ in range(5000):
            u = (1, rng.randrange(N))
            v = (rng.randrange(N), rng.randrange(N))
            R = ((rng.randrange(N), rng.randrange(N)), (rng.randrange(N), rng.randrange(N)))
            P = outer(u, v, N)
            if trmul(adj(P, N), R, N) % p == 0:
                mu = p * rng.randrange(1, p) % N  # t_e = 1
                nab = mat_add(P, mat_scale(mu, R, N), N)
                if val(det(nab, N), p, K) > 1:
                    found = val(det(nab, N), p, K)
                    break
        rep.check(f"p={p}: non-spectral R_e can give val(det nabla_e) > t_e (hypothesis needed)",
                  found is not None, f"t_e=1, val(det)={found}")
    rep.finish()


if __name__ == "__main__":
    main()
