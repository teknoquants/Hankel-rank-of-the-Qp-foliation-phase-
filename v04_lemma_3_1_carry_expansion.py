# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Lemma 3.1 (higher-digit expansion and carry structure).

Setting (Axioms III'-IV as restated in the paper): 2x2 edge matrices over Z_p
in spectral contraction form
    nabla_e = u_e v_e^T + mu(e) R_e,  v_e^T u_e = 1,  R_e = I - u_e v_e^T,
    mu(e) in p^{t_e} Z_p^x (t_e >= 1),  w(e,e') = v_{e'}^T u_e in Z_p^x.
For a closed NB word delta = (e_1..e_n): gamma_delta = nabla_{e_n}...nabla_{e_1},
W(delta) = prod_k w(e_k, e_{k+1}) (cyclic), T(delta) = sum_k t_{e_k},
t_min = min_e t_e, r(delta) = unit eigenvalue of gamma_delta = sum_j alpha_j p^j.

 (i)   Tr gamma = sum_{sigma subset [n]} Phi_sigma with Phi_empty = W(delta)
       and val(Phi_sigma) >= t(sigma) := sum_{k in sigma} t_{e_k}.
 (ii)  det gamma = prod mu(e_k); r exists, is unique, r == Tr gamma (mod p^T)
       and val(r - Tr gamma) = T exactly.
 (iii) For m < T: alpha_m = m-th digit of sum_{t(sigma) <= m} Phi_sigma (mod p^{m+1}).
       In particular for m < t_min: alpha_m = m-th digit of W(delta).
 (iv)  Carries come from multiplication: digit_1(xy) = x0 y1 + x1 y0 + floor(x0 y0/p)
       (mod p); digit_m(xy) is not a function of (digit_m x, digit_m y).

Also checks that for m >= t_min the R-corrections genuinely change alpha_m
(so the cut-off in (iii) is sharp).
"""
from __future__ import annotations

import itertools
import random

from common import (IDENT, K4, Report, W_cyclic, det, digit, dominant_root, gamma,
                    mat_mul, mat_scale, outer, spectral_edges, tr, val)

K = 16


def phi_sigma(word, data, sigma, N):
    M = IDENT
    mu = 1
    for k, e in enumerate(word):
        if k in sigma:
            F = data[e].R
            mu = mu * data[e].mu % N
        else:
            F = outer(data[e].u, data[e].v, N)
        M = mat_mul(F, M, N)
    return mu * tr(M, N) % N


def main() -> None:
    rep = Report("Lemma 3.1 -- higher-digit expansion and carries")
    rng = random.Random(3101)
    core = K4()
    for p in (3, 5, 7):
        N = p ** K
        ok_i = ok_ii = ok_iii = ok_iiib = True
        n_cases = 0
        r_terms_matter = 0
        for trial in range(6):
            t = [rng.choice((1, 2, 3)) for _ in range(core.d)]
            data = spectral_edges(core, p, K, t, rng)
            tmin = min(t)
            words = [w for n in (3, 4, 5, 6) for w in core.closed_words(n)]
            for w in rng.sample(words, 60):
                n_cases += 1
                G = gamma(w, data, N)
                Wd = W_cyclic(w, data, N)
                T = sum(t[e] for e in w)
                # (i) full expansion and valuations
                phis = {}
                for s in range(len(w) + 1):
                    for sigma in itertools.combinations(range(len(w)), s):
                        ph = phi_sigma(w, data, set(sigma), N)
                        phis[sigma] = ph
                        ok_i &= val(ph, p, K) >= min(K, sum(t[w[k]] for k in sigma))
                ok_i &= sum(phis.values()) % N == tr(G, N) and phis[()] == Wd
                # (ii)
                D = det(G, N)
                prod_mu = 1
                for e in w:
                    prod_mu = prod_mu * data[e].mu % N
                r = dominant_root(tr(G, N), D, p, K)
                ok_ii &= D == prod_mu and val(D, p, K) == min(T, K)
                ok_ii &= val(r - tr(G, N), p, K) == min(T, K)
                # (iii)
                for m in range(0, min(T, K - 1)):
                    trunc = sum(ph for sg, ph in phis.items()
                                if sum(t[w[k]] for k in sg) <= m) % p ** (m + 1)
                    ok_iii &= digit(r, m, p) == digit(trunc, m, p)
                    if m < tmin:
                        ok_iiib &= digit(r, m, p) == digit(Wd, m, p)
                    elif digit(r, m, p) != digit(Wd, m, p):
                        r_terms_matter += 1
        rep.check(f"p={p}: (i) Tr gamma = sum_sigma Phi_sigma, Phi_empty = W, val(Phi_sigma) >= t(sigma)",
                  ok_i, f"{n_cases} closed words, n=3..6")
        rep.check(f"p={p}: (ii) det = prod mu, val(r - Tr) = T exactly", ok_ii)
        rep.check(f"p={p}: (iii) alpha_m = digit_m of truncated expansion (t(sigma) <= m), all m < T", ok_iii)
        rep.check(f"p={p}: (iii) m < t_min  =>  alpha_m = digit_m(W(delta))", ok_iiib)
        rep.check(f"p={p}: cut-off is sharp (R-terms change alpha_m for some m >= t_min)",
                  r_terms_matter > 0, f"{r_terms_matter} (word, m) pairs")

        # (iv) carry formula and non-functionality
        ok_iv = True
        table = {}
        func = True
        for x in range(p * p):
            for y in range(p * p):
                if x % p == 0 or y % p == 0:
                    continue
                x0, x1, y0, y1 = x % p, x // p, y % p, y // p
                d1 = digit(x * y, 1, p)
                ok_iv &= d1 == (x0 * y1 + x1 * y0 + (x0 * y0) // p) % p
                key = (x1, y1)
                if key in table and table[key] != d1:
                    func = False
                table.setdefault(key, d1)
        rep.check(f"p={p}: (iv) digit_1(xy) = x0 y1 + x1 y0 + floor(x0 y0 / p) mod p", ok_iv)
        rep.check(f"p={p}: (iv) digit_1(xy) is not a function of (digit_1 x, digit_1 y)", not func)
    rep.finish()


if __name__ == "__main__":
    main()
