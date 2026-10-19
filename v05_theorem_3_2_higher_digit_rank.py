# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Theorem 3.2 (higher-digit Hankel rank) and Conjecture 3.3.

Let G_m = (Z/p^{m+1})^x, f_m(X) = psi(digit_m X) and
rho_m(p) = #{ chi in G_m^ : f_m^(chi) != 0 },  f_m^(chi) = sum_X f_m(X) chi-bar(X).

Theorem 3.2.  Let 0 <= m < t_min and assume Hypothesis (H_m).  Then the Hankel
matrix of phi_m = psi o alpha_m has rank >= rho_m(p), so every edge-by-edge
transfer-matrix representation over C of phi_m has dimension >= rho_m(p), and
  (a) rho_0 = p - 1;
  (b) m >= 1: f_m^(chi) = 0 for every imprimitive chi, so rho_m <= p^{m-1}(p-1)^2;
  (c) m >= 1, chi primitive with chi(1 + p^m s) = e(c_chi s / p):
          f_m^(chi) = p * sum_{0 <= x < p^m, x == c_chi (mod p)} chi-bar(x);
      for m = 1 this has modulus p, hence rho_1 = (p-1)^2 exactly;
  (d) m >= 1: rho_m >= (p-1)^2 (Parseval).
Conjecture 3.3.  rho_m(p) = p^{m-1}(p-1)^2 for every odd prime p and m >= 1.

Checks
------
1. Rank of the group matrix [f_m(XY)] equals rho_m (direct SVD, small cases).
2. FFT over G_m (p <= 31, m <= 3): imprimitive coefficients vanish (b);
   support == primitive characters (Conjecture 3.3 evidence); |f^| = p for
   m = 1 (c); Parseval identity and bound (d).
3. Formula (c) by direct summation (p <= 13, m <= 2).
4. End-to-end on K4 with spectral p-adic edge matrices (t_e = m+1): Hankel
   submatrix built from the true eigenvalue digits alpha_m(uv), coverage of
   G_m (Hypothesis (H_m)) and rank == rho_m, for (p, m) in {(3,0),(3,1),(5,0),(5,1)}.
"""
from __future__ import annotations

import cmath
import math
import random

import numpy as np

from common import (K4, Report, W_open, digit, dominant_root, gamma, prim_root_pk,
                    rank, spectral_edges, tr, det)


def group_data(p, m):
    Nm = p ** (m + 1)
    phi = p ** m * (p - 1)
    g = prim_root_pk(p)
    elems = np.empty(phi, dtype=np.int64)
    lg = {}
    x = 1
    for a in range(phi):
        elems[a] = x
        lg[x] = a
        x = x * g % Nm
    return Nm, phi, elems, lg


def fhat_fft(p, m):
    Nm, phi, elems, lg = group_data(p, m)
    f = np.exp(2j * np.pi * ((elems // p ** m) % p) / p)
    # f^(chi_k) = sum_a f(g^a) e(-k a / phi)
    return np.fft.fft(f), phi, lg


def main() -> None:
    rep = Report("Theorem 3.2 / Conjecture 3.3 -- higher-digit Hankel rank")

    def rho_formula(p, m):
        return p - 1 if m == 0 else p ** (m - 1) * (p - 1) ** 2

    # 1. direct rank of the group matrix
    for p, m in ((3, 0), (3, 1), (3, 2), (5, 0), (5, 1), (5, 2), (7, 1), (7, 2), (11, 1)):
        Nm = p ** (m + 1)
        U = [x for x in range(1, Nm) if x % p]
        M = np.array([[cmath.exp(2j * math.pi * digit(x * y % Nm, m, p) / p) for y in U] for x in U])
        r = rank(M)
        rep.check(f"p={p}, m={m}: rank [f_m(XY)] over G_m = {rho_formula(p, m)}",
                  r == rho_formula(p, m), f"rank={r}, |G_m|={len(U)}")

    # 2. FFT support, (b), (c) modulus for m=1, (d) Parseval
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        for m in (1, 2, 3):
            if p ** m * (p - 1) > 1_000_000:
                continue
            F, phi, _ = fhat_fft(p, m)
            k = np.arange(phi)
            prim = (k % p) != 0
            scale = np.abs(F).max()
            imprim_zero = np.all(np.abs(F[~prim]) < 1e-7 * scale)
            support_eq = np.all(np.abs(F[prim]) > 1e-7 * scale)
            pars = abs((np.abs(F) ** 2).sum() - phi ** 2) < 1e-6 * phi ** 2
            nz = int((np.abs(F) > 1e-7 * scale).sum())
            ok = imprim_zero and support_eq and pars and nz == rho_formula(p, m) and nz >= (p - 1) ** 2
            extra = ""
            if m == 1:
                ok &= np.all(np.abs(np.abs(F[prim]) - p) < 1e-7 * p)
                extra = ", |f^|=p on primitives"
            rep.check(f"p={p}, m={m}: (b) imprimitive -> 0; support = primitive; Parseval{extra}",
                      ok, f"rho={nz}, min|f^|/p={np.abs(F[prim]).min() / p:.3f}")

    # 3. formula (c) by direct summation
    for p in (3, 5, 7, 11, 13):
        for m in (1, 2):
            Nm, phi, elems, lg = group_data(p, m)
            F, _, _ = fhat_fft(p, m)
            ok = True
            one_plus = lg[(1 + p ** m) % Nm]
            for kk in range(phi):
                if kk % p == 0:
                    continue
                # chi_k(1 + p^m) = e(k * log(1+p^m) / phi) = e(c/p)  ->  c
                c = round((kk * one_plus % phi) * p / phi) % p
                s = sum(cmath.exp(-2j * math.pi * kk * lg[x] / phi)
                        for x in range(p ** m) if x % p == c)
                ok &= abs(p * s - F[kk]) < 1e-6 * p ** (m + 1)
            rep.check(f"p={p}, m={m}: (c) f^(chi) = p * sum_(x<p^m, x=c_chi mod p) chi-bar(x)", ok)

    # 4. end-to-end on K4 with true eigenvalue digits
    rng = random.Random(3202)
    core = K4()
    K = 10
    for p, m in ((3, 0), (3, 1), (5, 0), (5, 1)):
        Nm = p ** (m + 1)
        N = p ** K
        t = [m + 1] * core.d
        data = spectral_edges(core, p, K, t, rng)
        e0 = 0
        e_star = core.succ[core.succ[e0][0]][0]
        e_prime = core.succ[e_star][0]
        e1 = [e for e in range(core.d) if core.allowed(e, e0) and e != e_prime][0]
        pre, suf = {}, {}
        for L in range(1, 16):
            for w in core.nb_words(L, e0):
                if w[-1] == e_star:
                    pre.setdefault(W_open(w, data, N) % Nm, w)
            for w in core.nb_words(L, e_prime):
                if w[-1] == e1:
                    suf.setdefault(W_open(w, data, N) % Nm, w)
            target = p ** m * (p - 1)
            if len(pre) == target and len(suf) == target:
                break
        U, V = list(pre.values()), list(suf.values())

        def alpha(word):
            G = gamma(word, data, N)
            r = dominant_root(tr(G, N), det(G, N), p, K)
            return digit(r, m, p)

        H = np.array([[cmath.exp(2j * math.pi * alpha(u + v) / p) for v in V] for u in U])
        r = rank(H)
        rep.check(f"K4 end-to-end p={p}, m={m}: (H_m) coverage of G_m",
                  len(pre) == len(suf) == p ** m * (p - 1), f"|X|={len(pre)}, |Y|={len(suf)}, L<={L}")
        rep.check(f"K4 end-to-end p={p}, m={m}: Hankel rank from true alpha_m == rho_m = {rho_formula(p, m)}",
                  r == rho_formula(p, m), f"rank={r}")
    rep.finish()


if __name__ == "__main__":
    main()
