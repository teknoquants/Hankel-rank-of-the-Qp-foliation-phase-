# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Validation of Theorem 4.2 (valuation-clock leaf estimator).

Setting.  Edge exponents t_e >= 1 (non-uniform), leaf target m,
k_max = floor(m / t_min), P = all NB words of length 1..k_max, N = |P|.
f(delta) = 1 iff delta is closed (last -> first is NB) and T(delta) = m.
L_m = {f = 1},  S(L_m) = sum_{delta in L_m} psi(a-bar(delta)).

Theorem 4.2.
 (a) A unitary A preparing |Psi> = N^{-1/2} sum_{delta in P} |delta> is built
     from k_max + 1 layers of controlled rotations whose angles come from a
     classical table of NB-extension counts (size d * k_max).
 (b) With U = psi(y)^f (phase applied only when the flag is set),
         <Psi|U|Psi> = (S(L_m) + N - |L_m|) / N,   P(f = 1) = |L_m| / N,
     so S(L_m)/N = <U> - 1 + P(f=1).  (The phase step alone does NOT zero
     out paths outside the leaf.)
 (c) Amplitude estimation on the two Hadamard-test amplitudes and the flag
     amplitude estimates S(L_m)/N to additive error eps with probability
     >= 2/3 using O(1/eps) calls to A, U and their inverses.
 (d) Hence S(L_m) is estimated to additive error eps * N; classical sampling
     from P needs Theta(1/eps^2) samples for the same guarantee: the gain
     is quadratic.
 (e) The clock needs b = ceil(log2(k_max t_max + 1)) qubits (or a saturating
     adder).  ceil(log2 m) qubits wrap around and mis-flag walks.

Checks
------
1. (a) amplitudes produced by the DP rotation scheme are exactly N^{-1/2}.
2. (b) explicit Hadamard-test statevector simulation (real and imaginary
   parts) reproduces (S + N - |L|)/N; the combination <U> - 1 + P(f=1) gives S/N;
   <U> alone differs from S/N.
3. (c)/(d) exact canonical-AE outcome distributions: the Brassard et al.
   error bound holds with probability >= 8/pi^2; the full estimator meets
   |S^/N - S/N| <= 5(pi/M + pi^2/M^2) with probability >= 2/3; AE error
   scales like M^-1 versus M^-1/2 for classical Monte Carlo.
4. (e) wrap-around counterexample with ceil(log2 m) clock bits.
"""
from __future__ import annotations

import math
import random

import numpy as np

from common import K4, Report, cyclic_product, psi, random_digits


def build_instance(core, p, t, m, rng):
    a = random_digits(core, p, rng)
    tmin, tmax = min(t), max(t)
    kmax = m // tmin
    P = [w for L in range(1, kmax + 1) for w in core.nb_words(L)]
    return a, kmax, tmax, P


def dp_amplitudes(core, kmax, P):
    """State preparation by NB-extension counts; returns amplitude per word."""
    d = core.d
    # c[r][e] = number of NB words of length r starting at e (r >= 1)
    c = [[0] * d for _ in range(kmax + 1)]
    for e in range(d):
        c[1][e] = 1
    for r in range(2, kmax + 1):
        for e in range(d):
            c[r][e] = sum(c[r - 1][f] for f in core.succ[e])
    NL = [0] + [sum(c[L]) for L in range(1, kmax + 1)]
    Ntot = sum(NL)
    amp = {}
    for w in P:
        L = len(w)
        prob = NL[L] / Ntot                      # length register rotation
        prob *= c[L][w[0]] / NL[L]               # first edge
        for k in range(1, L):                    # each extension
            prob *= c[L - k][w[k]] / c[L - k + 1][w[k - 1]]
        amp[w] = math.sqrt(prob)
    return amp, Ntot


def registers(w, a, p, t, m, core, bits=None):
    clock = sum(t[e] for e in w)
    if bits is not None:
        clock %= 2 ** bits
    closed = core.allowed(w[-1], w[0])
    target = m if bits is None else m % 2 ** bits
    f = int(closed and clock == target)
    y = cyclic_product(w, a, p) if closed else 1
    return clock, y, f


def hadamard_test(amps, phases):
    """|0>|Psi> -H-> ctrl-U -H->; return P(anc=0) for U and for (S^dag) variant."""
    psi0 = np.array(amps, dtype=complex)
    ph = np.array(phases, dtype=complex)
    out = []
    for sgate in (1, -1j):
        branch0 = psi0 / math.sqrt(2)
        branch1 = sgate * ph * psi0 / math.sqrt(2)
        a0 = (branch0 + branch1) / math.sqrt(2)
        out.append(float(np.vdot(a0, a0).real))
    return out  # [ (1+Re<U>)/2 , (1+Im<U>)/2 ]


def ae_distribution(a, M):
    """Exact outcome distribution of canonical amplitude estimation (M points).

    Outcome y has probability (1/2)(|D(y/M - theta/pi)|^2 + |D(y/M + theta/pi)|^2),
    D(x) = (1/M) sum_j e^{2 pi i j x}, a = sin^2 theta  (Brassard et al. 2002).
    """
    theta = math.asin(math.sqrt(a))
    y = np.arange(M)
    probs = np.zeros(M)
    for sgn in (1, -1):
        x = y / M - sgn * theta / np.pi
        num = np.sin(np.pi * M * x) ** 2
        den = (M * np.sin(np.pi * x)) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(den < 1e-300, 1.0, num / np.where(den < 1e-300, 1, den))
        probs += 0.5 * val
    est = np.sin(np.pi * y / M) ** 2
    return probs / probs.sum(), est


def main() -> None:
    rep = Report("Theorem 4.2 -- valuation-clock leaf estimator")
    rng = random.Random(4202)
    core = K4()
    p, m = 5, 5
    t = [rng.choice((1, 2)) for _ in range(core.d)]
    a, kmax, tmax, P = build_instance(core, p, t, m, rng)
    N = len(P)

    # 1. state preparation
    amp, Ntot = dp_amplitudes(core, kmax, P)
    ok = Ntot == N and all(abs(v - 1 / math.sqrt(N)) < 1e-12 for v in amp.values())
    rep.check("(a) DP rotation scheme gives uniform amplitude N^-1/2 on all NB words",
              ok, f"N={N}, k_max={kmax}")

    # 2. Hadamard test
    regs = [registers(w, a, p, t, m, core) for w in P]
    L = [w for w, (_, _, f) in zip(P, regs) if f]
    S = sum(psi(cyclic_product(w, a, p), p) for w in L)
    phases = [psi(y, p) if f else 1 for (_, y, f) in regs]
    p_re, p_im = hadamard_test([amp[w] for w in P], phases)
    U_exp = complex(2 * p_re - 1, 2 * p_im - 1)
    pf = len(L) / N
    rep.check("(b) Hadamard test: <U> = (S + N - |L|)/N",
              abs(U_exp - (S + N - len(L)) / N) < 1e-12, f"|L|={len(L)}, S={S:.4f}")
    rep.check("(b) combination <U> - 1 + P(f=1) = S/N",
              abs(U_exp - 1 + pf - S / N) < 1e-12)
    rep.check("(b) <U> alone != S/N (paths outside the leaf are not zeroed)",
              abs(U_exp - S / N) > 1e-3, f"|<U> - S/N| = {abs(U_exp - S / N):.3f}")

    # 3. amplitude estimation
    targets = {"re": p_re, "im": p_im, "flag": pf}
    ok_bound = True
    for M in (16, 64, 256, 1024):
        for name, aval in targets.items():
            probs, est = ae_distribution(aval, M)
            bound = 2 * math.pi * math.sqrt(aval * (1 - aval)) / M + math.pi ** 2 / M ** 2
            succ = probs[np.abs(est - aval) <= bound + 1e-12].sum()
            ok_bound &= succ >= 8 / math.pi ** 2 - 1e-9
    rep.check("(c) Brassard et al. AE bound holds w.p. >= 8/pi^2 (Re, Im, flag; M=16..1024)", ok_bound)

    nrng = np.random.default_rng(42)
    Ms = [32, 64, 128, 256, 512, 1024, 2048]
    reps = 400
    ok_c = True
    worst_frac = 1.0
    mc_err = []
    vals = np.array([phases[i] * (1 if regs[i][2] else 0) for i in range(N)])
    for M in Ms:
        dists = {k: ae_distribution(v, M) for k, v in targets.items()}
        bound = 5 * (math.pi / M + math.pi ** 2 / M ** 2)  # 2*e_re + 2*e_im + e_flag
        hits = 0
        for _ in range(reps):
            draws = {}
            for k, (pr, est) in dists.items():
                draws[k] = np.median(est[nrng.choice(M, size=3, p=pr)])  # median of 3
            Shat = complex(2 * draws["re"] - 1, 2 * draws["im"] - 1) - 1 + draws["flag"]
            hits += abs(Shat - S / N) <= bound
        frac = hits / reps
        worst_frac = min(worst_frac, frac)
        ok_c &= frac >= 2 / 3
        # classical MC with the same number (3 x 3 x M) of oracle calls
        mc = [abs(vals[nrng.integers(0, N, size=9 * M)].mean() - S / N) for _ in range(reps)]
        mc_err.append(np.quantile(mc, 2 / 3))
    rep.check("(c) end-to-end: |S^/N - S/N| <= 5(pi/M + pi^2/M^2) w.p. >= 2/3, M=32..2048",
              ok_c, f"min success fraction={worst_frac:.2f}")

    # scaling over random amplitudes (removes grid-coincidence artefacts)
    ae_q = []
    arng = np.random.default_rng(7)
    avals = arng.uniform(0.02, 0.98, size=200)
    for M in Ms:
        qs = []
        for aval in avals:
            pr, est = ae_distribution(aval, M)
            err = np.abs(est - aval)
            o = np.argsort(err)
            qs.append(err[o][np.searchsorted(np.cumsum(pr[o]), 2 / 3)])
        ae_q.append(np.mean(qs))
    slope_ae = np.polyfit(np.log(Ms), np.log(ae_q), 1)[0]
    slope_mc = np.polyfit(np.log(Ms), np.log(mc_err), 1)[0]
    rep.check("(c) AE error ~ M^-1 (mean 2/3-quantile over 200 random amplitudes)",
              -1.15 < slope_ae < -0.85, f"fitted slope={slope_ae:.2f}")
    rep.check("(d) classical MC error ~ M^-1/2 (quadratic gap only)", -0.65 < slope_mc < -0.35,
              f"fitted slope={slope_mc:.2f}")
    Nk = [sum(1 for L in range(1, k + 1) for _ in core.nb_words(L)) for k in range(1, 9)]
    ratios = [Nk[i + 1] / Nk[i] for i in range(len(Nk) - 1)]
    rep.check("(d) N = |P| grows exponentially in k_max, so error eps*N on S is exponential",
              ratios[-1] > 1.9, f"N(k_max=1..8)={Nk}")

    # 4. clock wrap-around
    m2 = 4
    t2 = [1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 3, 3]  # t_min = 1 -> k_max = 4, T up to 12
    rng2 = random.Random(7)
    a2, kmax2, tmax2, P2 = build_instance(core, p, t2, m2, rng2)
    good_bits = math.ceil(math.log2(kmax2 * tmax2 + 1))
    bad_bits = math.ceil(math.log2(m2))
    L_true = {w for w in P2 if registers(w, a2, p, t2, m2, core)[2]}
    L_good = {w for w in P2 if registers(w, a2, p, t2, m2, core, bits=good_bits)[2]}
    L_bad = {w for w in P2 if registers(w, a2, p, t2, m2, core, bits=bad_bits)[2]}
    rep.check(f"(e) b = ceil(log2(k_max t_max + 1)) = {good_bits} bits: leaf exact",
              L_good == L_true, f"|L|={len(L_true)}")
    rep.check(f"(e) b = ceil(log2 m) = {bad_bits} bits: wrap-around mis-flags walks",
              len(L_bad - L_true) > 0, f"{len(L_bad - L_true)} false positives")
    rep.finish()


if __name__ == "__main__":
    main()
