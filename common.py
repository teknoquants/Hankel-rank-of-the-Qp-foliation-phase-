# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
"""Shared helpers for the Qp-Foliation validation suite.

Contents
--------
* elementary number theory (factorisation, primitive roots, discrete logs,
  instrumented modular exponentiation, Pohlig--Hellman);
* additive / multiplicative characters and Gauss sums over F_p;
* the test core X = K4 with its 12 oriented edges and non-backtracking (NB)
  transition structure;
* exact p-adic arithmetic in Z/p^K (Python integers) for 2x2 edge matrices
  in the spectral contraction form of Axiom III;
* a tiny PASS/FAIL reporter used by every script.

Only the Python standard library and NumPy are required.
"""
from __future__ import annotations

import cmath
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


class Report:
    """Collects named checks; exits non-zero if any check failed."""

    def __init__(self, title: str):
        self.title = title
        self.results: List[Tuple[str, bool, str]] = []
        print("=" * 78)
        print(title)
        print("=" * 78)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.results.append((name, ok, detail))
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))
        return ok

    def finish(self) -> None:
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print("-" * 78)
        print(f"{len(self.results) - n_fail}/{len(self.results)} checks passed")
        sys.exit(1 if n_fail else 0)


# ----------------------------------------------------------------------------
# Elementary number theory
# ----------------------------------------------------------------------------


def factor(n: int) -> Dict[int, int]:
    f: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def is_prime(n: int) -> bool:
    return n >= 2 and factor(n) == {n: 1}


def is_smooth(n: int, B: int) -> bool:
    return all(q <= B for q in factor(n))


def smoothness(n: int) -> int:
    return max(factor(n)) if n > 1 else 1


def prim_root(p: int) -> int:
    """Smallest primitive root modulo an odd prime p."""
    qs = factor(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in qs):
            return g
    raise ValueError("no primitive root")


def prim_root_pk(p: int) -> int:
    """A primitive root modulo p^k for every k >= 1 (p odd)."""
    g = prim_root(p)
    return g if pow(g, p - 1, p * p) != 1 else g + p


def dlog_table(p: int, g: int) -> Dict[int, int]:
    tab, x = {}, 1
    for k in range(p - 1):
        tab[x] = k
        x = x * g % p
    return tab


class Counter:
    """Counts modular multiplications (the 'field operations' of Lemma 2.1)."""

    def __init__(self) -> None:
        self.mults = 0


def powmod(b: int, e: int, m: int, c: Counter) -> int:
    """Square-and-multiply with an operation counter."""
    r = 1
    b %= m
    while e > 0:
        if e & 1:
            r = r * b % m
            c.mults += 1
        e >>= 1
        if e:
            b = b * b % m
            c.mults += 1
    return r


def crt(residues: Sequence[Tuple[int, int]]) -> int:
    x, M = 0, 1
    for r, m in residues:
        # solve x' = x (mod M), x' = r (mod m)
        t = ((r - x) * pow(M, -1, m)) % m
        x += M * t
        M *= m
    return x % M


def pohlig_hellman(y: int, g: int, p: int, c: Counter) -> int:
    """Discrete log of y to base g modulo p via Pohlig--Hellman.

    Each prime-power component q^e is solved digit by digit; every digit is
    found by exhaustive search in the subgroup of order q (O(q) multiplications)
    after one exponentiation (O(log p) multiplications).
    """
    n = p - 1
    g_inv = pow(g, -1, p)
    comps = []
    for q, e in factor(n).items():
        gamma = powmod(g, n // q, p, c)  # element of order q
        xq = 0
        for j in range(e):
            z = y * powmod(g_inv, xq, p, c) % p
            c.mults += 1
            h = powmod(z, n // q ** (j + 1), p, c)
            # exhaustive search for d with gamma^d = h, d in [0, q)
            acc, d = 1, 0
            while acc != h:
                acc = acc * gamma % p
                c.mults += 1
                d += 1
                if d > q:
                    raise RuntimeError("digit search failed")
            xq += d * q ** j
        comps.append((xq, q ** e))
    return crt(comps)


# ----------------------------------------------------------------------------
# Characters and Gauss sums over F_p
# ----------------------------------------------------------------------------


def psi(x: int, p: int) -> complex:
    """Canonical additive character psi(x) = exp(2 pi i x / p)."""
    return cmath.exp(2j * math.pi * (x % p) / p)


def mult_char_table(p: int, g: int) -> np.ndarray:
    """chi[k, x] = chi_k(x) for x in F_p (chi_k(0) := 0), chi_k(g) = e(k/(p-1))."""
    lg = dlog_table(p, g)
    tab = np.zeros((p - 1, p), dtype=complex)
    for x in range(1, p):
        for k in range(p - 1):
            tab[k, x] = cmath.exp(2j * math.pi * k * lg[x] / (p - 1))
    return tab


def gauss_sum(chi_row: np.ndarray, p: int) -> complex:
    """G(chi) = sum_{t in F_p^*} chi(t) psi(t)."""
    return sum(chi_row[t] * psi(t, p) for t in range(1, p))


# ----------------------------------------------------------------------------
# The test core X = K4 and its non-backtracking structure
# ----------------------------------------------------------------------------


@dataclass
class Core:
    """A finite graph given by its oriented edges, with NB transitions."""

    edges: List[Tuple[int, int]]
    index: Dict[Tuple[int, int], int] = field(init=False)
    succ: List[List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {e: i for i, e in enumerate(self.edges)}
        self.succ = []
        for (a, b) in self.edges:
            self.succ.append([self.index[(b, c)] for (bb, c) in self.edges
                              if bb == b and c != a])

    @property
    def d(self) -> int:
        return len(self.edges)

    def allowed(self, e: int, f: int) -> bool:
        return f in self.succ[e]

    def transitions(self) -> List[Tuple[int, int]]:
        return [(e, f) for e in range(self.d) for f in self.succ[e]]

    def nb_words(self, length: int, start: int | None = None):
        """All NB words (open) of a given length, optionally with fixed start."""
        starts = range(self.d) if start is None else [start]
        stack = [[s] for s in starts]
        while stack:
            w = stack.pop()
            if len(w) == length:
                yield tuple(w)
                continue
            for f in self.succ[w[-1]]:
                stack.append(w + [f])

    def closed_words(self, length: int):
        """Rooted closed NB words: every cyclic transition (incl. last->first) is NB."""
        for w in self.nb_words(length):
            if self.allowed(w[-1], w[0]):
                yield w


def K4() -> Core:
    return Core([(i, j) for i in range(4) for j in range(4) if i != j])


def random_digits(core: Core, p: int, rng: random.Random) -> Dict[Tuple[int, int], int]:
    """Random junction digits a(e,e') in F_p^* on every NB transition."""
    return {t: rng.randrange(1, p) for t in core.transitions()}


def cyclic_product(word: Sequence[int], a: Dict[Tuple[int, int], int], mod: int) -> int:
    x = 1
    n = len(word)
    for k in range(n):
        x = x * a[(word[k], word[(k + 1) % n])] % mod
    return x


def open_product(word: Sequence[int], a: Dict[Tuple[int, int], int], mod: int) -> int:
    x = 1
    for k in range(len(word) - 1):
        x = x * a[(word[k], word[k + 1])] % mod
    return x


# ----------------------------------------------------------------------------
# p-adic arithmetic in Z/p^K and spectral edge matrices (Axiom III)
# ----------------------------------------------------------------------------


def val(x: int, p: int, K: int) -> int:
    """p-adic valuation of x in Z/p^K (returns K for x == 0 mod p^K)."""
    x %= p ** K
    if x == 0:
        return K
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def digit(x: int, m: int, p: int) -> int:
    """m-th p-adic digit of a non-negative representative."""
    return (x // p ** m) % p


Mat = Tuple[Tuple[int, int], Tuple[int, int]]


def mat_mul(A: Mat, B: Mat, N: int) -> Mat:
    return (((A[0][0] * B[0][0] + A[0][1] * B[1][0]) % N,
             (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % N),
            ((A[1][0] * B[0][0] + A[1][1] * B[1][0]) % N,
             (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % N))


def mat_add(A: Mat, B: Mat, N: int) -> Mat:
    return tuple(tuple((A[i][j] + B[i][j]) % N for j in range(2)) for i in range(2))  # type: ignore


def mat_scale(s: int, A: Mat, N: int) -> Mat:
    return tuple(tuple(s * A[i][j] % N for j in range(2)) for i in range(2))  # type: ignore


def outer(u: Tuple[int, int], v: Tuple[int, int], N: int) -> Mat:
    return ((u[0] * v[0] % N, u[0] * v[1] % N), (u[1] * v[0] % N, u[1] * v[1] % N))


def tr(A: Mat, N: int) -> int:
    return (A[0][0] + A[1][1]) % N


def det(A: Mat, N: int) -> int:
    return (A[0][0] * A[1][1] - A[0][1] * A[1][0]) % N


IDENT: Mat = ((1, 0), (0, 1))


@dataclass
class EdgeData:
    u: Tuple[int, int]
    v: Tuple[int, int]
    mu: int
    t: int
    R: Mat
    nabla: Mat


def spectral_edges(core: Core, p: int, K: int, t: Sequence[int],
                   rng: random.Random, max_tries: int = 10000) -> List[EdgeData]:
    """Edge matrices nabla_e = u_e v_e^T + mu(e) R_e in *spectral* form.

    v_e^T u_e = 1, R_e = I - u_e v_e^T (complementary projector),
    mu(e) = p^{t_e} * unit.  Resamples until every NB junction weight
    w(e,e') = v_{e'}^T u_e is a p-adic unit (Axiom IV / non-degeneracy).
    """
    N = p ** K
    for _ in range(max_tries):
        data = []
        for e in range(core.d):
            u = (1, rng.randrange(N))
            v2 = rng.randrange(N)
            v1 = (1 - v2 * u[1]) % N
            v = (v1, v2)
            unit = rng.randrange(1, N)
            while unit % p == 0:
                unit = rng.randrange(1, N)
            mu = p ** t[e] * unit % N
            P = outer(u, v, N)
            R = mat_add(IDENT, mat_scale(-1, P, N), N)
            nabla = mat_add(P, mat_scale(mu, R, N), N)
            data.append(EdgeData(u, v, mu, t[e], R, nabla))
        if all(w_weight(data, e, f, N) % p for (e, f) in core.transitions()):
            return data
    raise RuntimeError("could not sample unit junction weights")


def w_weight(data: List[EdgeData], e: int, f: int, N: int) -> int:
    """w(e, e') = v_{e'}^T u_e."""
    return (data[f].v[0] * data[e].u[0] + data[f].v[1] * data[e].u[1]) % N


def gamma(word: Sequence[int], data: List[EdgeData], N: int) -> Mat:
    """gamma_delta = nabla_{e_n} ... nabla_{e_1}."""
    G = IDENT
    for e in word:
        G = mat_mul(data[e].nabla, G, N)
    return G


def W_cyclic(word: Sequence[int], data: List[EdgeData], N: int) -> int:
    x, n = 1, len(word)
    for k in range(n):
        x = x * w_weight(data, word[k], word[(k + 1) % n], N) % N
    return x


def W_open(word: Sequence[int], data: List[EdgeData], N: int) -> int:
    x = 1
    for k in range(len(word) - 1):
        x = x * w_weight(data, word[k], word[k + 1], N) % N
    return x


def dominant_root(T: int, D: int, p: int, K: int) -> int:
    """Unit root r of x^2 - T x + D in Z/p^K, assuming T unit and val(D) >= 1.

    Uses the contracting fixed point r <- T - D / r (Hensel / Newton-free).
    """
    N = p ** K
    if T % p == 0:
        raise ValueError("trace is not a unit")
    r = T % N
    for _ in range(K + 2):
        r = (T - D * pow(r, -1, N)) % N
    assert (r * r - T * r + D) % N == 0
    return r


def rank(M: np.ndarray, rtol: float = 1e-9) -> int:
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > rtol * s[0]).sum()) if s.size and s[0] > 0 else 0
