# Smoothness-Invariant Hankel Rank, Primitive-Character Bounds for Higher p-adic Digits, and Quadratic Valuation-Clock Estimation on Qp-Foliations

*Three structural extensions of the Qp-Foliation Interference Theorem, with machine validation*
Roger Cooperman · A.E. Petrunina V.V. · 2026

- **This white paper:** DOI `10.XXXX/XXXXX` *(to be assigned)*
- **Qp-Foliation Interference Theorem (Cooperman & Petrunina, 2026):** DOI `10.XXXX/XXXXX`
- **Repository:** `https://github.com/<OWNER>/<REPOSITORY>`

This repository contains the white paper and an MIT-licensed Python suite that
checks every lemma, theorem, proposition and conjecture in it. There is one
script per statement, and the full suite runs **214 checks**.

## Main results

1. **Smoothness invariance.** In the transfer-matrix model the Hankel rank of
   ψ ∘ ā is exactly p − 1 for every prime p. Smooth p − 1 makes the
   quotient-sector sums cheap, with transfer dimension dQ, but does not reduce
   the number of sectors.
2. **Higher p-adic digits.** For m < t_min, the m-th digit phase has Hankel rank
   ≥ ρ_m(p), and the following are proved:
   - ρ₁(p) = (p − 1)²;
   - (p − 1)² ≤ ρ_m(p) ≤ p^{m−1}(p − 1)²;
   - equality ρ_m(p) = p^{m−1}(p − 1)² is conjectured, and verified for all
     p ≤ 31, m ≤ 3.
3. **Valuation clocks.** S(L_m)/N is estimated to error ε with O(1/ε) oracle
   calls, a quadratic gain over classical sampling.

## Quick start

```bash
pip install -r requirements.txt
python validation/run_all.py      # exit status 0 = all checks pass (< 15 s)
```

## Statement ↔ script map

| Paper | Statement | Script | Checks |
|---|---|---|---|
| Lemma 2.1 | Pohlig–Hellman character reduction, O(log p (log p + B)) | `v01_lemma_2_1_pohlig_hellman.py` | 30 |
| Theorem 2.2 | Smoothness invariance of the sector bottleneck | `v02_theorem_2_2_smoothness_invariance.py` | 26 |
| Proposition 2.3 | Quotient sectors: transfer dimension dQ | `v03_proposition_2_3_quotient_sectors.py` | 53 |
| Lemma 3.1 | Higher-digit expansion and multiplicative carries | `v04_lemma_3_1_carry_expansion.py` | 21 |
| Theorem 3.2, Conjecture 3.3 | Higher-digit Hankel rank ρ_m(p) | `v05_theorem_3_2_higher_digit_rank.py` | 57 |
| Lemma 4.1 | Valuation additivity under spectral normalisation | `v06_lemma_4_1_valuation_additivity.py` | 16 |
| Theorem 4.2 | Valuation-clock leaf estimator | `v07_theorem_4_2_valuation_clock.py` | 11 |

`validation/common.py` holds the shared code: number theory, characters and
Gauss sums, the K4 test core with its non-backtracking structure, and exact
p-adic 2×2 arithmetic in Z/p^K.

## Scope

All lower bounds are for **transfer-matrix (weighted-automaton) representations**,
via the Fliess / Carlyle–Paz theorem. They are not lower bounds against
arbitrary classical algorithms. Numerical checks are evidence, not proofs:
the proofs are in the paper.

## Layout

```
paper/Qp_Foliations_Hankel_Rank_Valuation_Clocks.docx   white paper
validation/common.py                                     shared helpers
validation/v0*_*.py                                      one script per statement
validation/run_all.py                                    runs everything, prints summary
LICENSE                                                  MIT
```

## How to cite

Cooperman, R., Petrunina, A. E. (2026). *Smoothness-Invariant Hankel Rank,
Primitive-Character Bounds for Higher p-adic Digits, and Quadratic
Valuation-Clock Estimation on Qp-Foliations.* White paper. DOI: 10.XXXX/XXXXX.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Roger Cooperman and A.E. Petrunina V.V.
