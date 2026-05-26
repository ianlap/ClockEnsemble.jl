# ClockEnsemble.jl — Roadmap

Working list of outstanding engineering work. Items move from this
file to `CHANGELOG.md` as soon as they land — every shipped change
should remove the matching entry here and add one under
`## [Unreleased]` in the changelog in the same commit.

> **Audit date**: 2026-05-21 (initial release, split from SigmaTau.jl
> v0.2.0).

---

## 🟡 Correctness / completeness

- [ ] **Flicker-FM Markov-chain approximation.** The current process-
  noise integration treats `σ1` as white-FM only. A two-state Markov
  approximation à la Galleani / Davis–Howe lets the filter ingest
  flicker-FM noise without inflating `σ2` to absorb it. Add as an
  optional channel on `ThreeStateClock` (gated on a kwarg) and
  benchmark against synthetic-flicker traces.
- [ ] **`fit_clock_params(phase_record)`** — closed-form / MLE
  fit of `(σ1, σ2, σ3)` from a measured phase record, optionally
  with a known measurement-noise floor `R`. Operationally the
  thing every user actually needs before running the filter — currently
  they hand-tune diffusion coefficients to match observed ADEV.
- [ ] **Joseph-form covariance update.** The current
  `posterior_cov(P, K, H) = (I − KH) P` is the textbook update.
  Joseph form `(I − KH) P (I − KH)' + K R K'` preserves
  symmetry and positive-semidefiniteness exactly under round-off
  and is the right default once R can have non-trivial entries.
  Gate behind a kwarg on `update!` (`form = :joseph`).

## 🟡 Future scope (ensemble reintroduction)

- [ ] **`ClockEnsemble` struct.** Reintroduce the (deleted in v0.2.0
  cleanup) ensemble container: a vector of `AbstractClockModel`s
  plus a covariance-intersection or KF-fusion step. v0.1 ships only
  the single-clock-steered-to-reference case; the package name
  anticipates this work.
- [ ] **Mixed-dimension ensemble support.** Allow a single
  `ClockEnsemble` to mix `TwoStateClock` and `ThreeStateClock`
  members — Φ becomes a block-diagonal with heterogeneous block
  sizes, and the cross-clock comparison logic has to handle
  per-clock state dimensions in the difference rows.
- [ ] **Interacting Multiple Model (IMM) filter.** For clocks that
  switch regimes (cold-start vs warm-up, drift-relock events). An
  IMM bank of two `ThreeStateClock`s with different `σ3` budgets
  handles the typical caesium-vs-rubidium step in a single filter.
- [ ] **Stein per-pair shock recombination.** Per Stein 2003, when
  combining `N` clocks into a paper timescale the natural unit is
  pairs rather than absolute references. Each pair carries its own
  shock budget; recombining them is the optimal-fusion step.
- [ ] **`RelativisticClock`.** Stub today (throws on dispatch). Add
  Schwarzschild gravitational-redshift and special-relativistic
  time-dilation terms to the propagator for satellite-clock work.
- [ ] **JSMD lunar-stack integration.** Long-horizon mission timing
  for lunar surface assets — extends `RelativisticClock` with a
  Moon-frame frequency offset and an Earth-Moon two-way link delay
  estimator.

## 🟢 Polish

- [ ] **`save_clock` / `load_clock`** — serialize `KalmanFilter`
  state (`x`, `P`, `k`) for warm-start resumption. Requires a
  clock-model-type tag in the file header to reconstruct the
  correct `SVector`/`SMatrix` type parameters on load.
- [ ] **More examples** — GP-based holdover prediction via
  `TemporalGPs.jl` (Matérn-1/2 kernel ≡ OU) as a fourth reference
  line alongside the TDEV / HTDEV / KF curves in
  `examples/03_holdover_comparison.jl`. UDE drift learning via
  `DiffEqFlux.jl` — show how a `TwoStateClock`'s `predict!`
  composes with a NeuralODE that learns the unmodelled drift
  residual. Don't take SciML as a dep; keep it confined to the
  example.
- [ ] **Compat upper bounds.** Tighten `StaticArrays` upper bound
  once the dep matrix has been exercised on the General registry.

---

## ✅ Recently shipped

See [CHANGELOG.md](CHANGELOG.md) for the annotated `## [0.1.0]` block.
