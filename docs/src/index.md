# ClockEnsemble.jl

State-space models, Kalman filtering, and PID steering for precision
clocks. ClockEnsemble.jl exposes the polynomial clock SDE
([`TwoStateClock`](@ref), [`ThreeStateClock`](@ref)), a
zero-allocation [`KalmanFilter`](@ref) built on `StaticArrays`, and a
discrete [`PIDController`](@ref) for closed-loop steering of a
single clock to a reference.

The v0.1 surface covers the single-clock-steered-to-reference case.
Future versions will reintroduce ensemble combining (covariance-
intersection, Stein-pair recombination, IMM) under the same package
name without an API rename.

## Installation

```julia
using Pkg
Pkg.add("ClockEnsemble")
```

The package targets Julia ≥ 1.11 and depends only on `LinearAlgebra`,
`StaticArrays`, and `DocStringExtensions`. It does **not** depend on
[SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl) — that sister
package handles stability-deviation analysis (ADEV, MDEV, TDEV,
HDEV, MTIE, PDEV, totals, etc.) and consumes the same kind of phase
record from a complementary angle.

## Minimal example

```julia
using ClockEnsemble
using LinearAlgebra

model = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)
kf    = KalmanFilter(zeros(3), Matrix(1e-12 * I(3)))

for z in phase_measurements
    predict!(kf, model, model.tau)
    update!(kf, model, z)
end
```

See the [Tutorials](tutorials/01_kalman_single_clock.md) for full
worked examples including PID steering and holdover-budget projection.

## See also

- [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl) — Allan,
  Hadamard, total, and parabolic deviations; noise identification;
  Greenhall–Riley EDF / χ² confidence intervals.
- Source repository:
  [github.com/ianlap/ClockEnsemble.jl](https://github.com/ianlap/ClockEnsemble.jl)
