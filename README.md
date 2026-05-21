# ClockEnsemble.jl

[![CI](https://github.com/ianlap/ClockEnsemble.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ianlap/ClockEnsemble.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/ianlap/ClockEnsemble.jl/actions/workflows/Documentation.yml/badge.svg)](https://ianlap.github.io/ClockEnsemble.jl/)

State-space models, Kalman filtering, and PID steering for precision
clocks. v0.1 ships the single-clock-steered-to-reference case;
ensemble combining (covariance-intersection, IMM,
Stein-pair recombination) will land in later 0.x releases under the
same package name.

## Install

```julia
using Pkg
Pkg.add("ClockEnsemble")
```

Requires Julia ≥ 1.11. Runtime deps: `LinearAlgebra`,
`StaticArrays`, `DocStringExtensions`. ClockEnsemble does **not**
depend on the sister [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl)
package, which handles stability-deviation analysis.

## Minimal example

```julia
using ClockEnsemble
using LinearAlgebra

# Three-state polynomial clock (phase, frequency, drift)
model = ThreeStateClock(tau=1.0, q0=1e-22, q1=1e-23, q2=1e-33, q3=1e-43)

# Kalman filter initialised at zero state, small covariance
kf = KalmanFilter(zeros(3), Matrix(1e-12 * I(3)))

# Closed-loop PID steering of the predicted frequency state
pid = PIDController(g_p=0.5, g_i=0.05, g_d=0.1)

for z in phase_measurements
    corr = steer_to_correction(pid.last_steer, nstates(model), model.tau)
    predict!(kf, model, model.tau; steering=corr)
    update!(kf, model, z)
    step!(pid, kf.x)
end
```

## Public surface

- `AbstractClockModel`, `TwoStateClock`, `ThreeStateClock`
- `nstates`, `state_transition`, `process_noise`,
  `measurement_matrix`, `measurement_noise`
- `KalmanFilter`, `predict!`, `update!`, `prop!`
- `PIDController`, `step!`, `steer_to_correction`

## Documentation

Hosted at [ianlap.github.io/ClockEnsemble.jl](https://ianlap.github.io/ClockEnsemble.jl/).
The `examples/` directory ships three runnable Literate.jl scripts
covering open-loop tracking, closed-loop PID steering, and a
holdover-budget comparison against TDEV / HTDEV.

## Sister package

For stability-deviation analysis (ADEV, MDEV, TDEV, HDEV, MTIE,
PDEV, totals, noise identification, EDF / χ² confidence intervals),
see [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl).

## License

MIT. See [LICENSE](LICENSE).
