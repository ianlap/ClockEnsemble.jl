# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking.** Renamed clock-model diffusion-coefficient fields to
  Zucca–Tavella 2005 notation: `q0 → R`, `q1 → σ1`, `q2 → σ2`,
  `q3 → σ3` on `TwoStateClock` and `ThreeStateClock`. Aligns the
  code with the math used throughout the theory docs.
- Refactored `src/filters.jl` into named per-step helpers
  (`predict_mean`, `predict_cov`, `innovation`, `innovation_cov`,
  `kalman_gain`, `posterior_mean`, `posterior_cov`) plus a
  top-of-file walkthrough of the recursion. `predict!`, `update!`,
  and `prop!` are now thin orchestrators with one-line comments
  naming each textbook quantity. Public-API signatures unchanged;
  bit-exact under the existing test suite.

## [0.1.0] — 2026-05-21

### Added

- Initial release, split from SigmaTau.jl v0.2.0.
- Clock state-space models: `AbstractClockModel`, `TwoStateClock`,
  `ThreeStateClock`, plus the model hooks `nstates`,
  `state_transition`, `process_noise`, `measurement_matrix`,
  `measurement_noise`.
- Discrete-time Kalman filter: `KalmanFilter`, `predict!`,
  `update!`, `prop!`. Out-of-place `StaticArrays`-based math for
  AD-friendly downstream composition.
- PID steering controller for closed-loop single-clock steering:
  `PIDController`, `step!`, `steer_to_correction`.
- Three runnable Literate.jl examples: open-loop Kalman tracking,
  closed-loop PID steering, and a holdover-budget comparison
  against TDEV and HTDEV (the holdover example pulls in SigmaTau.jl
  as an example-only dep for the deviation cross-checks).
- Documentation site with theory pages on the SDE clock model, the
  Kalman recursion, and PID steering.
