# ClockEnsemble.jl — agent context

You are working on a Julia 1.11 package authored by Ian Lapinski.

## Authorship and attribution rules

- All changes are authored by Ian. Do not add yourself as a co-author or
  attribute work to "Claude" or "AI" anywhere.
- Do not add "Co-authored-by: Claude" or similar trailers to commits.
- Do not add "Generated with Claude Code" footers, signatures, or comments
  to code, commit messages, PRs, changelogs, or docs.
- Commit messages, CHANGELOG entries, and code comments should be written
  in Ian's voice as if he wrote them directly. No first-person from you.
- Do not add comments like "// added by AI" or "# Claude: refactored this".

## Package architecture

Single Julia 1.11 package. **Flat module**, no submodules. State-space
clock models, Kalman filtering, and PID steering for precision clocks.
v0.1 ships the single-clock-steered-to-reference case.

The sister package [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl)
handles stability-deviation analysis (ADEV, MDEV, TDEV, HDEV, MTIE,
PDEV, totals, noise identification, EDF / χ² confidence intervals).
ClockEnsemble.jl does **not** depend on SigmaTau.jl — the two are
peer packages that operate on the same kind of phase record from
complementary angles. Only `examples/03_holdover_comparison.jl`
crosses the line, pulling in SigmaTau as an example-only dep.

### File map (most-touched paths)

- `src/ClockEnsemble.jl` — top-level module. Imports `LinearAlgebra`,
  `StaticArrays`, `DocStringExtensions`. Includes `clocks.jl` and
  `filters.jl`; exports the public surface (see below).
- `src/clocks.jl` — `AbstractClockModel`, `TwoStateClock`,
  `ThreeStateClock` (kwdef structs with `tau`, `R`, `σ1`, `σ2`,
  optional `σ3` — Zucca–Tavella notation). Defines `nstates`,
  `state_transition`, `process_noise`, `measurement_matrix`,
  `measurement_noise` — each with a `dt`-aware overload returning
  `SMatrix` for zero-allocation Kalman propagation.
- `src/filters.jl` — `KalmanFilter` (`mutable struct` with `x`, `P`,
  `k`), constructor that lifts inputs to `SVector` / `SMatrix`;
  `predict!`, `update!`, `prop!` — out-of-place
  `StaticArrays`-based math (AD-friendly, no in-place mutation).
  `PIDController` + `step!` + `steer_to_correction` for closed-loop
  steering.
- `test/runtests.jl` — flat `@testset "ClockEnsemble"` block with
  PID, clock-model, predict / update, prop! parity, and steering
  testsets.
- `examples/01_kalman_single_clock.jl` — open-loop tracking.
- `examples/02_kalman_pid_steering.jl` — closed-loop PID steering
  with critical-damping gains.
- `examples/03_holdover_comparison.jl` — TDEV / HTDEV / KF RMS / KF
  1σ holdover budget. Pulls in SigmaTau as an example-only dep.
- `docs/src/theory/{kalman,steering,ensembles}.md` — theory pages.
- `docs/src/reference.md` — API reference (flat).
- `docs/src/tutorials/0X_*.md` — Literate.jl-generated from the
  `examples/*.jl` scripts; gitignored.

### Public surface

```
AbstractClockModel, TwoStateClock, ThreeStateClock
nstates, state_transition, process_noise,
    measurement_matrix, measurement_noise
KalmanFilter
predict!, update!, prop!
PIDController, step!, steer_to_correction
```

### Agent-context pair

`CLAUDE.md` and `AGENTS.md` are both checked into the repo. When you
change one, mirror the change in the other. They diverge only on the
agent-name in attribution rules.

## Critical conventions — do not violate

- Kalman filter math is out-of-place via `StaticArrays` for
  AD-friendliness. In-place mutation is opt-in only.
- The Φ and Q matrices are derived per-call from the model's `dt`
  argument (not pre-cached on the model struct). `dt ≠ model.tau`
  is a valid finer/coarser propagation step.
- `predict!` gates the first step on `est.k == 0` (no-op);
  `prop!` is unconditional and does not bump `est.k`. Do not
  collapse these into one function — they serve different purposes
  (live filtering vs covariance-band projection).

## Verification standards

- Tests in `test/runtests.jl` lock the dt-overload parity, the
  group / additivity composition of `prop!`, and the PID closed-loop
  convergence. Run `julia --project=. -e 'using Pkg; Pkg.test()'`
  after any change to `clocks.jl` or `filters.jl`.
- Reference math: Galleani / Zucca clock SDE; Tryon & Jones 1983;
  Breakiron 2001; Stein 2003; Matsakis & Coleman 2020.

## Development workflow — use Revise.jl

There is a persistent Julia REPL available in the VS Code Julia
extension. Use it. Do not spawn fresh `julia -e` invocations for
verification — those pay the full JIT compilation cost (30–60 s)
every time. The persistent REPL has Revise.jl loaded and hot-patches
changes in ~100 ms.

After editing a file:
1. Save the file. Revise picks up the change automatically.
2. Re-run the relevant function in the REPL to verify.
3. If you do not have REPL access, ask Ian to run a specific command
   and paste the output. Do not spawn a new Julia process unless
   necessary.

Revise CANNOT hot-patch the following — when you make these changes,
explicitly tell Ian to restart the Julia REPL:

- Adding/removing/reordering fields in a struct
- Changing a struct's type parameters
- Changes to any `Project.toml` or `Manifest.toml`
- New `@eval`'d definitions or some macro changes

## Testing

- Run all tests:              `julia --project=. -e 'using Pkg; Pkg.test()'`
- Inside the persistent REPL: `pkg> test` (faster, reuses session)
- `Random` is in `[extras]` — do not remove.

## Quality checks (run periodically, not every change)

- `using Aqua; Aqua.test_all(ClockEnsemble)` — method ambiguities,
  stale deps.
- `using JET; report_package(ClockEnsemble)` — type instabilities.

## TODO ↔ CHANGELOG workflow

Every shipped code change must, in the same commit:

1. Remove the matching item from `TODO.md`.
2. Add a Keep-a-Changelog entry under `## [Unreleased]` in
   `CHANGELOG.md` (terse, past tense, no marketing voice, no emoji).

If a change does not warrant a TODO/CHANGELOG entry (pure docs,
typo fixes), say so in the commit body so it's intentional.

## Editing rules

- Never edit `Manifest.toml` files; they are gitignored locally.
- When adding a new clock model, add the model-hook overloads
  (`nstates`, `state_transition`, `process_noise`,
  `measurement_matrix`, `measurement_noise`) in `clocks.jl` and a
  corresponding testset.
- Match existing surrounding code style. 4-space indent, no trailing
  whitespace, docstrings on all exported functions.

## Style for prose deliverables

CHANGELOG entries follow Keep-a-Changelog. Use the same terse,
factual voice as existing entries. Past tense, no marketing
language, no emoji. Commit messages: imperative mood, ≤72 char
subject, body explains *why*.
