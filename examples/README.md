# Examples

Each `.jl` file in this directory is a runnable Julia script that
doubles as a docs page on the published site. They're written in
[Literate.jl](https://github.com/fredrikekre/Literate.jl) format —
plain Julia code, with prose comments lifted into Markdown headings
when the docs build. Edit the script, rebuild, and the corresponding
`docs/src/tutorials/*.md` page regenerates.

| Script                              | Topic                                                                |
|-------------------------------------|----------------------------------------------------------------------|
| `01_kalman_single_clock.jl`         | Track a `ThreeStateClock` with `KalmanFilter` (open-loop).           |
| `02_kalman_pid_steering.jl`         | Closed-loop PID steering with critical-damping gains.                |
| `03_holdover_comparison.jl`         | Holdover budget four ways: TDEV / HTDEV / KF RMS / KF 1σ via `prop!`.|

Example 03 also pulls in [SigmaTau.jl](https://github.com/ianlap/SigmaTau.jl)
for the stability-deviation reference curves (TDEV, HTDEV). SigmaTau
is an example-only dep, declared in `examples/Project.toml`, and is
not a runtime dependency of ClockEnsemble itself.

## Running

From the repo root:

```bash
julia --project=examples examples/01_kalman_single_clock.jl
```

`examples/Project.toml` is the env that carries `Plots` + `PGFPlotsX`
+ `SigmaTau` (none are runtime deps of ClockEnsemble itself — they're
only needed for visualisation and stability cross-checks). On the
first run,
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'` will
resolve and download deps; subsequent runs skip straight to compile.
A clean checkout under `--project=.` would fail with
`Package Plots not found` — use `--project=examples`.

## Docs build

`docs/make.jl` walks this directory and runs `Literate.markdown` on
each `.jl` file, dropping the resulting `.md` into
`docs/src/tutorials/`. The generated pages are gitignored — the
`.jl` files are the single source of truth.
