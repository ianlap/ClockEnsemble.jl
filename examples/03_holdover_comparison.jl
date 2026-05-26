# # Single-clock holdover: TDEV vs HTDEV vs Kalman prediction error
#
# When a clock disconnects from its reference, what is the predicted
# 1σ time-error budget at any holdover horizon ``\tau``? This tutorial
# answers the question three ways on the same fixture and overlays
# them on a single log-log plot:
#
# 1. **TDEV** — time deviation, the σ\_x quantity
#    ``\sigma_x(\tau) = (\tau/\sqrt{3})\,\mathrm{MDEV}(\tau)``.
#    Statistical estimate from the observed phase residuals.
# 2. **HTDEV** — Hadamard time deviation. Same idea as TDEV, but
#    built from the third difference instead of the second, so it
#    rejects constant frequency drift.
# 3. **Kalman RMS prediction error** — run a `KalmanFilter`
#    over the observed phase data to maturity, then at each mature
#    epoch project the converged state forward by every horizon h
#    and compare to the actual phase. The RMS prediction error vs
#    horizon is the operational holdover budget.
#
# This mirrors the MATLAB `kf_predict.m` reference: filter to
# maturity, deterministically project the state forward, accumulate
# RMS error across many starting points. Both TDEV and the KF RMS
# error estimate the same physical quantity from different angles —
# they should overlay closely.

# ## 1. Frame the problem
#
# A clock under steering tracks a reference (GPS, a maser, a
# coordinated time scale) by continuously pulling its frequency back
# toward the reference's pace. When the reference is lost, the
# steering loop has nothing to lock to and the clock free-runs: it
# *holds over*. From that moment on, the clock's accumulated time
# error against the would-be reference grows at a rate set entirely
# by the clock's own intrinsic stability.
#
# Two ways to estimate the 1σ budget at horizon τ:
#
# - **Stability-deviation estimators** (TDEV, HTDEV) operate on a
#   record of free-running phase residuals, returning σ\_x(τ) at
#   every τ in the record's support.
# - **State-space prediction** runs a Kalman filter on the same
#   record to lock onto the state, then projects forward
#   deterministically. The RMS prediction error vs horizon is the
#   operational holdover budget — *this number, in seconds, is the
#   1σ time-error you'll accumulate after τ seconds of free-running*.

# ## 2. Build a Cs-like clock model
#
# A commercial caesium-beam clock is dominated by white-FM noise
# (WHFM) at short averaging times and random-walk-FM (RWFM) at long
# ones; on top of that there's white phase measurement noise (WPM)
# from the time-interval counter. We pick representative spectral
# densities:

using Random
using LinearAlgebra
using ClockEnsemble
# SigmaTau supplies the deviation-side reference curves (PhaseData,
# tdev, htdev). Import only the names we need so they don't shadow
# ClockEnsemble's exports (TwoStateClock, etc. existed in SigmaTau
# v0.2.0 too, before the split).
import SigmaTau: PhaseData, tdev, htdev

R  = 1.0e-22   # WPM (measurement noise variance)
σ1 = 1.0e-23   # WFM (state)
σ2 = 1.0e-32   # RWFM (state)
σ3 = 0.0       # RRFM (negligible for Cs over 10⁴ s horizons)
tau0 = 1.0

clock = ThreeStateClock(tau = tau0, R = R, σ1 = σ1, σ2 = σ2, σ3 = σ3)

# ## 3. Simulate the underlying state and measured phase
#
# Forward-simulate the state trajectory using Cholesky-sampled
# process noise. With `σ3 = 0` the drift row/col of Q are zero, so
# the noise increment lives in a 2×2 sub-block — Cholesky-factor
# that and pad the third channel with a zero. (A naive ridge on
# `(3,3)` would inject fake IRWFM noise that Φ integrates into a
# τ³ phase run-away over 10⁵ steps.)
#
# Then add white measurement noise (WPM) on top to produce the
# observed phase data. The Kalman filter sees the noisy
# measurements; we'll keep the noise-free state trajectory around
# only as a sanity check.

Random.seed!(42)
N = 100_000
Φ = state_transition(clock)
Q = process_noise(clock)
L2 = cholesky(Q[1:2, 1:2]).L

phase_state = zeros(N)
let x = zeros(3)
    for k in 1:N
        w12 = L2 * randn(2)
        x = Φ * x + [w12[1], w12[2], 0.0]
        phase_state[k] = x[1]
    end
end

# White phase measurement noise on top of the underlying state:
phase_meas = phase_state .+ sqrt(R) .* randn(N)
data = PhaseData(phase_meas, tau0)

# ## 4. Compute TDEV and HTDEV
#
# Both are σ\_x quantities (units of seconds), evaluated on a
# log-spaced averaging-factor grid:

m_values = unique(round.(Int, exp10.(range(0, 4, length = 20))))

result_tdev  = tdev(data,  m_values; calc_ci = false)
result_htdev = htdev(data, m_values; calc_ci = false)

# ## 5. Run the Kalman filter and extract converged state
#
# Standard predict / update loop on the measurement record. The
# filter starts from a near-zero state with a moderately wide
# covariance prior and locks onto the WHFM/RWFM dynamics within a
# few thousand samples. Capture the per-step phase / freq / drift
# estimates for the prediction analysis below.

est = KalmanFilter([phase_meas[1], 0.0, 0.0],
                           Matrix(1.0e-12 * I(3)))

phase_est = zeros(N)
freq_est  = zeros(N)
drift_est = zeros(N)
for k in 1:N
    predict!(est, clock, tau0)
    update!(est, clock, phase_meas[k])
    phase_est[k] = est.x[1]
    freq_est[k]  = est.x[2]
    drift_est[k] = est.x[3]
end

# ## 6. Empirical RMS prediction error vs horizon
#
# This is the procedure from the MATLAB `kf_predict.m` reference:
#
# - Pick a *maturity* epoch after which the filter is considered
#   converged (here `N/2`, which is conservative for our seed).
# - For every starting epoch `np` past maturity and every horizon
#   `h` in the τ-grid, deterministically project the converged
#   state forward:
#   ```
#   xpred(h) = Φ^h · x = phase_est[np] + freq_est[np]·(h·τ₀)
#                       + ½·drift_est[np]·(h·τ₀)²
#   ```
# - Compare against the actual phase measurement `phase_meas[np+h]`,
#   accumulate the squared error.
# - RMS over all (np, h) pairs at each h gives the empirical 1σ
#   prediction-error curve.
#
# This is purely deterministic propagation of `x` — no Q, no
# `predict!` recursion, just `Φ^h · x`. The companion *theoretical*
# 1σ curve is computed in §6.5 below using `prop!`, which integrates
# Q forward over the same horizon grid; the empirical RMS and the
# theoretical 1σ should overlay closely once the filter has
# converged.

const MATURITY = N ÷ 2

err_var  = zeros(length(m_values))
err_n    = zeros(Int, length(m_values))
for np in MATURITY:(N - 1)
    p1 = phase_est[np]
    f1 = freq_est[np]
    d1 = drift_est[np]
    for (i, h) in enumerate(m_values)
        np + h <= N || continue
        h_τ   = h * tau0
        xpred = p1 + f1 * h_τ + 0.5 * d1 * h_τ^2
        err   = phase_meas[np + h] - xpred
        err_var[i] += err^2
        err_n[i]   += 1
    end
end
kf_rms = sqrt.(err_var ./ err_n)

# ## 6.5. Theoretical 1σ via `prop!`
#
# `prop!(est, model, dt)` propagates `Φ x` and `Φ P Φ' + Q` over an
# arbitrary horizon `dt` without bumping the filter's step counter
# (no `update!`, so it's safe to use on a side-channel copy). At
# maturity the filter's covariance has converged to a steady-state
# value `P_mature`; for each horizon `h` we re-seed a fresh estimator
# at `P_mature` and prop! forward by `h·τ₀`, then read
# `√P[1,1]` as the theoretical 1σ phase error after `h·τ₀` of
# free-running.
#
# This is the closed-form companion to the empirical RMS curve in
# §6: same physics (state-noise integration over the horizon),
# different angle (analytic Q-integration vs. RMS over many starting
# epochs). They should overlay closely.

P_mature = Matrix(est.P)         # converged P at the end of the filtering loop
kf_1sigma = zeros(length(m_values))
for (i, h) in enumerate(m_values)
    side = KalmanFilter(zeros(3), copy(P_mature))
    prop!(side, clock, h * tau0)
    kf_1sigma[i] = sqrt(side.P[1, 1])
end

# ## 7. Side-by-side comparison
#
# Three curves on one log-log axis. Each estimates the σ\_x
# holdover budget from a different angle:
#
# - **TDEV / HTDEV** are structure-function moments: a second
#   (TDEV) or third (HTDEV) difference of the phase record,
#   normalised by ``\tau/\sqrt{3}``. They divide by `√3`-style
#   factors that come out of the moment definition.
# - **KF RMS prediction error** is a direct RMS of
#   `phase_meas[np+h] − xpred(h)` over many trial epochs. No
#   normalisation factor — it's the raw 1σ phase error you actually
#   accumulate when you stop steering at `np` and free-run for
#   `h` seconds.
#
# What to read off the plot:
#
# - **Three regimes.** Short τ is WPM-dominated (flat noise floor
#   from the measurement-noise term `R`). Mid-τ shows the WHFM
#   rise (slope ≈ +1/2). Long τ steepens to the RWFM regime
#   (slope ≈ +3/2). All three curves trace the same regime
#   transitions — the noise model is consistent across the three
#   estimators.
# - **Constant vertical offset.** The KF RMS curve sits a factor
#   of `≈√6` above TDEV in the WHFM regime — that's the structural
#   conversion between σ\_x as a *τ-window second-difference moment*
#   (TDEV) and σ\_x as the *RMS phase deviation* over a τ-second
#   propagation (KF). Both are valid 1σ holdover budgets; the
#   choice of which to quote is a community / spec-sheet
#   convention, not a physics distinction. (Other noise regimes
#   have their own factors.)
# - **TDEV ≈ HTDEV.** The fixture has no constant frequency drift,
#   so HTDEV's third-difference operator has nothing to reject.
#   See the "Adding ageing" note below for the modification that
#   pulls them apart.

using Plots

plot(result_tdev.tau, result_tdev.dev;
     label  = "TDEV",
     xscale = :log10, yscale = :log10,
     xlabel = raw"Holdover horizon \(\tau\) (s)",
     ylabel = "1σ time-error budget (s)",
     title  = "Single-clock holdover: stability vs prediction error",
     legend = :topleft,
     lw     = 1.5)
plot!(result_htdev.tau, result_htdev.dev;
      label = "HTDEV",
      lw    = 1.5)
plot!(result_tdev.tau, kf_rms;
      label = "KF RMS prediction error",
      ls    = :dash,
      lw    = 1.5)
plot!(result_tdev.tau, kf_1sigma;
      label = "KF 1σ via prop!",
      ls    = :dot,
      lw    = 1.5)

# !!! note "Adding ageing"
#     The fixture above is noise-only — the WHFM and RWFM
#     contributions are drift-free in expectation, so TDEV and
#     HTDEV agree. To exercise HTDEV's drift-rejection (and watch
#     TDEV inflate while HTDEV stays clean), add a quadratic
#     ageing term to `phase_state` before adding measurement
#     noise:
#     ```julia
#     d_f = 1.0e-18                                       # frequency ageing rate (1/s)
#     phase_state .+= 0.5 .* d_f .* (collect(0:N-1) .* tau0).^2
#     ```
#     A constant frequency *offset* (linear in `t`) doesn't inflate
#     TDEV — both TDEV and HTDEV cancel constants. It's the
#     time-dependence of frequency that separates them. The KF
#     prediction would also pick up the ageing if `σ3` is set to
#     model a random-run frequency drift, or stay clean if `σ3 = 0`
#     (drift-blind, like HTDEV).

# ## 8. Read-out at a few horizons
#
# Pick three horizons (100 s, 1000 s, 10 000 s) and print all three
# estimates side-by-side:

function readout(tau_grid, curve, target_tau)
    i = argmin(abs.(tau_grid .- target_tau))
    (tau = tau_grid[i], value = curve[i])
end

for τ_target in (100.0, 1_000.0, 10_000.0)
    t_tdev   = readout(result_tdev.tau,  result_tdev.dev,  τ_target)
    t_htdev  = readout(result_htdev.tau, result_htdev.dev, τ_target)
    t_kf     = readout(result_tdev.tau,  kf_rms,           τ_target)
    t_prop   = readout(result_tdev.tau,  kf_1sigma,        τ_target)
    @info "Horizon τ ≈ $(t_tdev.tau) s" tdev=t_tdev.value htdev=t_htdev.value kf_rms=t_kf.value kf_1sigma_prop=t_prop.value
end

# ## 9. When to use which
#
# - **TDEV / HTDEV** when you have phase data and want a quick
#   stability read-out without running a filter. HTDEV additionally
#   rejects constant frequency drift, so it's preferred when the
#   record carries an ageing term you don't want to budget against.
# - **KF RMS prediction error** when you've already deployed the
#   Kalman filter (e.g. for steering) and want the empirical
#   holdover budget that *that filter* would deliver under
#   free-running. This is the operational metric your production
#   timing system inherits.
#
# All three approaches work from the same noise spec. The KF route
# requires a model — but you needed one anyway to run the filter.
