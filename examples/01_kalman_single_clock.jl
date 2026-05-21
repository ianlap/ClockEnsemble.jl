# # Tracking a single clock with the Kalman filter
#
# `KalmanFilter` propagates a clock-state estimate through
# [`predict!`](@ref) (apply Φ, accumulate Q) and [`update!`](@ref)
# (fold in a phase measurement). Together they solve the
# minimum-variance state-tracking problem for the linear-Gaussian
# polynomial clock model.
#
# This tutorial:
#
# 1. Picks a [`ThreeStateClock`](@ref) with realistic noise levels.
# 2. Synthesises a ground-truth `[phase, frequency, drift]` trajectory
#    by sampling the model's process noise covariance directly.
# 3. Adds white phase-modulation measurement noise on top of the true
#    phase to build a noisy phase tape.
# 4. Streams those measurements through `predict!` / `update!`.
# 5. Plots the filter's phase estimate against the ground truth, and
#    the residual `phase_true − phase_est` to show convergence.
#
# This is open-loop — no controller. The closed-loop steering version
# is [Closed-loop steering with critical-damping PID gains](02_kalman_pid_steering.md).

using ClockEnsemble
using LinearAlgebra
using Random

# ## Clock model
#
# Strong WFM + RWFM contributions, no IRWFM (so Q is a 3×3 matrix
# with `q3=0` — the drift state is a pure integrator with no
# stochastic input). Tuned so the random-walk-frequency excursion
# accumulates a visible drift over the simulation horizon.

τ  = 1.0                                       # discretization step (s)
q0 = 1e-18                                     # WPM measurement noise
q1 = 1e-20                                     # WFM state noise
q2 = 1e-24                                     # RWFM state noise
q3 = 0.0                                       # IRWFM disabled

model = ThreeStateClock(tau=τ, q0=q0, q1=q1, q2=q2, q3=q3)

# Pull the closed-form Φ and Q out of the model. Both are
# `StaticArrays.SMatrix` for zero-allocation propagation.

Φ = state_transition(model)
Q = process_noise(model)

# ## Synthetic ground truth
#
# Sample the multivariate process noise via a Cholesky factor at each
# step. `Q + ε·I` regularisation handles the `q3 = 0` row that makes
# `Q` rank-deficient.

Random.seed!(20260509)

N = 2000                                       # number of samples

x_true = zeros(3, N)
x_true[:, 1] .= [0.0, 1e-10, 0.0]             # initial phase 0,
                                               # initial freq 1e-10
                                               # (deliberate offset so
                                               # the curve drifts).

L = cholesky(Symmetric(Matrix(Q) + 1e-30 * I)).L

for k in 2:N
    w = L * randn(3)
    x_true[:, k] = Φ * x_true[:, k-1] .+ w
end

phase_true = view(x_true, 1, :)

# ## Noisy measurement tape
#
# `update!` consumes scalar phase measurements `z[k] = x_true[1, k] +
# v[k]` with `v[k] ~ N(0, q0)`.

z = phase_true .+ sqrt(q0) .* randn(N)

# ## Run the filter
#
# `KalmanFilter` is constructed with an initial state mean
# and covariance. Both convert internally to `SVector` / `SMatrix`.
# Loose initial uncertainty (`P₀ = 1e-12 · I`) lets the filter
# converge in a handful of steps even though the seed is intentionally
# offset from the true state.

x₀ = [z[1], 0.0, 0.0]
P₀ = 1e-12 * Matrix(I(3))

est = KalmanFilter(x₀, P₀)

phase_est = zeros(N)
freq_est  = zeros(N)
drift_est = zeros(N)

for k in 1:N
    predict!(est, model, τ)
    update!(est, model, z[k])

    phase_est[k] = est.x[1]
    freq_est[k]  = est.x[2]
    drift_est[k] = est.x[3]
end

# ## Phase estimate vs ground truth
#
# At this scale the two curves should overlap to well within the
# measurement-noise envelope. The interesting figure is the residual.

using Plots
t = (0:N-1) .* τ

plot(t, phase_true;
     label = "true phase",
     xlabel = raw"Time \(t\) (s)",
     ylabel = raw"Phase \(x(t)\) (s)",
     title = "Filter estimate vs ground-truth phase",
     lw = 1.2)
plot!(t, phase_est;
      label = "Kalman estimate",
      ls = :dash,
      lw = 1.2)

# ## Estimation residual
#
# The residual `phase_true − phase_est` shows the post-convergence
# tracking error, which should sit comfortably below the
# measurement-noise level `√q0 = $(round(sqrt(q0); sigdigits=3))` s.

residual = phase_true .- phase_est
plot(t, residual;
     label = false,
     xlabel = raw"Time \(t\) (s)",
     ylabel = "True minus estimated phase (s)",
     title = "Estimation residual",
     lw = 1.0)

# Final reading: print the converged state estimate alongside the
# truth.

@info "Final true [phase, freq, drift]: " x_true[:, end]
@info "Final est. [phase, freq, drift]: " est.x
