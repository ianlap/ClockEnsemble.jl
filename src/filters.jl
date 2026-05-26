# filters.jl — Kalman filter + PID steering controller
#
# ── Kalman recursion at a glance ─────────────────────────────────────────
#
# State estimate `x` and its covariance `P` are evolved through two phases
# at each step. Symbols follow the standard textbook recursion (Bar-Shalom;
# Brown & Hwang) and the Zucca–Tavella 2005 notation used elsewhere in the
# package.
#
#   PREDICT (push the prior estimate forward by one time step):
#     x⁻ = Φ x                        # propagate mean
#     P⁻ = Φ P Φᵀ + Q                 # propagate covariance, inflate by Q
#
#   UPDATE (fold in a new measurement z):
#     ν  = z − H x⁻                   # innovation: measurement residual
#     S  = H P⁻ Hᵀ + R                # innovation covariance
#     K  = P⁻ Hᵀ S⁻¹                  # Kalman gain
#     x⁺ = x⁻ + K ν                   # corrected state
#     P⁺ = (I − K H) P⁻               # corrected covariance, symmetrised
#
# Quantities supplied by the clock model (`AbstractClockModel`):
#   Φ ← state_transition(model, dt)   # discrete propagator
#   Q ← process_noise(model, dt)      # integrated SDE diffusion
#   H ← measurement_matrix(model)     # phase-only [1 0 …]
#   R ← measurement_noise(model)      # 1×1 WPM noise covariance
#
# The per-step helpers below (`predict_mean`, `predict_cov`, `innovation`,
# `innovation_cov`, `kalman_gain`, `posterior_mean`, `posterior_cov`) are
# the literal one-liners above. `predict!`, `update!`, and `prop!` are
# thin orchestrators that call them in order. Open this file and read the
# bodies top-to-bottom to see the recursion line by line.

"""
    KalmanFilter{V,M}

Mutable discrete-time Kalman filter state: state vector `x`, covariance
`P`, and a step counter `k`. The counter is used by [`predict!`](@ref)
to gate the first propagation against an unset prior (legacy
`filter_step!` convention); [`prop!`](@ref) ignores it.
"""
mutable struct KalmanFilter{V<:AbstractVector{Float64}, M<:AbstractMatrix{Float64}}
    x::V
    P::M
    k::Int
end

"""
    KalmanFilter(x0, P0)

Build a `KalmanFilter` from initial state `x0` and covariance `P0`.
Inputs are converted to `SVector`/`SMatrix` for zero-allocation
dispatch through the update loop.
"""
function KalmanFilter(x0::AbstractVector{Float64}, P0::AbstractMatrix{Float64})
    n = length(x0)
    x = SVector{n, Float64}(x0...)
    P = SMatrix{n, n, Float64}(P0...)
    return KalmanFilter(x, P, 0)
end

# ── Per-step Kalman building blocks ──────────────────────────────────────
# Pure, allocation-free, out-of-place. Each function implements one line
# of the recursion at the top of this file. Not exported — they are
# internal so there's exactly one public way to drive the filter
# (`predict!` / `update!` / `prop!`) and the names below stay free to
# rename without breaking downstream code.

# Predicted state mean: x⁻ = Φ x  (plus an optional steering input u).
predict_mean(x, Φ) = Φ * x
predict_mean(x, Φ, u) = Φ * x + u

# Predicted covariance: P⁻ = Φ P Φᵀ + Q.
predict_cov(P, Φ, Q) = Φ * P * Φ' + Q

# Innovation (measurement residual): ν = z − H x.
innovation(z, H, x) = z - H * x

# Innovation covariance: S = H P Hᵀ + R.
innovation_cov(P, H, R) = H * P * H' + R

# Kalman gain: K = P Hᵀ S⁻¹.
kalman_gain(P, H, S) = P * H' / S

# Posterior state mean: x⁺ = x + K ν.
posterior_mean(x, K, ν) = x + K * ν

# Posterior covariance: P⁺ = (I − K H) P, symmetrised to suppress
# round-off-induced asymmetry across long horizons.
function posterior_cov(P, K, H)
    n = size(P, 1)
    Iₙ = SMatrix{n, n, Float64}(I)
    Pnew = (Iₙ - K * H) * P
    return Symmetric((Pnew + Pnew') ./ 2.0)
end

# Build the SVector{n,Float64} steering input from an arbitrary-length
# correction vector, zero-padding states beyond the supplied entries.
function _pad_steering(u, n)
    SVector{n, Float64}(ntuple(i -> i <= length(u) ? Float64(u[i]) : 0.0, n))
end

"""
    PIDController(; g_p=0.1, g_i=0.01, g_d=0.05)

Discrete PID controller for clock steering. Holds the running
phase-error sum and the last emitted steer for fold-in via
`predict!(…; steering=…)`.
"""
Base.@kwdef mutable struct PIDController
    g_p::Float64 = 0.1
    g_i::Float64 = 0.01
    g_d::Float64 = 0.05
    sumx::Float64 = 0.0
    last_steer::Float64 = 0.0
end

"""
    step!(pid::PIDController, x::AbstractVector{<:Real}) → Float64

Compute and store the next steer value from the current Kalman state
estimate. Sign convention: drives phase (and frequency, when present)
toward zero.
"""
function step!(pid::PIDController, x::AbstractVector{<:Real})
    pid.sumx += x[1]
    steer = -pid.g_p * x[1] - pid.g_i * pid.sumx
    length(x) >= 2 && (steer -= pid.g_d * x[2])
    pid.last_steer = steer
    return steer
end

"""
    steer_to_correction(steer::Float64, ns::Int, dt::Float64) → SVector{ns,Float64}

Build the steering correction vector that `predict!(…; steering=…)`
expects. Phase component is `steer·dt`, frequency component is
`steer`, higher-order states are zero.
"""
function steer_to_correction(steer::Float64, ns::Int, dt::Float64)
    if ns == 1
        return SVector{1,Float64}(steer * dt)
    elseif ns == 2
        return SVector{2,Float64}(steer * dt, steer)
    else
        return SVector{ns,Float64}(steer * dt, steer, ntuple(_ -> 0.0, ns - 2)...)
    end
end

"""
    predict!(est::KalmanFilter, model::AbstractClockModel, dt::Real;
             steering=nothing)

Propagate the estimator state forward in time by `dt`. Φ and Q are
re-derived from `model` for the supplied `dt` via the dt-aware
[`state_transition`](@ref) / [`process_noise`](@ref) overloads, so
`dt ≠ model.tau` is a valid finer/coarser propagation step.

Optionally adds a `steering` correction vector to the predicted state
mean — phase = `+u·dt`, frequency = `+u`, higher states zero.

On the first step (`k == 0`) the prediction is skipped — the initial
state is used directly, matching the legacy `filter_step!` convention.
Use [`prop!`](@ref) for an unconditional propagation that ignores this
gate.
"""
function predict!(est::KalmanFilter, model::AbstractClockModel, dt::Real;
                  steering::Union{Nothing,AbstractVector{Float64}} = nothing)
    Φ = state_transition(model, dt)
    Q = process_noise(model, dt)

    if est.k > 0
        # x⁻ = Φ x   (+ steering u, if supplied)
        if steering === nothing
            est.x = predict_mean(est.x, Φ)
        else
            u = _pad_steering(steering, length(est.x))
            est.x = predict_mean(est.x, Φ, u)
        end
        # P⁻ = Φ P Φᵀ + Q
        est.P = predict_cov(est.P, Φ, Q)
    end

    return est
end

"""
    prop!(est::KalmanFilter, model::AbstractClockModel, dt::Real;
          steering=nothing)

Unconditional covariance propagation: advances `est.x ← Φ(dt) x` and
`est.P ← Φ(dt) P Φ(dt)' + Q(dt)` regardless of `est.k`, and does not
increment `est.k`. Use this to project a 1σ covariance band from a
fresh side-channel filter (e.g. shaded ±1σ holdover bounds) without
disturbing the live filter's update sequencing.

Steering folds in identically to [`predict!`](@ref).
"""
function prop!(est::KalmanFilter, model::AbstractClockModel, dt::Real;
               steering::Union{Nothing,AbstractVector{Float64}} = nothing)
    Φ = state_transition(model, dt)
    Q = process_noise(model, dt)

    # x ← Φ x   (+ steering u, if supplied)
    if steering === nothing
        est.x = predict_mean(est.x, Φ)
    else
        u = _pad_steering(steering, length(est.x))
        est.x = predict_mean(est.x, Φ, u)
    end

    # P ← Φ P Φᵀ + Q, symmetrised against round-off
    Pm   = SMatrix(est.P)
    Pnew = predict_cov(Pm, Φ, Q)
    est.P = Symmetric((Pnew + Pnew') ./ 2.0)

    return est
end

"""
    update!(est::KalmanFilter, model::AbstractClockModel, z)

Scalar or vector measurement update. Computes innovation, Kalman gain,
and posterior covariance using out-of-place `StaticArrays` math
(AD-friendly — no in-place mutation), then symmetrises P.
"""
function update!(est::KalmanFilter, model::AbstractClockModel, z::Union{Real, AbstractVector})
    est.k += 1

    H = measurement_matrix(model)
    R = measurement_noise(model)

    z_vec = z isa Real ? SVector{1, Float64}(z) : SVector{length(z), Float64}(z...)

    # ν = z − H x   (innovation)
    ν = innovation(z_vec, H, est.x)

    # S = H P Hᵀ + R   (innovation covariance)
    Pm = SMatrix(est.P)
    S  = innovation_cov(Pm, H, R)

    # K = P Hᵀ S⁻¹   (Kalman gain)
    K  = kalman_gain(Pm, H, S)

    # x⁺ = x + K ν   (posterior mean)
    est.x = posterior_mean(est.x, K, ν)

    # P⁺ = (I − K H) P, symmetrised
    est.P = posterior_cov(Pm, K, H)

    return est
end
