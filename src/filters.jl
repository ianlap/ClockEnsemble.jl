# filters.jl — Kalman filter + PID steering controller
#
# Terminology follows the Wikipedia Kalman-filter article. Φ here is
# Wikipedia's F_k.
#
#   PREDICT
#     x̂_{k|k-1} = Φ x̂_{k-1|k-1}                          a priori state
#     P_{k|k-1} = Φ P_{k-1|k-1} Φᵀ + Q                    a priori covariance
#
#   UPDATE
#     ỹ_k     = z_k − H x̂_{k|k-1}                         innovation
#     S_k     = H P_{k|k-1} Hᵀ + R                        innovation covariance
#     K_k     = P_{k|k-1} Hᵀ S_k⁻¹                        Kalman gain
#     x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k                       a posteriori state
#     P_{k|k} = (I−K_k H) P_{k|k-1} (I−K_k H)ᵀ + K_k R K_kᵀ   a posteriori covariance (Joseph form)
#
# Φ, Q, H, R come from the clock model's `state_transition`,
# `process_noise`, `measurement_matrix`, `measurement_noise` hooks.

"""
    KalmanFilter{V,M}

Mutable discrete-time Kalman filter state: state vector `x`,
covariance `P`, and a step counter `k`. The counter is used by
[`predict!`](@ref) to gate the first propagation against an unset
prior (legacy `filter_step!` convention); [`prop!`](@ref) ignores it.
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

# Per-step Kalman building blocks. Each function is one line of the
# recursion at the top of this file. Internal, not exported.

# x̂_{k|k-1} = Φ x̂_{k-1|k-1}   (+ control / steering input u_k).
apriori_state(x, Φ)    = Φ * x
apriori_state(x, Φ, u) = Φ * x + u

# P_{k|k-1} = Φ P_{k-1|k-1} Φᵀ + Q.
apriori_cov(P, Φ, Q) = Φ * P * Φ' + Q

# ỹ_k = z_k − H x̂_{k|k-1}.
innovation(z, H, x) = z - H * x

# S_k = H P_{k|k-1} Hᵀ + R.
innovation_cov(P, H, R) = H * P * H' + R

# K_k = P_{k|k-1} Hᵀ S_k⁻¹.
kalman_gain(P, H, S) = P * H' / S

# x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k.
aposteriori_state(x, K, ỹ) = x + K * ỹ

# P_{k|k} = (I − K H) P (I − K H)ᵀ + K R Kᵀ   (Joseph form).
# Algebraically equal to (I − K H) P but stays symmetric and PSD
# under round-off — robust default. Symmetrised explicitly to absorb
# any residual asymmetry.
function aposteriori_cov(P, K, H, R)
    n = size(P, 1)
    Iₙ = SMatrix{n, n, Float64}(I)
    IKH  = Iₙ - K * H
    Pnew = IKH * P * IKH' + K * R * K'
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

Optionally adds a `steering` correction vector to the predicted (a
priori) state estimate — phase = `+u·dt`, frequency = `+u`, higher
states zero.

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
        # a priori state (with optional steering u_k)
        if steering === nothing
            est.x = apriori_state(est.x, Φ)
        else
            u = _pad_steering(steering, length(est.x))
            est.x = apriori_state(est.x, Φ, u)
        end
        # a priori covariance
        est.P = apriori_cov(est.P, Φ, Q)
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

    # a priori state (with optional steering u)
    if steering === nothing
        est.x = apriori_state(est.x, Φ)
    else
        u = _pad_steering(steering, length(est.x))
        est.x = apriori_state(est.x, Φ, u)
    end

    # a priori covariance, symmetrised against round-off
    Pm   = SMatrix(est.P)
    Pnew = apriori_cov(Pm, Φ, Q)
    est.P = Symmetric((Pnew + Pnew') ./ 2.0)

    return est
end

"""
    update!(est::KalmanFilter, model::AbstractClockModel, z)

Scalar or vector measurement update. Computes the innovation, Kalman
gain, and a posteriori state and covariance using out-of-place
`StaticArrays` math (AD-friendly — no in-place mutation). The
covariance update uses the Joseph form, which stays symmetric and
positive-semidefinite under round-off.
"""
function update!(est::KalmanFilter, model::AbstractClockModel, z::Union{Real, AbstractVector})
    est.k += 1

    H = measurement_matrix(model)
    R = measurement_noise(model)

    z_vec = z isa Real ? SVector{1, Float64}(z) : SVector{length(z), Float64}(z...)

    # innovation
    ỹ = innovation(z_vec, H, est.x)

    # innovation covariance
    Pm = SMatrix(est.P)
    S  = innovation_cov(Pm, H, R)

    # Kalman gain
    K  = kalman_gain(Pm, H, S)

    # a posteriori state
    est.x = aposteriori_state(est.x, K, ỹ)

    # a posteriori covariance (Joseph form)
    est.P = aposteriori_cov(Pm, K, H, R)

    return est
end
