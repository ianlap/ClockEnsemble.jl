# filters.jl — Kalman filter + PID steering controller
#
# ── Kalman recursion at a glance ─────────────────────────────────────────
#
# At step `k` the filter carries two snapshots of the clock state:
#   x̂_{k|k-1}, P_{k|k-1}   the *predicted* (a priori) estimate —
#                          everything you know having seen measurements
#                          z_1 … z_{k-1} but not yet z_k.
#   x̂_{k|k},   P_{k|k}     the *updated* (a posteriori) estimate —
#                          after folding in z_k.
#
# Each iteration alternates two phases. Names below match the Wikipedia
# / Bar-Shalom textbook recursion; Φ here is what Wikipedia calls F_k.
#
#   PREDICT (one step of clock dynamics, no measurement yet):
#     x̂_{k|k-1} = Φ x̂_{k-1|k-1}             # predicted (a priori) state estimate
#     P_{k|k-1} = Φ P_{k-1|k-1} Φᵀ + Q       # predicted (a priori) estimate covariance
#
#   UPDATE (fold measurement z_k into the prior):
#     ỹ_k       = z_k − H x̂_{k|k-1}          # innovation (measurement pre-fit residual)
#     S_k       = H P_{k|k-1} Hᵀ + R          # innovation covariance
#     K_k       = P_{k|k-1} Hᵀ S_k⁻¹          # optimal Kalman gain
#     x̂_{k|k}   = x̂_{k|k-1} + K_k ỹ_k         # updated (a posteriori) state estimate
#     P_{k|k}   = (I − K_k H) P_{k|k-1}       # updated (a posteriori) estimate covariance
#                                              (usual form; Joseph form is on TODO)
#
# Model-supplied quantities (one method per `AbstractClockModel`):
#   Φ ← state_transition(model, dt)   # discrete propagator (Wikipedia's F_k)
#   Q ← process_noise(model, dt)      # integrated SDE diffusion
#   H ← measurement_matrix(model)     # phase-only [1 0 …]
#   R ← measurement_noise(model)      # 1×1 WPM noise covariance
#
# The per-step helpers below — `apriori_state`, `apriori_cov`,
# `innovation`, `innovation_cov`, `kalman_gain`, `aposteriori_state`,
# `aposteriori_cov` — are literally the one-liners above. `predict!`,
# `update!`, and `prop!` are thin orchestrators that call them in order.
# Read this file top-to-bottom and the recursion appears line by line.

"""
    KalmanFilter{V,M}

Mutable discrete-time Kalman filter state: state vector `x` (carrying
the current a-priori or a-posteriori estimate, depending on where in
the iteration you read it), covariance `P`, and a step counter `k`.
The counter is used by [`predict!`](@ref) to gate the first
propagation against an unset prior (legacy `filter_step!` convention);
[`prop!`](@ref) ignores it.
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
# (`predict!` / `update!` / `prop!`); the names below stay free to
# rename without breaking downstream code.

# Predicted (a priori) state estimate: x̂_{k|k-1} = Φ x̂_{k-1|k-1}
# (plus an optional control / steering input u_k).
apriori_state(x, Φ)    = Φ * x
apriori_state(x, Φ, u) = Φ * x + u

# Predicted (a priori) estimate covariance: P_{k|k-1} = Φ P_{k-1|k-1} Φᵀ + Q.
apriori_cov(P, Φ, Q) = Φ * P * Φ' + Q

# Innovation (measurement pre-fit residual): ỹ_k = z_k − H x̂_{k|k-1}.
innovation(z, H, x) = z - H * x

# Innovation covariance: S_k = H P_{k|k-1} Hᵀ + R.
innovation_cov(P, H, R) = H * P * H' + R

# Optimal Kalman gain: K_k = P_{k|k-1} Hᵀ S_k⁻¹.
kalman_gain(P, H, S) = P * H' / S

# Updated (a posteriori) state estimate: x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k.
aposteriori_state(x, K, ỹ) = x + K * ỹ

# Updated (a posteriori) estimate covariance: P_{k|k} = (I − K_k H) P_{k|k-1},
# symmetrised to suppress round-off-induced asymmetry over long horizons.
# (Usual form; Joseph form is on TODO for the next iteration.)
function aposteriori_cov(P, K, H)
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
        # Predicted (a priori) state estimate: x̂_{k|k-1} = Φ x̂_{k-1|k-1}
        #   (+ control / steering input u_k, if supplied)
        if steering === nothing
            est.x = apriori_state(est.x, Φ)
        else
            u = _pad_steering(steering, length(est.x))
            est.x = apriori_state(est.x, Φ, u)
        end
        # Predicted (a priori) estimate covariance: P_{k|k-1} = Φ P_{k-1|k-1} Φᵀ + Q
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

    # A priori state propagation: x̂ ← Φ x̂   (+ steering u, if supplied)
    if steering === nothing
        est.x = apriori_state(est.x, Φ)
    else
        u = _pad_steering(steering, length(est.x))
        est.x = apriori_state(est.x, Φ, u)
    end

    # A priori covariance propagation: P ← Φ P Φᵀ + Q, symmetrised against round-off
    Pm   = SMatrix(est.P)
    Pnew = apriori_cov(Pm, Φ, Q)
    est.P = Symmetric((Pnew + Pnew') ./ 2.0)

    return est
end

"""
    update!(est::KalmanFilter, model::AbstractClockModel, z)

Scalar or vector measurement update. Computes the innovation, the
optimal Kalman gain, and the updated (a posteriori) state and
covariance using out-of-place `StaticArrays` math (AD-friendly — no
in-place mutation), then symmetrises P.
"""
function update!(est::KalmanFilter, model::AbstractClockModel, z::Union{Real, AbstractVector})
    est.k += 1

    H = measurement_matrix(model)
    R = measurement_noise(model)

    z_vec = z isa Real ? SVector{1, Float64}(z) : SVector{length(z), Float64}(z...)

    # Innovation (measurement pre-fit residual): ỹ_k = z_k − H x̂_{k|k-1}
    ỹ = innovation(z_vec, H, est.x)

    # Innovation covariance: S_k = H P_{k|k-1} Hᵀ + R
    Pm = SMatrix(est.P)
    S  = innovation_cov(Pm, H, R)

    # Optimal Kalman gain: K_k = P_{k|k-1} Hᵀ S_k⁻¹
    K  = kalman_gain(Pm, H, S)

    # Updated (a posteriori) state estimate: x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k
    est.x = aposteriori_state(est.x, K, ỹ)

    # Updated (a posteriori) estimate covariance: P_{k|k} = (I − K_k H) P_{k|k-1}
    est.P = aposteriori_cov(Pm, K, H)

    return est
end
