# filters.jl — Kalman filter + PID steering controller

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
    Phi = state_transition(model, dt)
    Q   = process_noise(model, dt)

    if est.k > 0
        x_pred = Phi * est.x
        if steering !== nothing
            n = length(x_pred)
            s = SVector{n, Float64}(ntuple(i -> i <= length(steering) ? steering[i] : 0.0, n))
            x_pred = x_pred + s
        end
        est.x = x_pred
        est.P = Phi * est.P * Phi' + Q
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
    Phi = state_transition(model, dt)
    Q   = process_noise(model, dt)

    x_pred = Phi * est.x
    if steering !== nothing
        n = length(x_pred)
        s = SVector{n, Float64}(ntuple(i -> i <= length(steering) ? steering[i] : 0.0, n))
        x_pred = x_pred + s
    end
    est.x = x_pred

    Pm   = SMatrix(est.P)
    Pnew = Phi * Pm * Phi' + Q
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

    ν = z_vec - H * est.x

    Pm = SMatrix(est.P)
    S  = H * Pm * H' + R
    K  = Pm * H' / S

    est.x = est.x + K * ν

    n     = length(est.x)
    I_mat = SMatrix{n, n, Float64}(I)
    Pnew  = (I_mat - K * H) * Pm
    Pnew  = (Pnew + Pnew') ./ 2.0
    est.P = Symmetric(Pnew)

    return est
end
