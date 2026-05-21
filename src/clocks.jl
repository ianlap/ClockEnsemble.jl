# clocks.jl — Clock state-space models

"""
    AbstractClockModel

Supertype for discrete-time clock state-space models used by the Kalman
estimators. Concrete subtypes parameterize the polynomial clock SDE
(phase / frequency / drift) plus per-process diffusion coefficients,
and must overload `nstates`, `state_transition`, `process_noise`,
`measurement_matrix`, and `measurement_noise`.

Shipped subtypes: [`TwoStateClock`](@ref) and [`ThreeStateClock`](@ref).
"""
abstract type AbstractClockModel end

"""
    TwoStateClock(; tau, q0=0.0, q1=0.0, q2=0.0)

Two-state polynomial clock model with state vector `[phase, frequency]`.
Step size `tau` is the discretization interval; `q0` is the white
phase-modulation (WPM) measurement-noise diffusion coefficient, `q1`
is white FM (state), and `q2` is random-walk FM (state). Parameterizes
the closed-form Φ and Q matrices used by the Kalman update loop.
"""
Base.@kwdef struct TwoStateClock <: AbstractClockModel
    tau::Float64
    q0::Float64 = 0.0 # WPM (measurement)
    q1::Float64 = 0.0 # WFM (state)
    q2::Float64 = 0.0 # RWFM (state)
end

"""
    ThreeStateClock(; tau, q0=0.0, q1=0.0, q2=0.0, q3=0.0)

Three-state polynomial clock model with state vector
`[phase, frequency, frequency_drift]`. Adds an integrated random-walk
FM (IRWFM / drift) channel with diffusion coefficient `q3` over
[`TwoStateClock`](@ref); meanings of `tau`, `q0`, `q1`, `q2` are
identical. Suited to clocks with non-negligible drift such as ageing
cesium tubes or GPS-grade rubidiums over long horizons.
"""
Base.@kwdef struct ThreeStateClock <: AbstractClockModel
    tau::Float64
    q0::Float64 = 0.0 # WPM (measurement)
    q1::Float64 = 0.0 # WFM (state)
    q2::Float64 = 0.0 # RWFM (state)
    q3::Float64 = 0.0 # IRWFM (state)
end

"""
    nstates(model::AbstractClockModel) → Int

Return the dimension of the state vector for `model`. `TwoStateClock`
returns `2`, `ThreeStateClock` returns `3`. Used by
[`steer_to_correction`](@ref) to size the steering `SVector`.
"""
nstates(::TwoStateClock) = 2
nstates(::ThreeStateClock) = 3

"""
    state_transition(model::AbstractClockModel)            → SMatrix
    state_transition(model::AbstractClockModel, dt::Real)  → SMatrix

Return the discrete-time state transition matrix Φ that propagates the
clock state forward by `dt` (or by `model.tau` when omitted). The
two-state Φ is the standard phase/frequency integrator `[1 dt; 0 1]`;
the three-state Φ adds the `dt²/2` and `dt` couplings for the drift
row. Returned as a `StaticArrays.SMatrix` for zero-allocation Kalman
propagation.

The `dt` overload is what `prop!` uses to integrate over arbitrary
horizons without requiring a separate model instance per step.
"""
function state_transition(m::TwoStateClock, dt::Real)
    @SMatrix [1.0 Float64(dt); 0.0 1.0]
end

function state_transition(m::ThreeStateClock, dt::Real)
    τ = Float64(dt)
    @SMatrix [1.0 τ τ^2 / 2.0; 0.0 1.0 τ; 0.0 0.0 1.0]
end

state_transition(m::TwoStateClock)   = state_transition(m, m.tau)
state_transition(m::ThreeStateClock) = state_transition(m, m.tau)

"""
    process_noise(model::AbstractClockModel)            → SMatrix
    process_noise(model::AbstractClockModel, dt::Real)  → SMatrix

Return the process-noise covariance matrix Q obtained by closed-form
analytic integration of the Wiener increments in the clock SDE over a
step of length `dt` (or `model.tau` when omitted), given the WFM /
RWFM / (IRWFM) diffusion coefficients on `model`. Coefficients match
the Galleani / Zucca derivations standard in the timescale literature.
Returned as an `SMatrix` for Kalman composition.

The `dt` overload is what `prop!` uses to integrate over arbitrary
horizons.
"""
function process_noise(m::TwoStateClock, dt::Real)
    τ = Float64(dt)
    Q11 = m.q1*τ + m.q2*τ^3/3.0
    Q12 = m.q2*τ^2/2.0
    Q22 = m.q2*τ
    @SMatrix [Q11 Q12; Q12 Q22]
end

function process_noise(m::ThreeStateClock, dt::Real)
    τ = Float64(dt)
    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = m.q1*τ + m.q2*τ3/3.0 + m.q3*τ5/20.0
    Q12 = m.q2*τ2/2.0 + m.q3*τ4/8.0
    Q13 = m.q3*τ3/6.0
    Q22 = m.q2*τ + m.q3*τ3/3.0
    Q23 = m.q3*τ2/2.0
    Q33 = m.q3*τ
    @SMatrix [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]
end

process_noise(m::TwoStateClock)   = process_noise(m, m.tau)
process_noise(m::ThreeStateClock) = process_noise(m, m.tau)

"""
    measurement_matrix(model::AbstractClockModel) → SMatrix

Return the linear measurement map H. Both shipped clock models observe
phase only, so H is a 1×n row vector picking out the first state
component (`[1 0]` for two-state, `[1 0 0]` for three-state).
Consumed by `update!` when forming the innovation `ν = z − H x`.
"""
measurement_matrix(::TwoStateClock) = @SMatrix [1.0 0.0]
measurement_matrix(::ThreeStateClock) = @SMatrix [1.0 0.0 0.0]

"""
    measurement_noise(model::AbstractClockModel) → SMatrix

Return the measurement-noise covariance R as a 1×1 `SMatrix` wrapping
the WPM diffusion coefficient `model.q0`. Identical for both shipped
clock types since the measurement is phase-only WPM. Consumed by
`update!` when forming the innovation covariance `S = H P Hᵀ + R`.
"""
measurement_noise(m::TwoStateClock) = @SMatrix [m.q0]
measurement_noise(m::ThreeStateClock) = @SMatrix [m.q0]
