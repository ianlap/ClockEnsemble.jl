# clocks.jl — Clock state-space models

"""
    AbstractClockModel

Supertype for discrete-time clock state-space models used by the Kalman
estimators. Concrete subtypes parameterize the polynomial clock SDE
(phase / frequency / drift) plus per-process diffusion coefficients,
and must overload `nstates`, `state_transition`, `process_noise`,
`measurement_matrix`, and `measurement_noise`.

Shipped subtypes: [`TwoStateClock`](@ref) and [`ThreeStateClock`](@ref).
Diffusion-coefficient names follow Zucca–Tavella 2005: `σ1, σ2, σ3`
for WFM / RWFM / RRFM state noises and `R` for the WPM measurement
noise.
"""
abstract type AbstractClockModel end

"""
    TwoStateClock(; tau, R=0.0, σ1=0.0, σ2=0.0)

Two-state polynomial clock model with state vector `[phase, frequency]`.
Step size `tau` is the discretization interval; `R` is the white
phase-modulation (WPM) measurement-noise diffusion coefficient, `σ1`
is white FM (state), and `σ2` is random-walk FM (state). Names follow
Zucca–Tavella 2005. Parameterises the closed-form Φ and Q matrices
used by the Kalman update loop.
"""
Base.@kwdef struct TwoStateClock <: AbstractClockModel
    tau::Float64
    R::Float64  = 0.0 # WPM (measurement)
    σ1::Float64 = 0.0 # WFM (state)
    σ2::Float64 = 0.0 # RWFM (state)
end

"""
    ThreeStateClock(; tau, R=0.0, σ1=0.0, σ2=0.0, σ3=0.0)

Three-state polynomial clock model with state vector
`[phase, frequency, frequency_drift]`. Adds a random-run FM (RRFM /
drift) channel with diffusion coefficient `σ3` over
[`TwoStateClock`](@ref); meanings of `tau`, `R`, `σ1`, `σ2` are
identical. Suited to clocks with non-negligible drift such as ageing
cesium tubes or GPS-grade rubidiums over long horizons.
"""
Base.@kwdef struct ThreeStateClock <: AbstractClockModel
    tau::Float64
    R::Float64  = 0.0 # WPM (measurement)
    σ1::Float64 = 0.0 # WFM (state)
    σ2::Float64 = 0.0 # RWFM (state)
    σ3::Float64 = 0.0 # RRFM (state)
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
RWFM / RRFM diffusion coefficients `σ1, σ2, σ3` on `model`. Matches
the Zucca–Tavella 2005 derivation. Returned as an `SMatrix` for
Kalman composition.

The `dt` overload is what `prop!` uses to integrate over arbitrary
horizons.
"""
function process_noise(m::TwoStateClock, dt::Real)
    τ = Float64(dt)
    Q11 = m.σ1*τ + m.σ2*τ^3/3.0
    Q12 = m.σ2*τ^2/2.0
    Q22 = m.σ2*τ
    @SMatrix [Q11 Q12; Q12 Q22]
end

function process_noise(m::ThreeStateClock, dt::Real)
    τ = Float64(dt)
    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = m.σ1*τ + m.σ2*τ3/3.0 + m.σ3*τ5/20.0
    Q12 = m.σ2*τ2/2.0 + m.σ3*τ4/8.0
    Q13 = m.σ3*τ3/6.0
    Q22 = m.σ2*τ + m.σ3*τ3/3.0
    Q23 = m.σ3*τ2/2.0
    Q33 = m.σ3*τ
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
the WPM diffusion coefficient `model.R`. Identical for both shipped
clock types since the measurement is phase-only WPM. Consumed by
`update!` when forming the innovation covariance `S = H P Hᵀ + R`.
"""
measurement_noise(m::TwoStateClock)   = @SMatrix [m.R]
measurement_noise(m::ThreeStateClock) = @SMatrix [m.R]
