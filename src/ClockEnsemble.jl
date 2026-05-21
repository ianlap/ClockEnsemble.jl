module ClockEnsemble

using DocStringExtensions
using LinearAlgebra
using StaticArrays

include("clocks.jl")
include("filters.jl")

export AbstractClockModel, TwoStateClock, ThreeStateClock
export nstates, state_transition, process_noise,
       measurement_matrix, measurement_noise
export KalmanFilter
export predict!, update!, prop!
export PIDController, step!, steer_to_correction

end # module
