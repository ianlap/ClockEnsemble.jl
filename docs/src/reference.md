# API Reference

## Clock models

```@docs
AbstractClockModel
TwoStateClock
ThreeStateClock
nstates
state_transition
process_noise
measurement_matrix
measurement_noise
```

## Kalman filter

```@docs
KalmanFilter
predict!
update!
prop!
```

## Steering

```@docs
PIDController
step!
steer_to_correction
```
