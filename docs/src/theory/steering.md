# Theory: Clock Steering with PID Controllers

Clock steering closes the loop between the Kalman estimator described
in [Kalman filter](kalman.md) and the physical oscillator's frequency
control input. Where the filter produces phase and frequency
estimates, the steering controller maps those estimates to a frequency
correction applied to the oscillator (or to the digital realisation of
the oscillator's output) so that the closed-loop clock tracks the
reference rather than drifting under its free-running noise.

This page covers the proportional-integral-derivative (PID) clock-
steering controller of
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers): the
augmented state-space form, the cubic phase-recursion characteristic,
the critical-gain surface, and the closed-loop transfer function used
to predict steered-clock Allan deviation from the free-running PSD.

## PID clock-steering controller

The PID clock-steering controller computes a frequency steer as the
negative dot product of a 3-component gain vector with a state vector
containing the integral of phase, the phase, and the frequency
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers):

```math
u \;=\; -\,G\,X \;=\; -\,(g_I,\, g_P,\, g_D)\cdot (I,\, p,\, f)^{\top}.
```

The integral state $I$ is **excluded** from the Kalman estimator's
measurement update because it is unobservable; it appears only in the
steering gain
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). In
the absence of unmodelled systematic errors, the minimum steady-state
variance of phase, frequency, and steer (PFS) occurs at zero integral
gain, so the integral term raises variance but provides protection
against unmodelled drift
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers).

### Augmented state-space form

The augmented evolution and input matrices for the PID-augmented state
are

```math
X_k \;=\; \begin{pmatrix} I_k \\ p_k \\ f_k \end{pmatrix}, \qquad
\Phi \;=\; \begin{pmatrix} 1 & 1 & \tau \\ 0 & 1 & \tau \\ 0 & 0 & 1 \end{pmatrix}, \qquad
B \;=\; \begin{pmatrix} \tau \\ \tau \\ 1 \end{pmatrix}, \qquad
G \;=\; (g_I,\, g_P,\, g_D).
```

Matsakis uses uppercase `Φ` for the augmented 3×3 evolution including
the integral state, and lowercase `φ` for the underlying 2×2 Kalman
evolution; downstream notes should be careful when cross-referencing
other state-estimation sources that use `Φ` for the Kalman evolution
itself
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers).

The Kalman gain and pre-/post-measurement covariances satisfy a
discrete-time algebraic Riccati equation `dare(Φ_d, H_d, Q, R)` and
are **independent of the steering gains**, because steering is
deterministic
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). The
closed-loop steady-state covariance solves a discrete Lyapunov
equation in `Φ(1 − k_p H)`, with control variance

```math
\langle u^{2} \rangle \;=\; G\,\Sigma_X\,G^{\top}.
```

Steer variance is therefore a quadratic form in the gain vector, and
direct gain selection trades off control activity against tracking
error
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers).

### Cubic characteristic and critical gains

The phase-recursion characteristic polynomial is the cubic

```math
r^{3} + (-3 + g_I + g_P + g_D)\,r^{2} + (3 - g_P - 2 g_D)\,r + (-1 + g_D) \;=\; 0.
```

The roots determine stability and time constants of the controlled
clock
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). The
critical gains satisfying a triple real root `r = e^{−τ/T}` for a
target time constant `T` are

```math
\tau\, g_P \;=\; 1 - 3 e^{-2\tau/T} + 2 e^{-3\tau/T}, \qquad
\tau\, g_I \;=\; 1 - 3 e^{-\tau/T} + 3 e^{-2\tau/T} - e^{-3\tau/T}, \qquad
g_D \;=\; 1 - e^{-3\tau/T}.
```

The PD critical gains `g_P = (1 − r)²`, `g_D = 1 − r²`, `g_I = 0` also
solve the PID cubic, so PD is the `g_I = 0` corner of the PID
critical surface
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). Larger
`g_I` pushes the stable region upward in `(g_P, g_D)` space; higher
integral gain demands higher minimum frequency gain
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers).

For an unmodelled frequency drift of 200 ns/day², a small nonzero
`g_I` dramatically improves phase RMS, vanishing for `g_I > 1.5`. A
heuristic for the variance penalty: an unfiltered phase outlier
permanently shifts the integral, causing several future phase points
to overshoot in the opposite direction before `I` returns to zero
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers).

### Closed-loop transfer function

The closed-loop transfer function in the `z`-domain is

```math
H_{cl}(z) \;=\; \frac{g_P\, z\,(z - 1) + g_D\,(z - 1)^{2} + g_I\, z^{2}}{(z - 1)^{3} + g_P\, z\,(z - 1) + g_D\,(z - 1)^{2} + g_I\, z^{2}}.
```

The Allan variance of the steered clock follows from

```math
\sigma_y^{2}(\tau) \;=\; \int_{0}^{\infty} df\;\bigl|H_{cl}(f)\bigr|^{2}\, S_y(f) \cdot \frac{2\,[\sin(\pi\tau f)]^{4}}{(\pi\tau f)^{2}}
```

once the oscillator's free-running PSD `S_y(f)` is known and
measurement noise is negligible
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). This
turns gain selection into a frequency-domain shaping exercise: pick
`(g_I, g_P, g_D)` to attenuate the regions of `S_y(f)` that dominate
the desired `τ` window, subject to the cubic stability constraint
above.

## ClockEnsemble implementation

The PID steering controller is exposed in `ClockEnsemble` as
[`PIDController`](@ref), with the per-step update wired through
[`step!`](@ref) and the bridge to the underlying SDE state through
[`steer_to_correction`](@ref). Steering folds into the prediction
half of the Kalman recursion by passing the current correction into
[`predict!`](@ref) via its `steering=` keyword argument:

```julia
# After at least one update! so kf.x carries a posterior estimate:
step!(controller, kf.x)                                # advances controller, stores last_steer
corr = steer_to_correction(controller.last_steer,
                           nstates(model), dt)         # SVector{ns} with phase = u·dt, freq = u
predict!(kf, model, dt; steering = corr)               # Φ propagation + steering fold-in
update!(kf, model, z)                                  # measurement update
```

The integral state lives on `controller`, not on the Kalman state, so
the Kalman gains and covariances computed from `Q`, `R` are unchanged
by the steering loop — matching the Matsakis result that the Riccati
equation is independent of the controller gains
[Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers). See
[`examples/02_kalman_pid_steering.jl`](../tutorials/02_kalman_pid_steering.md)
for a complete worked closed-loop run.

For background on the SDE clock model whose frequency state the
controller drives, see
[Theory: Clock state-space models](ensembles.md). For the
Kalman filter that supplies `(p, f)` to the controller, see
[Theory: Kalman filter and variants](kalman.md).

## See also

- [Theory: Clock State-Space Models](ensembles.md) — the SDE
  whose frequency input the steer adjusts.
- [Theory: Kalman Filter and Variants](kalman.md) — supplies the
  optimal phase and frequency estimates the controller consumes.
- [API: `ClockEnsemble`](../reference.md) —
  [`PIDController`](@ref), [`step!`](@ref),
  [`steer_to_correction`](@ref).

## References

- [Matsakis & Coleman 2020](@cite matsakis-2020-pid-controllers) —
  PID clock-steering controllers for Kalman-filtered atomic clocks.
- [Howe & Tasset 2001](@cite howe-2001-tothvar-steering) — total-
  variance steering context underpinning the closed-loop stability
  metric.
