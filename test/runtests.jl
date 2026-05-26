using Test
using LinearAlgebra
using StaticArrays
using ClockEnsemble

@testset "ClockEnsemble" begin

    # ── PID steering controller ──────────────────────────────────────────────
    @testset "PIDController.step!" begin
        # Zero state → zero steer.
        pid = PIDController()
        @test step!(pid, [0.0, 0.0, 0.0]) == 0.0

        # Positive phase → negative steer (drives toward zero).
        pid = PIDController(g_p=0.1, g_i=0.0, g_d=0.0)
        s1 = step!(pid, [1.0, 0.0, 0.0])
        @test s1 == -0.1
        @test pid.last_steer == -0.1

        # Integral term accumulates.
        pid = PIDController(g_p=0.0, g_i=0.5, g_d=0.0)
        step!(pid, [1.0, 0.0])    # sumx = 1.0 → -0.5
        step!(pid, [1.0, 0.0])    # sumx = 2.0 → -1.0
        @test pid.last_steer == -1.0

        # Derivative term picks up frequency state.
        pid = PIDController(g_p=0.0, g_i=0.0, g_d=0.2)
        s = step!(pid, [0.0, 1.5, 0.0])
        @test s ≈ -0.3
    end

    @testset "Predict with steering correction" begin
        import Random; Random.seed!(7)

        N   = 200
        tau = 1.0
        # Constant-frequency-offset clock: phase grows linearly with τ·offset.
        f_offset = 1e-9
        data = [k * tau * f_offset + 1e-12 * randn() for k in 0:N-1]

        model = ThreeStateClock(tau=tau, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)
        est   = KalmanFilter([data[1], 0.0, 0.0], 1e-12 * Matrix(I(3)))
        pid   = PIDController(g_p=0.5, g_i=0.05, g_d=0.1)

        for k in 1:N
            corr = steer_to_correction(pid.last_steer, 3, tau)
            predict!(est, model, tau; steering=corr)
            update!(est, model, data[k])
            step!(pid, est.x)
        end

        # PID should have driven the residual phase down to a small fraction
        # of the unsteered drift after N=200 steps.
        unsteered_endpoint = N * tau * f_offset
        @test abs(est.x[1]) < 1e-2 * unsteered_endpoint
    end

    # ── TwoStateClock smoke test ─────────────────────────────────────────────
    @testset "TwoStateClock basic run" begin
        import Random; Random.seed!(99)

        N   = 50
        tau = 1.0
        data = cumsum(randn(N) * 1e-10)

        model = TwoStateClock(tau=tau, R=1e-2, σ1=1e-3, σ2=1e-4)
        est   = KalmanFilter([data[1], 0.0], Matrix(1.0 * I(2)))

        for k in 1:N
            predict!(est, model, tau)
            update!(est, model, data[k])
        end

        @test length(est.x) == 2
        @test size(est.P) == (2, 2)
        @test est.k == N
    end

    # ── prop!: covariance-only propagation ───────────────────────────────────
    @testset "state_transition / process_noise dt-overload parity" begin
        # The single-arg methods must equal the two-arg form at dt = model.tau,
        # bit-exact — locks that the dt refactor introduced no drift.
        m2 = TwoStateClock(tau=2.5, R=1e-22, σ1=1e-23, σ2=1e-33)
        m3 = ThreeStateClock(tau=2.5, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)

        @test state_transition(m2)             == state_transition(m2, m2.tau)
        @test state_transition(m3)             == state_transition(m3, m3.tau)
        @test Matrix(process_noise(m2))        == Matrix(process_noise(m2, m2.tau))
        @test Matrix(process_noise(m3))        == Matrix(process_noise(m3, m3.tau))

        # Φ scales linearly in dt for the polynomial integrator.
        @test state_transition(m3, 1.0)[1, 2]  == 1.0
        @test state_transition(m3, 5.0)[1, 2]  == 5.0
        @test state_transition(m3, 5.0)[1, 3]  == 5.0^2 / 2.0
    end

    @testset "prop! Q-integration parity" begin
        # Hand-derive Q(dt) for ThreeStateClock and check the SMatrix output.
        σ1 = 1e-23; σ2 = 1e-33; σ3 = 1e-43
        dt = 0.7
        Q11 = σ1*dt + σ2*dt^3/3 + σ3*dt^5/20
        Q12 = σ2*dt^2/2 + σ3*dt^4/8
        Q13 = σ3*dt^3/6
        Q22 = σ2*dt    + σ3*dt^3/3
        Q23 = σ3*dt^2/2
        Q33 = σ3*dt
        Q_expected = [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]

        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=σ1, σ2=σ2, σ3=σ3)
        @test Matrix(process_noise(m, dt)) ≈ Q_expected atol=0.0 rtol=1e-14
    end

    @testset "prop! single-step propagation" begin
        # Φ(dt) x and Φ(dt) P Φ(dt)' + Q(dt) — exact match against manual math.
        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)
        x0 = [3.0, 1e-10, 1e-15]
        P0 = Matrix(1e-18 * I(3))

        est = KalmanFilter(copy(x0), copy(P0))
        prop!(est, m, 1.5)

        Phi = Matrix(state_transition(m, 1.5))
        Q   = Matrix(process_noise(m, 1.5))
        @test Vector(est.x) ≈ Phi * x0       atol=0.0 rtol=1e-14
        @test Matrix(est.P) ≈ Phi * P0 * Phi' + Q  atol=0.0 rtol=1e-14
        # prop! must NOT increment k.
        @test est.k == 0
    end

    @testset "prop! group / additivity composition" begin
        # Two prop!s of dt₁ then dt₂ must equal one prop! of dt₁+dt₂ exactly,
        # because Φ has the group property and Q is additive under it:
        #   Q(dt₁+dt₂) = Φ(dt₂) Q(dt₁) Φ(dt₂)' + Q(dt₂)
        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)
        x0 = [1.0, 1e-10, 1e-16]
        P0 = Matrix(1e-18 * I(3))

        a = KalmanFilter(copy(x0), copy(P0))
        prop!(a, m, 0.4)
        prop!(a, m, 0.6)

        b = KalmanFilter(copy(x0), copy(P0))
        prop!(b, m, 1.0)

        @test Vector(a.x)  ≈ Vector(b.x)  atol=0.0 rtol=1e-14
        @test Matrix(a.P)  ≈ Matrix(b.P)  atol=0.0 rtol=1e-14
    end

    @testset "prop! parity with predict! after k>0" begin
        # Once est.k > 0, predict! and prop! must produce identical state/covariance
        # when called with dt = model.tau (since the gate is the only difference).
        import Random; Random.seed!(123)

        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)
        x0 = [0.0, 0.0, 0.0]
        P0 = Matrix(1e-12 * I(3))

        a = KalmanFilter(copy(x0), copy(P0))
        b = KalmanFilter(copy(x0), copy(P0))

        # Drive both filters past the k>0 gate with one update.
        update!(a, m, 1e-9)
        update!(b, m, 1e-9)
        @test a.k == 1 && b.k == 1

        # Step forward identically with predict! vs prop!.
        for _ in 1:10
            predict!(a, m, m.tau)
            prop!(b, m, m.tau)
        end

        @test Vector(a.x) ≈ Vector(b.x) atol=0.0 rtol=1e-14
        @test Matrix(a.P) ≈ Matrix(b.P) atol=0.0 rtol=1e-14
        # Crucially, prop! did not bump k.
        @test a.k == 1 && b.k == 1
    end

    @testset "prop! steering correction" begin
        # Steering vector adds to the predicted state mean exactly as in predict!.
        m = TwoStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33)
        x0 = [0.0, 0.0]
        P0 = Matrix(1e-18 * I(2))

        u = -3.0e-10              # PID-style frequency correction
        dt = 0.5
        steer = steer_to_correction(u, 2, dt)

        est = KalmanFilter(copy(x0), copy(P0))
        prop!(est, m, dt; steering=steer)

        # Φ(dt)·x₀ = 0; steering adds [u·dt, u].
        @test est.x[1] ≈ u * dt atol=0.0 rtol=1e-14
        @test est.x[2] ≈ u      atol=0.0 rtol=1e-14
    end

    @testset "prop! covariance band over horizons (holdover example pattern)" begin
        # Operationally: build a 1σ covariance band around a deterministic
        # forward projection by prop!ing a side-channel estimator from a fixed
        # starting P0 over each horizon h·τ. Check monotonic growth of σ_x(τ),
        # which is the property the holdover plot relies on.
        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=0.0)
        x0 = [0.0, 0.0, 0.0]
        P0 = Matrix(1e-24 * I(3))

        horizons = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        sigmas = Float64[]
        for h in horizons
            est = KalmanFilter(copy(x0), copy(P0))
            prop!(est, m, h)
            push!(sigmas, sqrt(est.P[1, 1]))
        end

        @test all(diff(sigmas) .> 0.0)
    end

    # ── Internal helpers compose to the bundled functions ────────────────────
    @testset "Kalman helpers ↔ update! parity" begin
        # The named per-step helpers (innovation, innovation_cov, kalman_gain,
        # aposteriori_state, aposteriori_cov) must reproduce update! exactly
        # when composed in order. This locks the pedagogical building blocks
        # against silent drift from the high-level wrappers.
        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=0.0)

        # Drive a filter past the k>0 gate, then take a snapshot.
        ref = KalmanFilter([0.0, 0.0, 0.0], Matrix(1e-12 * I(3)))
        update!(ref, m, 5.0e-10)
        predict!(ref, m, m.tau)
        x_pre = ref.x
        P_pre = SMatrix(ref.P)

        # Bundled path: update! folds in z.
        z = 7.3e-10
        a = KalmanFilter(Vector(x_pre), Matrix(P_pre))
        update!(a, m, z)

        # Hand-composed path: call the helpers in textbook order.
        H = ClockEnsemble.measurement_matrix(m)
        R = ClockEnsemble.measurement_noise(m)
        z_vec = SVector{1, Float64}(z)
        ỹ = ClockEnsemble.innovation(z_vec, H, x_pre)
        S = ClockEnsemble.innovation_cov(P_pre, H, R)
        K = ClockEnsemble.kalman_gain(P_pre, H, S)
        x_post = ClockEnsemble.aposteriori_state(x_pre, K, ỹ)
        P_post = ClockEnsemble.aposteriori_cov(P_pre, K, H, R)

        @test Vector(a.x) ≈ Vector(x_post) atol=0.0 rtol=1e-14
        @test Matrix(a.P) ≈ Matrix(P_post) atol=0.0 rtol=1e-14
    end

    @testset "Kalman helpers ↔ predict! parity" begin
        # apriori_state / apriori_cov must reproduce predict! when est.k > 0.
        m = ThreeStateClock(tau=1.0, R=1e-22, σ1=1e-23, σ2=1e-33, σ3=1e-43)

        ref = KalmanFilter([0.0, 0.0, 0.0], Matrix(1e-12 * I(3)))
        update!(ref, m, 1.0e-9)
        x_pre = ref.x
        P_pre = SMatrix(ref.P)

        dt = 0.75
        Φ = ClockEnsemble.state_transition(m, dt)
        Q = ClockEnsemble.process_noise(m, dt)

        a = KalmanFilter(Vector(x_pre), Matrix(P_pre))
        a.k = ref.k                     # carry the gate state across
        predict!(a, m, dt)

        x_pred = ClockEnsemble.apriori_state(x_pre, Φ)
        P_pred = ClockEnsemble.apriori_cov(P_pre, Φ, Q)

        @test Vector(a.x) ≈ Vector(x_pred) atol=0.0 rtol=1e-14
        @test Matrix(a.P) ≈ Matrix(P_pred) atol=0.0 rtol=1e-14
    end

end
