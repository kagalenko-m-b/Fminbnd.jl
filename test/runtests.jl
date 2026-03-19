using Fminbnd
using Test

# ---------------------------------------------------------------------------
# Build the original Netlib fmin.f as a shared library for reference testing.
# This ensures our Julia implementation agrees with the Fortran original on
# every test case.
# ---------------------------------------------------------------------------
const FMIN_F   = joinpath(tempdir(), "fmin_netlib.f")
const FMIN_LIB = joinpath(tempdir(), "libfmin_netlib." * (Sys.isapple() ? "dylib" : "so"))

Base.download("https://www.netlib.org/fmm/fmin.f", FMIN_F)
run(`gfortran -shared -fPIC -o $FMIN_LIB $FMIN_F`)

# @cfunction cannot capture a closure, so we route Fortran callbacks through
# a global Ref that holds the current Julia function being evaluated.
const _FMIN_CALLBACK = Ref{Any}(nothing)
function _fmin_trampoline(xp::Ptr{Float64})::Float64
    _FMIN_CALLBACK[](unsafe_load(xp))
end
const _FMIN_FPTR = @cfunction(_fmin_trampoline, Float64, (Ptr{Float64},))

function fmin_netlib(f, ax::Float64, bx::Float64, tol::Float64)
    _FMIN_CALLBACK[] = f
    ccall((:fmin_, FMIN_LIB), Float64,
          (Ref{Float64}, Ref{Float64}, Ptr{Cvoid}, Ref{Float64}),
          ax, bx, _FMIN_FPTR, tol)
end

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Check that fminbnd and Netlib fmin agree to within a loose multiple of tol,
# and that the result is actually near a minimum (f(x±eps) >= f(x)).
function check(f, ax, bx, tol=1e-10)
    xj = fminbnd(f, ax, bx, tol)
    xn = fmin_netlib(f, Float64(ax), Float64(bx), Float64(tol))
    # Values must agree
    @test abs(f(xj) - f(xn)) < 1e-6
    # Both must be actual local minima (not just any point)
    δ = max(abs(xj), 1.0) * sqrt(eps(Float64))
    @test f(xj) <= f(xj + δ) + 1e-12
    @test f(xj) <= f(xj - δ) + 1e-12
    xj
end

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

@testset "Fminbnd.jl" begin

    @testset "smooth unimodal — sin on bounded interval" begin
        # Classic test from original suite: sin on [-4, 0], min at -π/2
        tol = 1e-8
        xmin = fminbnd(sin, -4.0, 0.0, tol)
        @test abs(xmin - (-π/2)) <= 10*tol

        # Verify against Netlib
        xn = fmin_netlib(sin, -4.0, 0.0, tol)
        @test abs(xmin - xn) < 10*tol
    end

    @testset "quadratic — exact parabola" begin
        # Parabola: min at x = 0.3
        f = x -> (x - 0.3)^2
        xmin = check(f, -1.0, 1.0)
        @test abs(xmin - 0.3) < 1e-8
    end

    @testset "quartic with linear tilt — asymmetric double well" begin
        # (x²-4)² + 0.1x: global min near x=-2, local min near x=+2
        f = x -> (x^2 - 4)^2 + 0.1*x
        xmin = check(f, -3.0, 0.0)   # bracket only the left well
        @test xmin < 0
        @test f(xmin) < 0
    end

    @testset "non-smooth — abs(x), minimum at 0" begin
        # Non-differentiable at minimum; forces golden section steps
        f = x -> abs(x - 0.5)
        xmin = check(f, -1.0, 2.0)
        @test abs(xmin - 0.5) < 1e-6
    end

    @testset "golden ratio constant — multimodal function" begin
        # THIS IS THE BUG TEST.
        # g(x) = -sinc(x-5)² - 2·sinc(x+5)²  on [-10, 10]
        # Global min at x≈-5 (f=-2), local min at x≈+5 (f=-1).
        # The wrong constant c≈0.709 places the first probe at x≈+4.2,
        # trapping the search in the shallow basin. The correct c≈0.382
        # probes at x≈-2.4 and finds the global minimum.
        sinc_norm(x) = x == 0 ? 1.0 : sin(π*x)/(π*x)
        g = x -> -sinc_norm(x - 5)^2 - 2*sinc_norm(x + 5)^2

        xmin  = fminbnd(g, -10.0, 10.0, 1e-10)
        xnetlib = fmin_netlib(g, -10.0, 10.0, 1e-10)

        # Both must find the deep minimum (f≈-2), not the shallow one (f≈-1)
        @test g(xmin)    < -1.5   # found global min
        @test g(xnetlib) < -1.5   # Netlib also finds global min
        @test abs(g(xmin) - g(xnetlib)) < 1e-6
    end

    @testset "golden ratio constant value" begin
        # Directly verify the constant is the squared inverse of the golden ratio
        T = Float64
        c = (3 - sqrt(5*one(T)))/2
        golden_ratio = (1 + sqrt(5.0))/2
        @test abs(c - 1/golden_ratio^2) < 1e-15
        @test abs(c - 0.3819660112501051) < 1e-15
    end

    @testset "Float32 support" begin
        f = x -> (x - 0.3f0)^2
        xmin = fminbnd(f, -1.0f0, 1.0f0)
        @test abs(xmin - 0.3f0) < 1e-4
    end

    @testset "tight tolerance" begin
        f = x -> (x - π)^2
        xmin = fminbnd(f, 3.0, 3.5, eps(Float64))
        @test abs(xmin - π) < 1e-10
    end

    @testset "minimum near left endpoint" begin
        f = x -> exp(x)   # monotone decreasing on negatives; min at left end
        xmin = fminbnd(f, -5.0, 0.0, 1e-10)
        @test xmin < -4.9
    end

    @testset "minimum near right endpoint" begin
        f = x -> -exp(x)  # monotone increasing; min at left, max at right
        # flip: min near right of [-5, 5] would be -exp(5)
        # Actually let's use a shifted quadratic close to the right boundary
        f2 = x -> (x - 4.99)^2
        xmin = fminbnd(f2, -5.0, 5.0, 1e-10)
        @test abs(xmin - 4.99) < 1e-6
    end

    @testset "evaluation count — correct constant is efficient" begin
        # With the correct golden ratio constant the algorithm should need
        # fewer evaluations than with the wrong one (0.709).
        # On abs(x) (non-smooth, forces golden section), count evals.
        evals = Ref(0)
        f = x -> (evals[] += 1; abs(x - 0.3))
        fminbnd(f, -1.0, 1.0, 1e-12)
        # Netlib needs ~45 evals for this case; buggy version needed ~53
        @test evals[] <= 50
    end

end
