using FMin
using Test

@testset "FMin.jl" begin
    tol = 1e-8
    ax = -4.0
    bx = 0.0
    x = -pi/2
    xmin = fmin(ax, bx, sin, tol)
    @test abs(xmin - x) <= 10*tol
end
