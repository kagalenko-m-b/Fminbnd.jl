using Fminbnd
using Test

@testset "Fminbnd.jl" begin
    tol = 1e-8
    ax = -4.0
    bx = 0.0
    x = -pi/2
    xmin = fminbnd(ax, bx, sin, tol)
    @test abs(xmin - x) <= 10*tol
end
