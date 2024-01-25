module FMin

export fmin
"""
    x = fmin(ax, bx, f, tol)


    an approximation  x  to the point where  f  attains a minimum  on
the interval  (ax,bx)  is determined.


input..

ax    left endpoint of initial interval
bx    right endpoint of initial interval
f     function subprogram which evaluates  f(x)  for any  x
      in the interval  (ax,bx)
tol   desired length of the interval of uncertainty of the final
      result ( .ge. 0.0d0)


output..

fmin  abcissa approximating the point where  f  attains a minimum


    the method used is a combination of  golden  section  search  and
successive parabolic interpolation.  convergence is never much slower
than  that  for  a  fibonacci search.  if  f  has a continuous second
derivative which is positive at the minimum (which is not  at  ax  or
bx),  then  convergence  is  superlinear, and usually of the order of
about  1.324....
    the function  f  is never evaluated at two points closer together
than  eps*abs(fmin) + (tol/3), where eps is  approximately the square
root  of  the  relative  machine  precision.   if   f   is a unimodal
function and the computed values of   f   are  always  unimodal  when
separated by at least  eps*abs(x) + (tol/3), then  fmin  approximates
the abcissa of the global minimum of  f  on the interval  ax,bx  with
an error less than  3*eps*abs(fmin) + tol.  if   f   is not unimodal,
then fmin may approximate a local, but perhaps non-global, minimum to
the same accuracy.
    this function subprogram is a slightly modified  version  of  the
algol  60 procedure  localmin  given in richard brent, algorithms for
minimization without derivatives, prentice - hall, inc. (1973).

###

Rewrite in Julia of [fmin from Netlib](http://www.netlib.org/fmm/fmin.f)
by Mikhail Kagalenko, January 2024
"""
function fmin(ax::T, bx::T, f::Function, tol::T=eps(T)) where T<:AbstractFloat
    #
    #  c is the squared inverse of the golden ratio
    #
    c = (3 - sqrt(5*one(T)/2))/2
    #
    #  eps is the square root of the relative machine precision.
    #
    sqrt_eps = sqrt(eps(T))
    #
    #  initialization
    #
    a = ax
    b = bx
    v = a + c*(b - a)
    w = v
    x = v
    e = zero(T)
    fx = f(x)
    fv = fx
    fw = fx
    #
    #  main loop starts here
    #
    @label L20
    xm = (a + b)/2
    tol1 = sqrt_eps*abs(x) + tol/3
    tol2 = 2*tol1
    #
    #  check stopping criterion
    #
    if abs(x - xm) <= (tol2 - (b - a)/2);  @goto L90; end
    #
    # is golden-section necessary
    #
    if (abs(e) <= tol1); @goto L40; end
    #
    #  fit parabola
    #
    r = (x - w)*(fx - fv)
    q = (x - v)*(fx - fw)
    p = (x - v)*q - (x - w)*r
    q = 2*(q - r)
    if (q > 0); p = -p; end
    q =  abs(q)
    r = e
    e = d
    #
    #  is parabola acceptable
    #
    # @label L30
    if (abs(p) >= abs(q*r/2)); @goto L40; end
    if (p <= q*(a - x)); @goto L40; end
    if (p >= q*(b - x)); @goto L40; end
    #
    #  a parabolic interpolation step
    #
    d = p/q
    u = x + d
    #
    #  f must not be evaluated too close to ax or bx
    #
    if ((u - a) < tol2); d = copysign(tol1, xm - x); end
    if ((b - u) < tol2); d = copysign(tol1, xm - x); end
    @goto L50
    #
    #  a golden-section step
    #
    @label L40
    if (x >= xm); e = a - x; end
    if (x < xm); e = b - x; end
    d = c*e
    #
    #  f must not be evaluated too close to x
    #
    @label L50
    if (abs(d) >= tol1); u = x + d; end
    if (abs(d) < tol1); u = x + copysign(tol1, d); end
    fu = f(u)
    #
    #  update  a, b, v, w, and x
    #
    if (fu > fx); @goto L60; end
    if (u >= x); a = x; end
    if (u < x); b = x; end
    v = w
    fv = fw
    w = x
    fw = fx
    x = u
    fx = fu
    @goto L20
    @label L60
    if (u < x); a = u; end
    if (u >= x); b = u; end
    if (fu <= fw) @goto L70; end
    if (w == x); @goto L70; end
    if (fu <= fv); @goto L80; end
    if (v == x); @goto L80; end
    if (v == w); @goto L80; end
    @goto L20
    @label L70
    v = w
    fv = fw
    w = u
    fw = fu
    @goto L20
    @label L80
    v = u
    fv = fu
    @goto L20
    #
    #  end of main loop
    #
    @label L90
    fmin = x
    return fmin
end

end # module
