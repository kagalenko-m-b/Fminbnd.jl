# Fminbnd

[![Build Status](https://github.com/kagalenko-m-b/Fminbnd.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kagalenko-m-b/Fminbnd.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package provides a single function `fminbnd()` that searches on a bounded
interval for the minimum point of a univariate function. It is a literal rewrite in Julia
of Fortran routine [fmin](http://www.netlib.org/fmm/fmin.f) from Netlib.

This function uses the same algorithm and similar calling convention to Matlab/Octave
`fminbnd()` function.
