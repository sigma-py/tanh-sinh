<p align="center">
  <a href="https://github.com/nschloe/tanh_sinh"><img alt="logo" src="https://nschloe.github.io/tanh_sinh/logo-tanhsinh.svg" width="60%"></a>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/tanh_sinh.svg?style=flat-square)](https://pypi.org/project/tanh_sinh)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tanh_sinh.svg?style=flat-square)](https://pypi.org/pypi/tanh_sinh/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/tanh_sinh.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/tanh_sinh)
[![PyPi downloads](https://img.shields.io/pypi/dm/tanh_sinh.svg?style=flat-square)](https://pypistats.org/packages/tanh_sinh)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/Z6DMsJh4Hr)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/tanh_sinh/ci?style=flat-square)](https://github.com/nschloe/tanh_sinh/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/tanh_sinh.svg?style=flat-square)](https://codecov.io/gh/nschloe/tanh_sinh)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/tanh_sinh.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/tanh_sinh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

The rather modern tanh-sinh quadrature is different from classical Gaussian integration
methods in that it doesn't integrate any function exactly, not even polynomials of low
degree. Its tremendous usefulness rather comes from the fact that a wide variety of
functions, even seemingly difficult ones with (integrable) singularities, can be
integrated with _arbitrary_ precision.

Install with

```
pip install tanh-sinh
```

and use it like

```python
import tanh_sinh
import numpy as np

val, error_estimate = tanh_sinh.integrate(
    lambda x: np.exp(x) * np.cos(x),
    0,
    np.pi / 2,
    1.0e-14,
    # Optional: Specify first and second derivative for better error estimation
    # f_derivatives={
    #     1: lambda x: np.exp(x) * (np.cos(x) - np.sin(x)),
    #     2: lambda x: -2 * np.exp(x) * np.sin(x),
    # },
)
```

If you want more digits, use [mpmath](http://mpmath.org/) for arbitrary precision
arithmetic:

```python
import tanh_sinh
from mpmath import mp
import sympy

mp.dps = 50

val, error_estimate = tanh_sinh.integrate(
    lambda x: mp.exp(x) * sympy.cos(x),
    0,
    mp.pi / 2,
    1.0e-50,  # !
    mode="mpmath",
)
```

If the function has a singularity at a boundary, it needs to be shifted such that the
singularity is at 0. (This is to avoid round-off errors for points that are very close
to the singularity.)
If there are singularities at both ends, the function can be shifted both ways and be
handed off to `integrate_lr`; For example, for the function `1 / sqrt(1 - x**2)`, this
gives

```python
import numpy
import tanh_sinh

# def f(x):
#    return 1 / numpy.sqrt(1 - x ** 2)

val, error_estimate = tanh_sinh.integrate_lr(
    lambda x: 1 / numpy.sqrt(-(x ** 2) + 2 * x),  # = 1 / sqrt(1 - (x-1)**2)
    lambda x: 1 / numpy.sqrt(-(x ** 2) + 2 * x),  # = 1 / sqrt(1 - (-(x-1))**2)
    2,  # length of the interval
    1.0e-10,
)
print(numpy.pi)
print(val)
```

<!--pytest-codeblocks:expected-output--->
```
3.141592653589793
3.1415926533203944
```

### Relevant publications

- [Hidetosi Takahasi, Masatake Mori, Double Exponential Formulas for Numerical Integration, PM. RIMS, Kyoto Univ., 9 (1974), 721-741](https://doi.org/10.2977%2Fprims%2F1195192451)
- [Masatake Mori, Discovery of the double exponential transformation and its developments, Publications of the Research Institute for Mathematical Sciences, 41 (4): 897–935, ISSN 0034-5318](https://doi.org/10.2977/prims/1145474600)
- [David H. Bailey, Karthik Jeyabalan, and Xiaoye S. Li, Error function quadrature, Experiment. Math., Volume 14, Issue 3 (2005), 317-329](https://projecteuclid.org/euclid.em/1128371757)
- [David H. Bailey, Tanh-Sinh High-Precision Quadrature, 2006](https://www.davidhbailey.com/dhbpapers/dhb-tanh-sinh.pdf)

### Testing

To run the unit tests, check out this repository and type

```
tox
```

### License

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
