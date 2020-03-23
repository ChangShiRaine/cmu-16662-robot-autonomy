# Copyright (c) 2017, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import division, print_function, absolute_import

## Allow the code to function without numba, but discourage it
try:
    from numba import njit, jit, vectorize, int64, float64, complex128
    from numba.utils import IS_PY3
    GOT_NUMBA = True
except ImportError:
    import warnings
    import sys
    warning_text = \
        "\n\n" + "!" * 53 + "\n" + \
        "Could not import from numba, which means that some\n" + \
        "parts of this code may run MUCH more slowly.  You\n" + \
        "may wish to install numba." + \
        "\n" + "!" * 53 + "\n"
    warnings.warn(warning_text)
    def _identity_decorator_outer(*args, **kwargs):
        def _identity_decorator_inner(fn):
            return fn
        return _identity_decorator_inner
    njit = _identity_decorator_outer
    jit = _identity_decorator_outer
    vectorize = _identity_decorator_outer
    int64 = int
    float64 = float
    complex128 = complex
    IS_PY3 = (sys.version_info[:2] >= (3, 0))
    GOT_NUMBA = False

if IS_PY3:
    xrange = range
else:
    xrange = xrange
