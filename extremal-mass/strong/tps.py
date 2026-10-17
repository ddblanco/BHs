"""Minimal truncated power series in x; coefficients in a numpy array
(float, complex, or object for mpmath)."""
import numpy as np


def _dtype(v):
    if isinstance(v, np.ndarray):
        return v.dtype
    if isinstance(v, (complex, np.complexfloating)):
        return np.complex128
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.float64
    return object


class TPS:
    __array_priority__ = 1000
    K = 12

    def __init__(self, c, dtype=None):
        c = np.asarray(c) if dtype is None else np.asarray(c, dtype=dtype)
        if c.dtype.kind in 'iub':
            c = c.astype(float)
        self.c = np.zeros(TPS.K+1, dtype=c.dtype)
        n = min(len(c), TPS.K+1)
        self.c[:n] = c[:n]

    def _co(self, o):
        if isinstance(o, TPS):
            return o
        dt = np.result_type(self.c.dtype, _dtype(o)) if _dtype(o) is not object else object
        z = TPS(np.zeros(TPS.K+1, dtype=dt))
        z.c[0] = o
        return z

    def __add__(self, o): o = self._co(o); return TPS(self.c+o.c)
    __radd__ = __add__
    def __neg__(self): return TPS(-self.c)
    def __sub__(self, o): return self+(-self._co(o))
    def __rsub__(self, o): return self._co(o)-self

    def __mul__(self, o):
        if not isinstance(o, TPS):
            return TPS(self.c*o)
        a, b = self.c, o.c
        out = np.zeros(TPS.K+1, dtype=np.result_type(a.dtype, b.dtype))
        for i in range(TPS.K+1):
            if a[i] != 0:
                out[i:] += a[i]*b[:TPS.K+1-i]
        return TPS(out)
    __rmul__ = __mul__

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out = self._co(1.)
        base = self
        while n:
            if n & 1: out = out*base
            base = base*base
            n >>= 1
        return out

    def deriv(self):
        d = np.zeros(TPS.K+1, dtype=self.c.dtype)
        d[:TPS.K] = self.c[1:]*np.arange(1, TPS.K+1)
        return TPS(d)

    def shift(self, k):
        d = np.zeros(TPS.K+1, dtype=self.c.dtype)
        if k >= 0:
            d[k:] = self.c[:TPS.K+1-k]
        else:
            d[:TPS.K+1+k] = self.c[-k:]
        return TPS(d)
