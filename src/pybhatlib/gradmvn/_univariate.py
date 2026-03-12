"""Standard univariate normal distribution wrappers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def normal_pdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal probability density function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    pdf : ndarray
        phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_pdf(x)


def normal_cdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal cumulative distribution function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    cdf : ndarray
        Phi(x)
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_cdf(x)


def normal_ppf(p: NDArray, *, xp=None) -> NDArray:
    """Standard normal quantile function (inverse CDF).

    Parameters
    ----------
    p : ndarray
        Probabilities in (0, 1).
    xp : backend, optional

    Returns
    -------
    x : ndarray
        Phi^{-1}(p)
    """
    if xp is None:
        xp = array_namespace(p)
    return xp.normal_ppf(p)


def normal_logpdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal log probability density function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    logpdf : ndarray
        log(phi(x)) = -0.5 * (x^2 + log(2*pi))
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_logpdf(x)


# ---------------------------------------------------------------------------
# Genz (2004) BVND algorithm — high-precision bivariate normal CDF
# ---------------------------------------------------------------------------

# Gauss-Legendre weights and abscissae (symmetric pairs on [-1,1])
_GL6_W = np.array([
    0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
])
_GL6_X = np.array([
    0.9324695142031521, 0.6612093864662645, 0.2386191860831969,
])
_GL12_W = np.array([
    0.0471753363865118, 0.1069393259953184, 0.1600783285433462,
    0.2031674267230659, 0.2334925365383548, 0.2491470458134028,
])
_GL12_X = np.array([
    0.9815606342467192, 0.9041172563704749, 0.7699026741943047,
    0.5873179542866175, 0.3678314989981802, 0.1252334085114689,
])
_GL20_W = np.array([
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
    0.1527533871307258,
])
_GL20_X = np.array([
    0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
    0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
    0.5108670019508271, 0.3737060887154195, 0.2277858511416451,
    0.0765265211334973,
])


def _bvnd(dh: float, dk: float, r: float) -> float:
    """Compute P(X > dh, Y > dk) for standard bivariate normal with corr r.

    Direct translation of Alan Genz's BVND algorithm (TVPACK).
    Uses adaptive Gauss-Legendre quadrature (6/12/20 points) depending on |r|.
    """
    from scipy.stats import norm as _norm

    TWOPI = 2.0 * np.pi

    if abs(r) < 0.925:
        # Low-to-moderate correlation: direct GL quadrature
        if abs(r) < 0.3:
            ng = 3
            w, x = _GL6_W, _GL6_X
        elif abs(r) < 0.75:
            ng = 6
            w, x = _GL12_W, _GL12_X
        else:
            ng = 10
            w, x = _GL20_W, _GL20_X

        hk = dh * dk
        bvn = 0.0

        if abs(r) > 0:
            hs = (dh * dh + dk * dk) / 2.0
            asr = np.arcsin(r)
            for i in range(ng):
                for isign in (1, -1):
                    sn = np.sin(asr * (isign * x[i] + 1.0) / 2.0)
                    bvn += w[i] * np.exp((sn * hk - hs) / (1.0 - sn * sn))
            bvn *= asr / (2.0 * TWOPI)

        return bvn + _norm.cdf(-dh) * _norm.cdf(-dk)

    else:
        # High correlation: complementary formula
        if r < 0:
            k = -dk
            hk = -dh * dk
        else:
            k = dk
            hk = dh * dk

        ass1 = (1.0 - r) * (1.0 + r)
        a = np.sqrt(ass1)
        bs = (dh - k) ** 2
        c = (4.0 - hk) / 8.0
        d = (12.0 - hk) / 16.0

        if ass1 > 0:
            asr = -(bs / ass1 + hk) / 2.0
        else:
            asr = -200.0

        if asr > -100:
            bvn = (
                a
                * np.exp(asr)
                * (
                    1.0
                    - c * (bs - ass1) * (1.0 - d * bs / 5.0) / 3.0
                    + c * d * ass1 * ass1 / 5.0
                )
            )
        else:
            bvn = 0.0

        if -hk < 100:
            b = np.sqrt(bs)
            bvn -= (
                np.exp(-hk / 2.0)
                * np.sqrt(TWOPI)
                * _norm.cdf(-b / a)
                * b
                * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
            )

        a = a / 2.0

        # GL quadrature for remainder (always 20-point for high corr)
        w, x = _GL20_W, _GL20_X
        for i in range(10):
            for isign in (1, -1):
                xs = (a * (isign * x[i] + 1.0)) ** 2
                rs = np.sqrt(1.0 - xs)
                asr2 = -(bs / xs + hk) / 2.0
                if asr2 > -100:
                    bvn += a * w[i] * np.exp(asr2) * (
                        np.exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                        - (1.0 + c * xs * (1.0 + d * xs))
                    )

        bvn = -bvn / TWOPI

        if r > 0:
            bvn += _norm.cdf(-max(dh, k))
        else:
            bvn = -bvn
            if k > dh:
                bvn += _norm.cdf(k) - _norm.cdf(dh)

        return max(0.0, min(1.0, bvn))


def bivariate_normal_cdf(
    x1: float, x2: float, rho: float, *, xp=None
) -> float:
    """Bivariate standard normal CDF P(X1 <= x1, X2 <= x2) with correlation rho.

    Uses the Genz (2004) algorithm with Gauss-Legendre quadrature, achieving
    ~1e-15 absolute precision. This replaces scipy's general-purpose MVN CDF
    which only achieves ~1e-7 precision for the bivariate case.

    Parameters
    ----------
    x1, x2 : float
        Upper integration limits.
    rho : float
        Correlation coefficient in [-1, 1].
    xp : backend, optional

    Returns
    -------
    prob : float
        P(X1 <= x1, X2 <= x2)

    References
    ----------
    Genz, A. (2004). Numerical computation of rectangular bivariate and
    trivariate normal and t probabilities. Statistics and Computing, 14, 251-260.

    Drezner, Z. & Wesolowsky, G.O. (1990). On the computation of the
    bivariate normal integral. Journal of Statistical Computation and
    Simulation, 35, 101-107.
    """
    from scipy.stats import norm as _norm

    x1, x2, rho = float(x1), float(x2), float(rho)

    # Clamp rho to valid range
    rho = max(-1.0, min(1.0, rho))

    # Handle edge cases
    if abs(rho) < 1e-15:
        return float(_norm.cdf(x1) * _norm.cdf(x2))

    if rho > 1.0 - 1e-15:
        return float(_norm.cdf(min(x1, x2)))

    if rho < -1.0 + 1e-15:
        if x1 + x2 < 0:
            return 0.0
        return float(max(0.0, _norm.cdf(x1) + _norm.cdf(x2) - 1.0))

    # Genz (2004) BVND algorithm: computes P(X > -x1, Y > -x2 | rho)
    # which equals P(X < x1, Y < x2 | rho) by symmetry of standard normal.
    result = _bvnd(-x1, -x2, rho)
    return max(0.0, min(1.0, result))


def trivariate_normal_cdf(
    x1: float, x2: float, x3: float, sigma: NDArray, *, xp=None
) -> float:
    """Trivariate standard normal CDF P(X1<=x1, X2<=x2, X3<=x3).

    Parameters
    ----------
    x1, x2, x3 : float
        Upper integration limits.
    sigma : ndarray, shape (3, 3)
        Correlation matrix.
    xp : backend, optional

    Returns
    -------
    prob : float
    """
    from scipy.stats import multivariate_normal

    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.array([x1, x2, x3])
    try:
        result = multivariate_normal.cdf(x, mean=np.zeros(3), cov=sigma)
        return max(0.0, min(1.0, float(result)))
    except Exception:
        return 0.0


def quadrivariate_normal_cdf(
    x1: float, x2: float, x3: float, x4: float, sigma: NDArray, *, xp=None
) -> float:
    """Quadrivariate standard normal CDF P(X1<=x1, ..., X4<=x4).

    Parameters
    ----------
    x1, x2, x3, x4 : float
        Upper integration limits.
    sigma : ndarray, shape (4, 4)
        Correlation matrix.
    xp : backend, optional

    Returns
    -------
    prob : float
    """
    from scipy.stats import multivariate_normal

    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.array([x1, x2, x3, x4])
    try:
        result = multivariate_normal.cdf(x, mean=np.zeros(4), cov=sigma)
        return max(0.0, min(1.0, float(result)))
    except Exception:
        return 0.0
