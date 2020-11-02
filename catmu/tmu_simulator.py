import numpy as np


def tex2d(x: float, y: float, psf: np.ndarray):
    """

    tex(x,y) = (1−α)(1−β)T[i,j] + α(1−β)T[i+1,j] + (1−α)βT[i,j+1] + αβT[i+1,j+1]

    where:

    i=floor(x-0.5), α=frac(x-0.5)
    j=floor(y-0.5), β=frac(y-0.5)

    T = psf

    """
    psf = np.pad(psf, pad_width=1, mode='constant', constant_values=0)
    m, n = psf.shape

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    x += 1.0
    y += 1.0

    x[x < 0.5] = 0.5
    x[x > n] = n
    y[y < 0.5] = 0.5
    y[y > m] = m

    a, i = np.modf(y - 0.5)
    b, j = np.modf(x - 0.5)

    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    ip = np.clip(i+1, 0, m-1)
    jp = np.clip(j+1, 0, n-1)

    t1 = (1 - a) * (1 - b) * psf[i, j]
    t2 = a * (1 - b) * psf[ip, j]
    t3 = (1 - a) * b * psf[i, jp]
    t4 = a * b * psf[ip, jp]

    result = t1 + t2 + t3 + t4
    return result
