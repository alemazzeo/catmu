import numpy as np


def tex2d(x: float, y: float, psf: np.ndarray):
    """

    tex(x,y) = (1−α)(1−β)T[i,j] + α(1−β)T[i+1,j] + (1−α)βT[i,j+1] + αβT[i+1,j+1]

    where:

    i=floor(x-0.5), α=frac(x-0.5)
    j=floor(y-0.5), β=frac(y-0.5)

    T = psf

    """
    x = np.array(x)
    y = np.array(y)

    mask_x = np.logical_or(x < 0, x > 1.0)
    mask_y = np.logical_or(y < 0, y > 1.0)
    mask = np.logical_or(mask_x, mask_y)

    x[x < 0] = 0.0
    x[x > 1.0] = 1.0
    y[y < 0] = 0.0
    y[y > 1.0] = 1.0

    x = x * (psf.shape[1] - 1) + 0.5
    y = y * (psf.shape[0] - 1) + 0.5

    a, i = np.modf(x - 0.5)
    b, j = np.modf(y - 0.5)

    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)

    t1 = (1 - a) * (1 - b) * psf[i, j]
    t2 = a * (1 - b) * psf[np.minimum(psf.shape[0] - 1, i + 1), j]
    t3 = (1 - a) * b * psf[i, np.minimum(psf.shape[1] - 1, j + 1)]
    t4 = a * b * psf[np.minimum(psf.shape[1] - 1, i + 1), np.minimum(psf.shape[0] - 1, j + 1)]

    result = t1 + t2 + t3 + t4
    result[mask] = 0.0
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    psf_lut = np.arange(9).reshape((3, 3))
    n = 100
    x, y = 2 * np.mgrid[0:n + 1, 0:n + 1] / n - 0.5
    print(x, y)
    texture_applied = tex2d(x=x, y=y, psf=psf_lut)
    print(texture_applied)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(psf_lut, extent=(0, 1, 0, 1), vmin=np.min(psf_lut), vmax=np.max(psf_lut))
    ax2.imshow(texture_applied, vmin=np.min(psf_lut), vmax=np.max(psf_lut),
               extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    plt.show()
