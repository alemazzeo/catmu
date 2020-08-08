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

    # mask_x = np.logical_or(x < 0, x > 1.0)
    # mask_y = np.logical_or(y < 0, y > 1.0)
    # mask = np.logical_or(mask_x, mask_y)

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
    # result[mask] = 0.0
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # psf_lut = np.arange(9).reshape((3, 3))
    # n = 100
    # x, y = 2 * np.mgrid[0:n + 1, 0:n + 1] / n - 0.5
    # print(x, y)
    # texture_applied = tex2d(x=x, y=y, psf=psf_lut)
    # print(texture_applied)
    # _, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(psf_lut, extent=(0, 1, 0, 1), vmin=np.min(psf_lut), vmax=np.max(psf_lut))
    # ax2.imshow(texture_applied, vmin=np.min(psf_lut), vmax=np.max(psf_lut),
    #            extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
    # plt.show()

    n = 100
    pad = 3

    psf = np.pad(np.ones((11, 11)), pad_width=pad, mode='constant', constant_values=0)

    y, x = np.mgrid[0:n * (psf.shape[0]), 0:n * (psf.shape[1])] / n

    plt.imshow(tex2d(x, y, psf), extent=(x.min(), x.max(), y.min(), y.max()))

    for i in range(psf.shape[0]):
        if i == pad or i == psf.shape[0] - pad:
            plt.axhline(i, color='red')
        else:
            plt.axhline(i)
    for i in range(psf.shape[1]):
        if i == pad or i == psf.shape[1] - pad:
            plt.axvline(i, color='red')
        else:
            plt.axvline(i)

    plt.show()





