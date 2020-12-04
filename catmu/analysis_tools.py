from catmu import ValidSizes
import numpy as np


def make_gaussian_psf_lut(psf_size: ValidSizes,
                          sigma: float = 2.0) -> np.ndarray:
    if len(psf_size) == 2:
        x_psf, y_psf = np.mgrid[0:psf_size[0], 0:psf_size[1]]
        x_psf -= psf_size[0] // 2
        y_psf -= psf_size[1] // 2
        return np.exp(-(x_psf ** 2 + y_psf ** 2) / sigma ** 2 / 2)
    elif len(psf_size) == 3:
        x_psf, y_psf, z_psf = np.mgrid[0:psf_size[0], 0:psf_size[1], 0:psf_size[2]]
        x_psf -= psf_size[0] // 2
        y_psf -= psf_size[1] // 2
        z_psf -= psf_size[2] // 2
        return np.exp(-(x_psf ** 2 + y_psf ** 2 + z_psf ** 2) / sigma ** 2 / 2)
    else:   # pragma: no cover
        raise ValueError


def make_n_random_positions(n: int = 100,
                            n_sources: int = 6000,
                            n_dim: int = 2,
                            convolution_size: ValidSizes = (64, 64)) -> np.ndarray:
    return np.random.rand(n, n_sources, n_dim) * (np.array(convolution_size)[::-1] - 1)
