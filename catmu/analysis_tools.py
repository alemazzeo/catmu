from typing import Tuple, NamedTuple
import numpy as np


class MeasuredPSF(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    amplitude: float
    sigma2: float
    offset: float
    rmse: float
    gaussian_fit: callable


def make_gaussian_psf_lut(psf_size: Tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    x_psf, y_psf = np.mgrid[0:psf_size[0], 0:psf_size[1]]
    x_psf -= psf_size[0] // 2
    y_psf -= psf_size[1] // 2
    return np.exp(-(x_psf ** 2 + y_psf ** 2) / sigma ** 2 / 2)


def make_random_positions(n_sources: int = 6000,
                          convolution_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(np.random.rand(n_sources, 2) * convolution_size)


def make_n_random_positions(n: int = 100,
                            n_sources: int = 6000,
                            convolution_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(np.random.rand(n, n_sources, 2) * convolution_size)


def convolve_guassian_psf(positions: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    exact_image = np.zeros(convolution_size[::-1])
    x, y = np.mgrid[0:convolution_size[0], 0:convolution_size[1]]

    for point in positions:
        px, py = point
        displaced_psf = np.exp(-((x - px) ** 2 + (y - py) ** 2) / sigma ** 2 / 2)
        exact_image += displaced_psf

    return exact_image


def load_measured_psf() -> MeasuredPSF:
    data = np.load(__file__.replace('analysis_tools.py', 'measured_psf.npz'))

    def gaussian_fit(x, y):
        return data['amplitude'] * np.exp(-(x ** 2 + y ** 2) / data['sigma2'] / 2) + data['offset']

    return MeasuredPSF(x=data['x'],
                       y=data['y'],
                       z=data['z'],
                       amplitude=data['amplitude'],
                       sigma2=data['sigma2'],
                       offset=data['offset'],
                       rmse=data['rmse'],
                       gaussian_fit=gaussian_fit)
