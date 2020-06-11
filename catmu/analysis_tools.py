import numpy as np
from typing import Tuple


def make_gaussian_psf_lut(psf_size: Tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    x_psf, y_psf = np.mgrid[0:psf_size[0], 0:psf_size[1]]
    x_psf -= psf_size[0] // 2
    y_psf -= psf_size[1] // 2
    return np.exp(-(x_psf ** 2 + y_psf ** 2) / sigma ** 2 / 2)


def make_random_positions(n_sources: int = 6000, convolution_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(np.random.rand(n_sources, 2) * convolution_size)


def convolve_guassian_psf(positions: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    exact_image = np.zeros(convolution_size[::-1])
    x, y = np.mgrid[0:convolution_size[0], 0:convolution_size[1]]

    for point in positions:
        px, py = point
        displaced_psf = np.exp(-((x - px) ** 2 + (y - py) ** 2) / sigma ** 2 / 2)
        exact_image += displaced_psf

    return exact_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from catmu.wrapper import ConvolveLibrary

    convolution_size = (64, 64)
    image_pixel = (1.0, 1.0)
    psf_pixel = (1.0, 1.0)
    psf_size = (25, 25)
    n_sources = 6000
    sigma = 2.0

    pos = make_random_positions(n_sources=n_sources, convolution_size=convolution_size)
    psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=2.0)

    convolution = ConvolveLibrary(kernel='d00',
                                  image_size=convolution_size,
                                  image_pixel_size=image_pixel,
                                  psf_pixel_size=psf_pixel,
                                  debug=False)

    convolution.positions = pos
    convolution.psf = psf

    convolution.launch()

    plt.plot(convolution.image.flatten())
    plt.show()