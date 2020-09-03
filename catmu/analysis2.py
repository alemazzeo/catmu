from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from catmu.analysis_tools import load_measured_psf
from catmu.wrapper import ConvolveLibrary

measured_psf = load_measured_psf()

psf = measured_psf.z
# psf = measured_psf.gaussian_fit(measured_psf.x, measured_psf.y)
gaussian_fit = measured_psf.gaussian_fit


def lut_convolution(image_size: Tuple[int, int] = (64, 64),
                    dx: float = 0.0,
                    dy: float = 0.0,
                    image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                    psf_pixel_size: Tuple[float, float] = (1.0, 1.0),
                    kernel: str = "v0_2") -> np.ndarray:
    # Creamos el wrapper para el kernel
    convolution = ConvolveLibrary(kernel=kernel,
                                  image_size=image_size,
                                  image_pixel_size=image_pixel_size,
                                  psf_pixel_size=psf_pixel_size,
                                  debug=False)

    # Indicamos la posición de desplazamiento
    convolution.positions = np.asarray([[image_size[1] / 2 - dx,
                                         image_size[0] / 2 - dy]])
    convolution.psf = psf

    convolution.launch()

    return convolution.image


def expression_convolution(image_size: Tuple[int, int] = (64, 64),
                           dx: float = 0.0,
                           dy: float = 0.0,
                           image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                           psf_pixel_size: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    rx = image_pixel_size[0] / psf_pixel_size[0]
    ry = image_pixel_size[1] / psf_pixel_size[1]

    y, x = np.mgrid[0:image_size[0], 0:image_size[1]]

    return gaussian_fit((x - image_size[1] / 2 + dx) * rx, (y - image_size[0] / 2 + dy) * ry)


def comparation(image_size: Tuple[int, int] = (64, 64), rx: float = 1.0, ry: float = 1.0,
                dx: float = 0.0, dy: float = 0.0, plot: bool = False, save: bool = False):
    result_lut = lut_convolution(image_size=image_size, psf_pixel_size=(rx, ry), dx=dx, dy=dy)
    result_expression = expression_convolution(image_size=image_size, psf_pixel_size=(rx, ry), dx=dx, dy=dy)

    if plot is True:
        fig: plt.Figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6), sharex='all', sharey='all')
        ax1.set_title('Resultado por LUT', fontsize=15)
        ax2.set_title('Resultado por Expresión', fontsize=15)
        ax3.set_title('Error relativo (%)', fontsize=15)
        m1 = ax1.imshow(result_lut, origin='lower', cmap='hot')
        m2 = ax2.imshow(result_expression, origin='lower', cmap='hot')
        m3 = ax3.imshow(np.abs(result_expression - result_lut) / np.max(result_expression) * 100,
                        cmap='hot', vmin=0, vmax=10)
        if save is False:
            plt.colorbar(m1, ax=ax1, orientation='horizontal')
            plt.colorbar(m2, ax=ax2, orientation='horizontal')
            plt.colorbar(m3, ax=ax3, orientation='horizontal')
        # fig.suptitle(f"Imagen de {image_size[1]}x{image_size[1]} - "
        #              f"PSF de {psf.shape[1]}x{psf.shape[0]}\n"
        #              f"Pixel imagen = 1.00\n"
        #              f"Pixel PSF    = {rx:0.2f}\n"
        #              f"(dx, dy) = ({dx:0.2f}, {dy:0.2f})\n")

        fig.tight_layout()
        if save is True:
            ax1.set_xticks([])
            ax1.set_xticks([], minor=True)
            ax2.set_xticks([])
            ax2.set_xticks([], minor=True)
            ax3.set_xticks([])
            ax3.set_xticks([], minor=True)
            print(f"gif_rx_{int(rx*100):05d}.png")
            fig.savefig(f"gif_rx_{int(rx*100):05d}.png")
        else:
            plt.show()

    rmse = np.sqrt(np.mean((result_expression.flatten() - result_lut.flatten()) ** 2))
    norm_max_error = np.max(np.abs(result_expression - result_lut)) / np.max(np.abs(result_expression)) * 100
    return norm_max_error


image_size = (64, 64)
psf_size = psf.shape
max_factor = image_size[0] / psf_size[0]

# comparation(rx=1.0, ry=1.0, dx=0.0, dy=0.0, plot=True)
#
# comparation(rx=1.0, ry=1.0, dx=0.0, dy=0.0, plot=True)
# comparation(rx=2.0, ry=2.0, dx=0.0, dy=0.0, plot=True)
# comparation(rx=4.0, ry=4.0, dx=0.0, dy=0.0, plot=True)
#
# comparation(rx=1.0, ry=1.0, dx=0.5, dy=0.0, plot=True)
# comparation(rx=2.0, ry=2.0, dx=0.5, dy=0.0, plot=True)
# comparation(rx=4.0, ry=4.0, dx=0.5, dy=0.0, plot=True)
#
# exit()


error_base = comparation(rx=1.0, ry=1.0, dx=0.0, dy=0.0, plot=True)

errors = []
factor = np.arange(100) * 0.1 + 0.1

for r in factor:
    errors.append(comparation(image_size=image_size, rx=r, ry=r))

errors = np.asarray(errors)

fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex='all', sharey='all')
ax.plot(factor, errors, marker='o', ls='')
ax.set_xlabel("Factor", fontsize=15)
ax.set_ylabel("Error máximo (%)", fontsize=15)
fig.tight_layout()
plt.show()

r = [0.5, 1.0, 2]
n = 50
rmse_xy = np.zeros((len(r), n + 1, n + 1))

for k in range(len(r)):
    for i in range(n + 1):
        for j in range(n + 1):
            rmse_xy[k, j, i] = comparation(image_size=image_size, rx=r[k], ry=r[k], dx=i / (n / r[k]),
                                           dy=j / (n / r[k]))

fig, axs = plt.subplots(1, len(r), figsize=(10, 6), sharex='all', sharey='all')
for k, ax in enumerate(axs):
    ax.set_title(f'R={r[k]}', fontsize=15)
    m = ax.imshow(rmse_xy[k], origin='lower', extent=(0, 1, 0, 1), cmap='hot')
    plt.colorbar(m, ax=ax, orientation='horizontal')

fig.tight_layout()
plt.show()
