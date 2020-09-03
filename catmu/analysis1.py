import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from catmu.tmu_simulator import tex2d
from catmu.analysis_tools import make_gaussian_psf_lut, make_random_positions
from catmu.wrapper import ConvolveLibrary


def displaced_psf_tmu(psf: np.ndarray,
                      image_size: Tuple[int, int] = (64, 64),
                      dx: float = 0.0,
                      dy: float = 0.0) -> np.ndarray:
    # Creamos el wrapper para el kernel d03
    convolution = ConvolveLibrary(kernel='d03',
                                  image_size=image_size,
                                  debug=True)

    # Indicamos la posición de desplazamiento
    convolution.positions = np.asarray([[dx, dy]])
    convolution.psf = psf

    convolution.launch()

    return convolution.image


def displaced_psf_cpu(psf: np.ndarray,
                      image_size: Tuple[int, int] = (64, 64),
                      dx: float = 0.0,
                      dy: float = 0.0) -> np.ndarray:
    # Calculamos las posiciones desplazadas
    y, x = np.mgrid[0:image_size[0], 0:image_size[1]]
    x = (x + dx) / (image_size[1] - 1) * (psf.shape[1] - 1) + 0.5
    y = (y + dy) / (image_size[0] - 1) * (psf.shape[0] - 1) + 0.5

    return tex2d(x=x, y=y, psf=psf)


psf = make_gaussian_psf_lut(psf_size=(11, 11), sigma=2.0)

result_tmu = displaced_psf_tmu(psf, dx=0.0, dy=0.0)
result_cpu = displaced_psf_cpu(psf, dx=0.0, dy=0.0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6), sharex='all', sharey='all')
ax1.set_title('Resultado TMU', fontsize=15)
ax2.set_title('Resultado CPU', fontsize=15)
ax3.set_title('Error relativo (%)', fontsize=15)
m1 = ax1.imshow(result_tmu, origin='lower', cmap='hot')
m2 = ax2.imshow(result_cpu, origin='lower', cmap='hot')
m3 = ax3.imshow(np.abs(result_cpu-result_tmu) / np.max(result_cpu) * 100, cmap='hot')
plt.colorbar(m1, ax=ax1, orientation='horizontal')
plt.colorbar(m2, ax=ax2, orientation='horizontal')
plt.colorbar(m3, ax=ax3, orientation='horizontal')

fig.tight_layout()
plt.show()

plt.plot(result_tmu[30, :], 'bo')
plt.plot(result_cpu[30, :], 'y*')

plt.show()
