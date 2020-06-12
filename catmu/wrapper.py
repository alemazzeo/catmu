import pathlib
from ctypes import CDLL, c_int, c_float, POINTER
from typing import Tuple

import numpy as np

from catmu.structures import (sImage2d, sPositions2d, sPSF)

__here__ = pathlib.Path(__file__).parent

MAKE_COMMAND = f'make -C {__here__ / "cuda_sources"} all'


class ConvolveLibrary:
    def __init__(self, kernel: str = 'v0_1',
                 image_size: Tuple[int, int] = (64, 64),
                 image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                 psf_pixel_size: Tuple[float, float] = (1.0, 1.0),
                 debug: bool = False):
        """

        Parámetros
        ----------
        kernel: str
            Identificación del kernel seleccionado
        image_size: Tuple[int, int]
            Tamaño de la imagen de salida
        image_pixel_size: Tuple[float, float]
            Tamaño del pixel de la imagen
        psf_pixel_size: Tuple[float, float]
            Tamaño del pixel de la PSF.
        debug: bool
            Kernel compilado en modo DEBUG. Por defecto deshabilitado.

        """

        self._kernel = pathlib.Path(__here__ / f'./cuda_sources/kernel_{kernel}.cu')

        if debug is True:
            self._lib_name = pathlib.Path(__here__ / f'./bin/libConvolveLUT_D_{kernel}.so')
        else:
            self._lib_name = pathlib.Path(__here__ / f'./bin/libConvolveLUT_{kernel}.so')

        if self._kernel.exists() is False:
            raise FileNotFoundError(f'No se encontró el kernel {self._kernel}')
        if self._lib_name.exists() is False:
            import subprocess
            subprocess.run(MAKE_COMMAND, shell=True)
            if self._lib_name.exists() is False:
                raise FileNotFoundError(f'La biblioteca {self._lib_name} no pudo ser compilada.')

        self._lib = CDLL(self._lib_name)

        self._lib.lutConvolution2D.argtypes = [POINTER(sImage2d), POINTER(sPositions2d), POINTER(sPSF), c_int]
        self._lib.lutConvolution2D.restype = c_int

        self._image_size = None
        self._image_pixel_size = None
        self._psf_pixel_size = None

        self._image = None
        self._positions = None
        self._psf = None

        self.image_size = image_size
        self.image_pixel_size = image_pixel_size
        self.psf_pixel_size = psf_pixel_size

    @property
    def kernel_name(self) -> str:
        return f'kernel_{self._kernel}.cu'

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @image_size.setter
    def image_size(self, new_size: Tuple[int, int]):
        if not isinstance(new_size, Tuple):
            raise TypeError
        if self._image_size != new_size:
            self._image_size = new_size
            self._image = np.zeros(self._image_size, dtype=c_float, order='c')

    @property
    def image_pixel_size(self) -> Tuple[float, float]:
        return self._image_pixel_size

    @image_pixel_size.setter
    def image_pixel_size(self, new_size: Tuple[float, float]):
        if not isinstance(new_size, Tuple):
            raise TypeError
        if self._image_pixel_size != new_size:
            self._image_pixel_size = new_size

    @property
    def psf_pixel_size(self) -> Tuple[float, float]:
        return self._image_size

    @psf_pixel_size.setter
    def psf_pixel_size(self, new_size: Tuple[float, float]):
        if not isinstance(new_size, Tuple):
            raise TypeError
        if self._psf_pixel_size != new_size:
            self._psf_pixel_size = new_size

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @positions.setter
    def positions(self, new_positions):
        if not isinstance(new_positions, np.ndarray):
            raise TypeError
        if np.array_equal(self._positions, new_positions) is False:
            self._positions = new_positions.astype(c_float, order='c', copy=True)

    @property
    def psf(self) -> np.ndarray:
        return self._psf

    @psf.setter
    def psf(self, new_psf):
        if not isinstance(new_psf, np.ndarray):
            raise TypeError
        if np.array_equal(self._psf, new_psf) is False:
            self._psf = new_psf.astype(c_float, order='c', copy=True)

    def launch(self):
        _image = sImage2d.set_data(self._image,
                                   pixel_width=self.image_pixel_size[1],
                                   pixel_height=self.image_pixel_size[0])
        _psf = sPSF.set_data(self._psf,
                             pixel_width=self._psf_pixel_size[1],
                             pixel_height=self._psf_pixel_size[0])
        _positions = sPositions2d.set_data(self._positions)

        r = self._lib.lutConvolution2D(_image, _positions, _psf, 0)
        if r != 0:
            if r == 100:
                raise RuntimeError('No se encontró ninguna GPU disponible en el sistema\n\n'
                                   'Verifique el estado del dispositivo y su driver ejecutando:\n'
                                   'nvidia-smi')
            else:
                raise RuntimeError(f'lutConvolution2D return error code: {r}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from catmu.analysis_tools import make_gaussian_psf_lut, make_random_positions
    convolution_size = (64, 64)
    image_pixel = (1.0, 1.0)
    psf_pixel = (1.0, 1.0)
    psf_size = (25, 25)
    n_sources = 6000
    sigma = 2.0

    pos = make_random_positions(n_sources=n_sources, convolution_size=convolution_size)
    psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

    convolution = ConvolveLibrary(kernel='v0_1',
                                  image_size=convolution_size,
                                  image_pixel_size=image_pixel,
                                  psf_pixel_size=psf_pixel,
                                  debug=False)

    convolution.positions = pos
    convolution.psf = psf

    convolution.launch()

    plt.imshow(convolution.image)
    plt.plot(pos[:, 0], pos[:, 1], color='k', ls='', marker='.', markersize=1.0)
    plt.show()

