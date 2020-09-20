import pathlib
from ctypes import CDLL, c_int, c_float, POINTER, byref, c_void_p
from typing import Tuple, List, Union

import numpy as np

from catmu.structures import (sImage2d, sPositions2d, sPSF)

__here__ = pathlib.Path(__file__).parent

MAKE_COMMAND = f'make -C {__here__ / "cuda_sources"} all'


class ConvolveLibraryMultiple:
    def __init__(self,
                 image_size: Tuple[int, int] = (64, 64),
                 image_pixel_size: Tuple[float, float] = (1.0, 1.0),
                 psf_pixel_size: Tuple[float, float] = (1.0, 1.0),
                 subpixel: int = 1,
                 block_size: int = 8,
                 device: int = 0,
                 debug: bool = False):
        """

        Parámetros
        ----------
        image_size: Tuple[int, int]
            Tamaño de la imagen de salida
        image_pixel_size: Tuple[float, float]
            Tamaño del pixel de la imagen
        psf_pixel_size: Tuple[float, float]
            Tamaño del pixel de la PSF.
        debug: bool
            Kernel compilado en modo DEBUG. Por defecto deshabilitado.

        """

        if debug is True:
            self._lib_name = pathlib.Path(__here__ / f'bin/libConvolveLUTd.so')
        else:
            self._lib_name = pathlib.Path(__here__ / f'bin/libConvolveLUT.so')

        if self._lib_name.exists() is False:
            import subprocess
            subprocess.run(MAKE_COMMAND, shell=True)
            if self._lib_name.exists() is False:
                raise FileNotFoundError(f'La biblioteca {self._lib_name} no pudo ser compilada.')

        self._lib = CDLL(self._lib_name)

        self._lib.lutConvolution2D.argtypes = [POINTER(sImage2d), POINTER(sPositions2d), POINTER(sPSF),
                                               c_int, c_int, c_int]
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

        self._subpixel = subpixel
        self._block_size = block_size
        self._device = device

    @property
    def kernel_name(self) -> str:
        return f'stable version'

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
    def positions(self) -> List[np.ndarray]:
        return self._positions

    @positions.setter
    def positions(self, new_positions: np.ndarray):
        if not isinstance(new_positions, np.ndarray):
            raise TypeError
        if new_positions.ndim == 3:
            self._positions = new_positions.astype(c_float, order='c', copy=True)
        elif new_positions.ndim == 2:
            self._positions = np.array([new_positions], dtype=c_float, order='c', copy=True)
        else:
            raise ValueError

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

        n = len(self._positions)
        self._image = np.zeros((n, self._image_size[0], self._image_size[1]), dtype=c_float, order='c')

        _image = sImage2d.create(self._image,
                                 pixel_width=self.image_pixel_size[1],
                                 pixel_height=self.image_pixel_size[0])

        _positions = sPositions2d.create(self._positions)

        _psf = sPSF.create(self._psf,
                           pixel_width=self._psf_pixel_size[1],
                           pixel_height=self._psf_pixel_size[0])

        r = self._lib.lutConvolution2D(_image, _positions, _psf, n, self._subpixel, self._block_size, self._device)
        if r != 0:
            if r == 100:
                raise RuntimeError('No se encontró ninguna GPU disponible en el sistema\n\n'
                                   'Verifique el estado del dispositivo y su driver ejecutando:\n'
                                   'nvidia-smi')
            else:
                raise RuntimeError(f'lutConvolution2D return error code: {r}')


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions
    import time

    convolution_size = (8, 8)
    image_pixel = (1.0, 1.0)
    psf_pixel = (1.0, 1.0)
    psf_size = (10, 10)
    n_sources = 60000
    sigma = 2.0

    pos = make_n_random_positions(n=100, n_sources=n_sources, convolution_size=(8, 8))
    psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

    convolution = ConvolveLibraryMultiple(image_size=convolution_size,
                                          image_pixel_size=image_pixel,
                                          psf_pixel_size=psf_pixel,
                                          subpixel=1,
                                          block_size=8,
                                          debug=False)

    convolution.positions = pos
    convolution.psf = psf

    t = time.time()
    convolution.launch()
    print(f'{(time.time() - t) * 1000}mS')
    # plt.imshow(convolution.image[5])
    # plt.plot(pos[5][:, 0], pos[5][:, 1], color='k', ls='', marker='.', markersize=1.0)
    # plt.show()
