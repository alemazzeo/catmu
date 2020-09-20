import numpy as np
from ctypes import Structure, c_int, c_void_p, c_float, POINTER
from typing import Tuple

class sImage2d(Structure):
    """ Estructura de Ctypes para la imagen convolucionada """
    _fields_ = [("width", c_int),
                ("height", c_int),
                ("pixel_width", c_float),
                ("pixel_height", c_float),
                ("data", c_void_p)]

    @classmethod
    def create(cls, image: np.ndarray,
               pixel_width: float = 1.0, pixel_height: float = 1.0):
        structure = cls()
        structure.width = image.shape[-1]
        structure.height = image.shape[-2]
        structure.pixel_width = c_float(pixel_width)
        structure.pixel_height = c_float(pixel_height)
        structure.data = image.ctypes.data_as(c_void_p)
        return structure

    def set_data(self, image: np.ndarray, pixel_width: float = 1.0, pixel_height: float = 1.0):
        self.width = image.shape[-1]
        self.height = image.shape[-2]
        self.pixel_width = c_float(pixel_width)
        self.pixel_height = c_float(pixel_height)
        self.data = image.ctypes.data_as(c_void_p)

    @classmethod
    def array(cls, n: int = 1):
        if isinstance(n, int):
            return (cls * n)()
        else:
            raise TypeError


class sPositions2d(Structure):
    """ Estructura de Ctypes para la lista de posiciones """
    _fields_ = [("n", c_int),
                ("data", c_void_p)]

    @classmethod
    def create(cls, positions: np.ndarray):
        structure = cls()
        structure.n = positions.shape[-2]
        structure.data = positions.ctypes.data_as(c_void_p)
        return structure

    def set_data(self, positions: np.ndarray):
        if positions.ndim != 2:
            raise ValueError(f'positions.ndim = {positions.ndim} != 2\n')
        self.n = positions.shape[-2]
        self.data = positions.ctypes.data_as(c_void_p)

    @classmethod
    def array(cls, n: int = 1):
        if isinstance(n, int):
            return (cls * n)()
        else:
            raise TypeError


class sPSF(Structure):
    """ Estructura de Ctypes para los datos de una PSF tipo LUT """
    _fields_ = [("width", c_int),
                ("height", c_int),
                ("pixel_width", c_float),
                ("pixel_height", c_float),
                ("data", c_void_p),]

    @classmethod
    def create(cls, psf_data: np.ndarray,
               pixel_width: float = 1.0, pixel_height: float = 1.0):
        structure = cls()
        structure.width = psf_data.shape[-1]
        structure.height = psf_data.shape[-2]
        structure.pixel_width = c_float(pixel_width)
        structure.pixel_height = c_float(pixel_height)
        structure.data = psf_data.ctypes.data_as(c_void_p)

        return structure


# Punteros a cada tipo de estructura
pImage2d = POINTER(sImage2d)
pPositions2d = POINTER(sPositions2d)
pPSF = POINTER(sPSF)