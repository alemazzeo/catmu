from ctypes import Structure
from ctypes import c_int, c_float, c_void_p, c_size_t

import numpy as np


class sImage2d(Structure):
    """ Estructura de Ctypes para la imagen convolucionada """
    _fields_ = [("N", c_int),
                ("width", c_int),
                ("height", c_int),
                ("pixel_width", c_float),
                ("pixel_height", c_float),
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    @classmethod
    def create(cls, image: np.ndarray,
               pixel_width: float = 1.0, pixel_height: float = 1.0):
        structure = cls()
        if image.ndim == 3:
            structure.N = c_int(image.shape[0])
        else:
            structure.N = c_int(1)
        structure.width = image.shape[-1]
        structure.height = image.shape[-2]
        structure.pixel_width = c_float(pixel_width)
        structure.pixel_height = c_float(pixel_height)
        structure.data = image.ctypes.data_as(c_void_p)
        structure.allocated_size = c_size_t(0)
        return structure

    def set_data(self, image: np.ndarray, pixel_width: float = 1.0, pixel_height: float = 1.0):
        if image.ndim == 3:
            self.N = c_int(image.shape[0])
        else:
            self.N = c_int(1)

        self.width = c_int(image.shape[-1])
        self.height = c_int(image.shape[-2])
        self.pixel_width = c_float(pixel_width)
        self.pixel_height = c_float(pixel_height)
        self.data = image.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)

    @classmethod
    def array(cls, n: int = 1):
        if isinstance(n, int):
            return (cls * n)()
        else:
            raise TypeError


class sPositions2d(Structure):
    """ Estructura de Ctypes para la lista de posiciones """
    _fields_ = [("N", c_int),
                ("n", c_int),
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    @classmethod
    def create(cls, positions: np.ndarray):
        structure = cls()
        if positions.ndim == 3:
            structure.N = positions.shape[0]
        else:
            structure.N = positions.shape[0]
        structure.n = positions.shape[-2]
        structure.data = positions.ctypes.data_as(c_void_p)
        structure.allocated_size = c_size_t(0)
        return structure

    def set_data(self, positions: np.ndarray):
        if positions.ndim == 3:
            self.N = positions.shape[0]
        else:
            self.N = positions.shape[0]
        self.n = positions.shape[-2]
        self.data = positions.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)

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
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    @classmethod
    def create(cls, psf_data: np.ndarray,
               pixel_width: float = 1.0, pixel_height: float = 1.0):
        structure = cls()
        structure.width = psf_data.shape[-1]
        structure.height = psf_data.shape[-2]
        structure.pixel_width = c_float(pixel_width)
        structure.pixel_height = c_float(pixel_height)
        structure.data = psf_data.ctypes.data_as(c_void_p)
        structure.allocated_size = c_size_t(0)

        return structure


class sConfig(Structure):
    """ Estructura de Ctypes para los datos de una PSF tipo LUT """
    _fields_ = [("device", c_int),
                ("sub_pixel", c_int),
                ("block_size", c_int),
                ("n_streams", c_int)]

    @classmethod
    def create(cls, device: int, sub_pixel: int, block_size: int, n_streams: int):
        structure = cls()
        structure.device = c_int(device)
        structure.sub_pixel = c_int(sub_pixel)
        structure.block_size = c_int(block_size)
        structure.n_streams = c_int(n_streams)

        return structure
