from ctypes import Structure
from ctypes import c_int, c_float, c_void_p, c_size_t, c_double, c_char
from typing import Tuple, Union, Iterable

import numpy as np

Size2D = Tuple[int, int]
Size3D = Tuple[int, int, int]
PixelSize2D = Tuple[float, float]
PixelSize3D = Tuple[float, float, float]
ValidSizes = Union[Size2D, Size3D]
ValidPixelSizes = Union[PixelSize2D, PixelSize3D]


class Image(Structure):
    """ Estructura de Ctypes para la imagen convolucionada """
    _fields_ = [("N", c_int),
                ("width", c_int),
                ("height", c_int),
                ("depth", c_int),
                ("pixel_width", c_float),
                ("pixel_height", c_float),
                ("pixel_depth", c_float),
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    def __init__(self, image: np.ndarray = None, pixel_size: ValidPixelSizes = None):
        super().__init__()
        self.N = 0
        self.width, self.height, self.depth = (0, 0, 0)
        self.pixel_width, self.pixel_height, self.pixel_depth = (0, 0, 0)
        self.data = 0
        self.allocated_size = 0

        if image is not None:
            self.set_data(image=image, pixel_size=pixel_size)

    def set_data(self, image: np.ndarray,
                 pixel_size: ValidPixelSizes = None):

        if pixel_size is None:
            pixel_size = (1.0, 1.0, 1.0)
        else:
            if len(pixel_size) == 2:
                pixel_size = (1.0, pixel_size[0], pixel_size[1])

        self.width = c_int(image.shape[-1])
        self.height = c_int(image.shape[-2])
        if image.ndim == 3:
            self.depth = c_int(1)
            self.N = image.shape[-3]
        elif image.ndim == 4:
            self.depth = image.shape[-3]
            self.N = image.shape[-4]

        self.pixel_width = c_float(pixel_size[-1])
        self.pixel_height = c_float(pixel_size[-2])
        self.pixel_depth = c_float(pixel_size[-3])
        self.data = image.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)


class Positions(Structure):
    """ Estructura de Ctypes para la lista de posiciones """
    _fields_ = [("N", c_int),
                ("n", c_int),
                ("dim", c_int),
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    def __init__(self, positions: np.ndarray = None):
        super().__init__()
        self.N = 0
        self.n = 0
        self.dim = 0
        self.data = 0
        self.allocated_size = 0
        if positions is not None:
            self.set_data(positions=positions)

    def set_data(self, positions: np.ndarray):
        if positions.ndim == 2:  # pragma: no cover
            self.N = 1
        else:
            self.N = positions.shape[-3]
        self.n = positions.shape[-2]
        self.dim = positions.shape[-1]
        self.data = positions.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)


class LutPSF(Structure):
    """ Estructura de Ctypes para los datos de una PSF tipo LUT """
    _fields_ = [("width", c_int),
                ("height", c_int),
                ("depth", c_int),
                ("dim", c_int),
                ("pixel_width", c_float),
                ("pixel_height", c_float),
                ("pixel_depth", c_float),
                ("data", c_void_p),
                ("allocated_size", c_size_t)]

    def __init__(self, psf_data: np.ndarray, pixel_size: ValidPixelSizes = None):
        super().__init__()

        if pixel_size is None:
            pixel_size = (1.0, 1.0, 1.0)

        if len(pixel_size) == 2:
            pixel_size = (1.0, pixel_size[0], pixel_size[1])

        self.dim = c_int(psf_data.ndim)
        self.width = c_int(psf_data.shape[-1])
        self.height = c_int(psf_data.shape[-2])
        if psf_data.ndim == 3:
            self.depth = c_int(psf_data.shape[-3])
        else:
            self.depth = c_int(1)
        self.pixel_width = c_float(pixel_size[-1])
        self.pixel_height = c_float(pixel_size[-2])
        self.pixel_depth = c_float(pixel_size[-3])
        self.data = psf_data.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)


class ExpressionPSF(Structure):
    """ Estructura de Ctypes para los datos de una PSF tipo Expresión """
    _fields_ = [("id_function", c_int),
                ("n_params", c_int),
                ("params", c_void_p),
                ("allocated_size", c_size_t)]

    def __init__(self, id_function, params: Iterable):
        super().__init__()

        self.params_array = np.array(params, dtype=c_double)
        self.id_function = c_int(id_function)
        self.n_params = c_int(self.params_array.size)
        self.params = self.params_array.ctypes.data_as(c_void_p)
        self.allocated_size = c_size_t(0)


class DevConfig(Structure):
    """ Estructura de Ctypes para los datos de una PSF tipo LUT """
    _fields_ = [("device", c_int),
                ("block_size", c_int),
                ("n_streams", c_int)]

    def __init__(self, device: int, block_size: int, n_streams: int):
        super().__init__()
        self.device = c_int(device)
        self.block_size = c_int(block_size)
        self.n_streams = c_int(n_streams)


class DeviceProperties(Structure):
    _fields_ = [
        ("name", c_char * 256),
        ("multiProcessorCount", c_int),
        ("totalGlobalMem", c_size_t),
        ("sharedMemPerBlock", c_size_t),
        ("major", c_int),
        ("minor", c_int),
        ("regsPerBlock", c_int),
        ("warpSize", c_int),
        ("memPitch", c_size_t),
        ("maxThreadsPerBlock", c_int),
        ("maxThreadsDim", c_int * 3),
        ("maxGridSize", c_int * 3),
        ("clockRate", c_int),
        ("totalConstMem", c_size_t),
        ("textureAlignment", c_size_t),
        ("deviceOverlap", c_int),
        ("kernelExecTimeoutEnabled", c_int)]

    def __str__(self):
        return (f"Name:                             {self.name}\n"
                f"Number of multiprocessors:        {self.multiProcessorCount}\n"
                f"Total global memory:              {self.totalGlobalMem}\n"
                f"Total shared memory per block:    {self.sharedMemPerBlock}\n"
                f"Major revision number:            {self.major}\n"
                f"Minor revision number:            {self.minor}\n"
                f"Total registers per block:        {self.regsPerBlock}\n"
                f"Warp size:                        {self.warpSize}\n"
                f"Maximum memory pitch:             {self.memPitch}\n"
                f"Maximum threads per block:        {self.maxThreadsPerBlock}\n"
                f"Maximum dimension of block:       ("
                f"{self.maxThreadsDim[0]}, "
                f"{self.maxThreadsDim[1]}, "
                f"{self.maxThreadsDim[2]}) \n"
                f"Maximum dimension of grid:        ("
                f"{self.maxGridSize[0]}, "
                f"{self.maxGridSize[1]}, "
                f"{self.maxGridSize[2]}) \n"
                f"Clock rate (KHz):                 {self.clockRate}\n"
                f"Total constant memory:            {self.totalConstMem}\n"
                f"Texture alignment:                {self.textureAlignment}\n"
                f"Concurrent copy and execution:    {self.deviceOverlap}\n"
                f"Kernel execution timeout:         {self.kernelExecTimeoutEnabled}\n")