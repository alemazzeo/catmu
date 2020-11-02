# -*- coding: utf-8 -*-
"""
    Prueba básica de funcionamiento optimizada para ver los resultados en el NVVP.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from catmu import ConvolutionManager
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 2
image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0, 1.0)
psf_size = (10, 10)
n_sources = 6400
sigma = 2.0

pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

convolution = ConvolutionManager(device=0,
                                 block_size=8,
                                 n_streams=10,
                                 sub_pixel=1,
                                 debug=True)

convolution.setting(psf=psf,
                    image_size=image_size,
                    image_pixel_size=image_pixel_size,
                    psf_pixel_size=psf_pixel_size)

# Diez convoluciones utilizando el mismo contexto
for i in range(10):
    results = convolution.sync_convolve(positions=pos)
    print(f'{convolution.loop_counter} -> {convolution.last_elapsed_time * 1000}mS')

# Diez convoluciones reiniciando el contexto (como pasaba antes)
for i in range(10):
    convolution.setting(psf=psf,
                        image_size=image_size,
                        image_pixel_size=image_pixel_size,
                        psf_pixel_size=psf_pixel_size)
    results = convolution.sync_convolve(positions=pos)
    print(f'{convolution.loop_counter} -> {convolution.last_elapsed_time * 1000}mS')
