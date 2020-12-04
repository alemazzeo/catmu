# -*- coding: utf-8 -*-
"""
    Prueba básica de funcionamiento.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import matplotlib.pyplot as plt
import numpy as np

from catmu import ConvolutionManagerGPU
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 100
image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0, 1.0)
psf_size = (10, 10)
n_sources = 6400
sigma1 = 2.0
sigma2 = 1.0

pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf1 = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma1)
psf2 = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma2)

convolution1 = ConvolutionManagerGPU(device=0,
                                     block_size=8,
                                     n_streams=10,
                                     debug=True)

convolution1.prepare_lut_psf(psf=psf1,
                             image_size=image_size,
                             image_pixel_size=image_pixel_size,
                             psf_pixel_size=psf_pixel_size)

convolution2 = ConvolutionManagerGPU(device=0,
                                     block_size=8,
                                     n_streams=10,
                                     debug=True)

convolution2.prepare_lut_psf(psf=psf2,
                             image_size=image_size,
                             image_pixel_size=image_pixel_size,
                             psf_pixel_size=psf_pixel_size)

convolution1.async_convolve(positions=pos)
convolution2.async_convolve(positions=pos)
results1 = convolution1.sync_get_results()
results2 = convolution2.sync_get_results()
results3 = convolution1.sync_convolve(positions=pos)
results4 = convolution2.sync_convolve(positions=pos)

print('\n\nComparación de resultados (1=3 y 2=4)')
print('1=2', np.all(results1 == results2))
print('1=3', np.all(results1 == results3))
print('1=4', np.all(results1 == results4))
print('2=4', np.all(results2 == results4))

for i in range(n // 20):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(results1[i])
    ax1.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax2.imshow(results2[i])
    ax2.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    plt.show()
