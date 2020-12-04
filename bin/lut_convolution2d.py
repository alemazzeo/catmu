# -*- coding: utf-8 -*-
"""
    Prueba básica de funcionamiento.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import matplotlib.pyplot as plt

from catmu import ConvolutionManagerGPU
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 100
image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0, 1.0)
psf_size = (10, 10)
n_sources = 6400
sigma = 2.0

pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

convolution = ConvolutionManagerGPU(device=0,
                                    block_size=8,
                                    n_streams=10,
                                    debug=True)

convolution.prepare_lut_psf(psf=psf,
                            image_size=image_size,
                            image_pixel_size=image_pixel_size,
                            psf_pixel_size=psf_pixel_size)

results = convolution.sync_convolve(positions=pos)

for i in range(n // 20):
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(results[i])
    ax1.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    plt.show()
