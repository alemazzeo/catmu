# -*- coding: utf-8 -*-
"""
    Prueba básica de funcionamiento.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import matplotlib.pyplot as plt
import numpy as np

from catmu import ConvolutionManagerGPU, list_devices
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 100
image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0 / 32.0, 1.0 / 32.0)
psf_size = (513, 513)
n_sources = 800
sigma = np.sqrt(2.0) * 32

pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

convolution = ConvolutionManagerGPU(device=0,
                                    block_size=8,
                                    n_streams=10,
                                    debug=True)

convolution.prepare_expression_psf(id_function=0, params=[1.0, 2.0, 0.0])
results1 = convolution.sync_convolve(positions=pos)

convolution.prepare_lut_psf(psf=psf,
                            image_size=image_size,
                            image_pixel_size=image_pixel_size,
                            psf_pixel_size=psf_pixel_size)
results2 = convolution.sync_convolve(positions=pos)

for i in range(n // 20):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='all', sharey='all')
    ax1.imshow(results1[i])
    ax2.imshow(results2[i])
    ax3.imshow(np.abs(results2[i] - results1[i]))
    ax1.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax2.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax3.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    plt.show()
