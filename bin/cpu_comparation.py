# -*- coding: utf-8 -*-
"""
    Comparación de resultados entre GPU y CPU en versión LUT.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import matplotlib.pyplot as plt
import numpy as np

from catmu import ConvolutionManagerGPU, ConvolutionManagerCPU
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 5
image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0, 1.0)
psf_size = (64, 64)
n_sources = 6400
sigma = 5

pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

convolution_gpu = ConvolutionManagerGPU(device=0,
                                        block_size=8,
                                        n_streams=10,
                                        debug=True)

convolution_gpu.prepare_lut_psf(psf=psf,
                                image_size=image_size,
                                image_pixel_size=image_pixel_size,
                                psf_pixel_size=psf_pixel_size)

convolution_cpu = ConvolutionManagerCPU(open_mp=True,
                                        debug=True)

convolution_cpu.prepare_lut_psf(psf=psf,
                                image_size=image_size,
                                image_pixel_size=image_pixel_size,
                                psf_pixel_size=psf_pixel_size)

results_gpu = convolution_gpu.sync_convolve(positions=pos)
results_cpu = convolution_cpu.sync_convolve(positions=pos)

for i in range(n):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_title('Resultado en GPU')
    ax1.imshow(results_gpu[i])
    ax1.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax2.set_title('Resultado en CPU')
    ax2.imshow(results_cpu[i])
    ax2.plot(pos[i][:, 0], pos[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax3.set_title(r'$\left| GPU-CPU \right|\; / \; max(\left|GPU\right|) $')
    ax3.imshow(np.abs(results_gpu[i] - results_cpu[i]) / np.max(np.abs(results_gpu[i])) * 100)
    ax4.set_title("Histograma de errores")
    ax4.hist((np.abs(results_gpu[i] - results_cpu[i]) / np.max(np.abs(results_gpu[i]))).flatten() * 100)
    fig.tight_layout()
    plt.show()
