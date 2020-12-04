# -*- coding: utf-8 -*-
"""
    Prueba de funcionamiento para 2 GPUs.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import matplotlib.pyplot as plt

from catmu import ConvolutionManagerGPU, CatmuError
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

# En este ejemplo n corresponde al trabajo de cada GPU
n = 50

image_size = (64, 64)
image_pixel_size = (1.0, 1.0)
psf_pixel_size = (1.0, 1.0)
psf_size = (10, 10)
n_sources = 6400
sigma = 2.0

pos1 = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)
pos2 = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

try:
    convolution_gpu_0 = ConvolutionManagerGPU(device=0,
                                              block_size=8,
                                              n_streams=10,
                                              debug=False)

    convolution_gpu_1 = ConvolutionManagerGPU(device=1,
                                              block_size=8,
                                              n_streams=10,
                                              debug=False)

    convolution_gpu_0.prepare_lut_psf(psf=psf,
                                      image_size=image_size,
                                      image_pixel_size=image_pixel_size,
                                      psf_pixel_size=psf_pixel_size)

    convolution_gpu_1.prepare_lut_psf(psf=psf,
                                      image_size=image_size,
                                      image_pixel_size=image_pixel_size,
                                      psf_pixel_size=psf_pixel_size)

except CatmuError as e:
    if e.code == 101:
        print("Prueba abortada: No se dispone de dos GPUs en el sistema")
        exit()
        raise
    else:
        raise

# Lanza las convoluciones en forma asincrónica
convolution_gpu_0.async_convolve(positions=pos1)
convolution_gpu_1.async_convolve(positions=pos2)

# Espera los resultados
results_gpu_0 = convolution_gpu_0.sync_get_results(get_copy=True)
results_gpu_1 = convolution_gpu_1.sync_get_results(get_copy=True)

for i in range(n // 10):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(f"GPU 0 - Individuo {i}")
    ax1.imshow(results_gpu_0[i])
    ax1.plot(pos1[i][:, 0], pos1[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    ax2.set_title(f"GPU 1 - Individuo {i + n}")
    ax2.imshow(results_gpu_1[i])
    ax2.plot(pos2[i][:, 0], pos2[i][:, 1], color='k', ls='', marker='.', markersize=1.0)
    plt.show()
