# -*- coding: utf-8 -*-
"""
    Prueba básica de funcionamiento.

    :copyright: 2019 by Fotónica FIUBA, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from catmu import ConvolutionManagerGPU
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions

n = 2
image_size = (64, 64, 64)
image_pixel_size = (1.0, 1.0, 1.0)
psf_pixel_size = (1.0, 1.0, 1.0)
psf_size = (10, 10, 10)
n_sources = 100
sigma = 2.0

pos = make_n_random_positions(n=n,
                              n_dim=3,
                              n_sources=n_sources,
                              convolution_size=image_size)

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

for i, image in enumerate(results[0]):
    u8_image = (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(u8_image)
    img.save(f"./tiff/file_{i:02d}.png")


for i in range(image_size[0]):
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(results[0, i])
    ax1.plot(pos[0][:, -3], pos[0][:, -2], color='w', ls='', marker='.', markersize=1.0)
    plt.tight_layout()
    plt.show()
