from pysuppose.bases.device import CPU, GPU
from pysuppose.bases.psf import GaussianPSF
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions
from catmu.wrapper_stable import ConvolveLibraryMultiple
import time

convolution_size = (64, 64)
psf_size = (10, 10)
n_sources = 6000
n_individuals = 100
iterations = 2

cpu_expresion = CPU()
cpu_expresion.set_output_shape(convolution_size)

gpu_expresion = GPU(0)
gpu_expresion.set_output_shape(convolution_size)

psf_expresion = GaussianPSF()
psf_lut = make_gaussian_psf_lut(psf_size=psf_size)

positions = make_n_random_positions(n=n_individuals, n_sources=n_sources, convolution_size=convolution_size)

cpu_expresion.convolve_positions(positions[0], psf=psf_expresion)
t = time.time()
for i in range(iterations):
    for p in positions:
        cpu_expresion.convolve_positions(p, psf=psf_expresion)
print(f'CPU: {(time.time() - t) * 1000 / n_individuals / iterations}mS')

gpu_expresion.convolve_positions(positions[0], psf=psf_expresion)
t = time.time()
for i in range(iterations):
    for p in positions:
        gpu_expresion.convolve_positions(p, psf=psf_expresion)
print(f'GPU: {(time.time() - t) * 1000 / n_individuals / iterations}mS')

convolution = ConvolveLibraryMultiple(image_size=convolution_size,
                                      subpixel=4,
                                      block_size=8,
                                      debug=False)

convolution.positions = positions
convolution.psf = psf_lut

t = time.time()
for i in range(iterations):
    convolution.launch()
print(f'LUT: {(time.time() - t) * 1000 / n_individuals / iterations}mS')
