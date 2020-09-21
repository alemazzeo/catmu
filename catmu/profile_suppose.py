from pysuppose.bases.device import CPU, GPU
from pysuppose.bases.psf import GaussianPSF
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions
from catmu.wrapper_stable import ConvolveLibraryMultiple
import time
import argparse

parser = argparse.ArgumentParser(description='Profiler')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--psf_size', type=int, default=10)
parser.add_argument('--n_sources', type=int, default=6000)
parser.add_argument('--n_individuals', type=int, default=100)
parser.add_argument('--iterations', type=int, default=100)

parser.add_argument('--subpixel', type=int, default=4)
parser.add_argument('--block_size', type=int, default=8)

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--lut', action='store_true')

args = parser.parse_args()

convolution_size = (args.image_size, args.image_size)
psf_size = (args.psf_size, args.psf_size)
n_sources = args.n_sources
n_individuals = args.n_individuals
iterations = args.iterations

cpu_expresion = CPU()
cpu_expresion.set_output_shape(convolution_size)

gpu_expresion = GPU(0)
gpu_expresion.set_output_shape(convolution_size)

psf_expresion = GaussianPSF()
psf_lut = make_gaussian_psf_lut(psf_size=psf_size)

positions = make_n_random_positions(n=n_individuals, n_sources=n_sources, convolution_size=convolution_size)

convolution = ConvolveLibraryMultiple(image_size=convolution_size,
                                      subpixel=args.subpixel,
                                      block_size=args.block_size,
                                      debug=False)

convolution.positions = positions
convolution.psf = psf_lut


def run_cpu():
    cpu_expresion.convolve_positions(positions[0], psf=psf_expresion)
    t = time.time()
    for i in range(iterations):
        for p in positions:
            cpu_expresion.convolve_positions(p, psf=psf_expresion)
    total = time.time() - t
    print(f'CPU: {total * 1000 / n_individuals / iterations}mS')
    return total


def run_gpu():
    gpu_expresion.convolve_positions(positions[0], psf=psf_expresion)
    t = time.time()
    for i in range(iterations):
        for p in positions:
            gpu_expresion.convolve_positions(p, psf=psf_expresion)
    total = time.time() - t
    print(f'GPU: {total * 1000 / n_individuals / iterations}mS')
    return total


def run_lut():
    convolution.launch()
    t = time.time()
    for i in range(iterations):
        convolution.launch()
    total = time.time() - t
    print(f'LUT: {total * 1000 / n_individuals / iterations}mS')
    return total


if args.cpu is True:
    run_cpu()

if args.gpu is True:
    run_gpu()

if args.lut is True:
    run_lut()
