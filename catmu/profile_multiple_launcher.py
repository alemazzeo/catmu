import time
import click
from timeit import Timer
from typing import Tuple, NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions
from catmu.wrapper import ConvolveLibrary
from catmu.wrapper_stable import ConvolveLibraryMultiple



import GPUtil
try:
    gpu = GPUtil.getGPUs()[0]
    GPU_NAME = gpu.name
    GPU_SERIAL = gpu.serial
    print(f'\nGPU:    {GPU_NAME}'
          f'\nSERIAL: {GPU_SERIAL}')
except ValueError:
    raise RuntimeError('GPU NO DISPONIBLE')


class TimeitProfile(NamedTuple):
    mean: float
    std: float
    repeat: int
    number: int


def timeit_kernel(convolution_size: Tuple[int, int] = (64, 64),
                  image_pixel: Tuple[float, float] = (1.0, 1.0),
                  psf_pixel: Tuple[float, float] = (1.0, 1.0),
                  psf_size: Tuple[int, int] = (25, 25),
                  n_sources: int = 6000,
                  n_individuals: int = 100,
                  sigma: float = 2.0,
                  subpixel: int = 1,
                  block_size: int = 1,
                  timeout: float = 5.0):

    pos = make_n_random_positions(n_individuals, n_sources, convolution_size)
    psf = make_gaussian_psf_lut(psf_size, sigma)

    convolution = ConvolveLibraryMultiple(image_size=convolution_size,
                                          image_pixel_size=image_pixel,
                                          psf_pixel_size=psf_pixel,
                                          subpixel=subpixel,
                                          block_size=block_size,
                                          debug=False)

    convolution.positions = pos
    convolution.psf = psf

    convolution.launch()
    t = Timer("convolution.launch()", globals={'convolution': convolution})
    number, elapsed_time = t.autorange()
    repeat = int(timeout / elapsed_time)

    if repeat < 1:
        results = np.asarray([elapsed_time]) / number / n_individuals
    else:
        results = np.asarray(t.repeat(repeat=repeat, number=number)) / number / n_individuals

    return TimeitProfile(mean=float(np.mean(results)),
                         std=float(np.std(results)),
                         repeat=results.size,
                         number=number)


def profile_kernel(convolution_size: List[int],
                   psf_size: List[int],
                   n_sources: List[int],
                   n_individuals: int = 100,
                   subpixel: int = 1,
                   block_size: int = 1,
                   timeout: float = 1.0) -> pd.DataFrame:

    image_pixel: Tuple[float, float] = (1.0, 1.0)
    psf_pixel: Tuple[float, float] = (1.0, 1.0)
    sigma: float = 2.0

    df = pd.DataFrame()

    eta = timeout * len(convolution_size) * len(psf_size) * len(n_sources)
    print(f'ETA: {time.strftime("%H:%M:%S", time.gmtime(eta))}')

    for x in convolution_size:
        for y in psf_size:
            print(f'Convolution size: ({x:3d} x {x:3d}) '
                  f'Psf size: ({y:3d} x {y:3d})')
            for z in n_sources:
                try:
                    result = timeit_kernel(convolution_size=(x, x),
                                           image_pixel=image_pixel,
                                           psf_size=(y, y),
                                           psf_pixel=psf_pixel,
                                           n_sources=z,
                                           n_individuals=n_individuals,
                                           sigma=sigma,
                                           subpixel=subpixel,
                                           block_size=block_size,
                                           timeout=timeout)
                except RuntimeError as e:
                    print(f"Error on {x} -> {y} -> {z}\n {e}")
                    continue

                new_row = {'timestamp': time.strftime("%m%d%Y-%H%M%S"),
                           'kernel': 'multiple_launcher',
                           'convolution_size': x,
                           'image_pixel': image_pixel,
                           'psf_size': y,
                           'psf_pixel': psf_pixel,
                           'n_sources': z,
                           'n_individuals': n_individuals,
                           'sigma': sigma,
                           'mean': result.mean,
                           'std': result.std,
                           'repeat': result.repeat,
                           'number': result.number,
                           'subpixel': subpixel,
                           'block_size': block_size,
                           'gpu': GPU_NAME,
                           'gpu_serial': GPU_SERIAL}

                df = df.append(new_row, ignore_index=True)
    return df


def test(timeout: float = 5.0, subpixel: int = 1, block_size: int = 8, plot: bool = True) -> pd.DataFrame:
    test3a = profile_kernel(convolution_size=[x * 2 + 2 for x in range(32)],
                            psf_size=[16],
                            n_sources=[3200],
                            subpixel=subpixel,
                            block_size=block_size,
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='convolution_size', y='mean', data=test3a, height=8)
        print(np.polyfit(test3a['convolution_size'], test3a['mean'], 1))
        plt.show()

    test3b = profile_kernel(convolution_size=[32],
                            psf_size=[x * 2 + 2 for x in range(32)],
                            n_sources=[3200],
                            subpixel=subpixel,
                            block_size=block_size,
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='psf_size', y='mean', data=test3b, height=8)
        print(np.polyfit(test3b['psf_size'], test3b['mean'], 1))
        plt.show()

    test3c = profile_kernel(convolution_size=[32],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            subpixel=subpixel,
                            block_size=block_size,
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='n_sources', y='mean', data=test3c, height=8)
        print(np.polyfit(test3c['n_sources'], test3c['mean'], 1))
        plt.show()

    return pd.concat([test3a, test3b, test3c], ignore_index=True)


@click.command()
@click.option('--timeout', default=1.0)
@click.option('--subpixel', default=1)
@click.option('--block_size', default=8)
def run(timeout, subpixel, block_size):
    df = test(timeout=timeout, subpixel=subpixel, block_size=block_size, plot=True)
    df.to_csv(f'test_multiple_{time.strftime("%m%d%Y_%H%M%S")}.csv')


if __name__ == '__main__':
    run()
