import time
from timeit import Timer
from typing import Tuple, NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catmu.analysis_tools import make_gaussian_psf_lut, make_random_positions
from catmu.wrapper import ConvolveLibrary


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


def timeit_kernel(kernel: str = 'd00',
                  convolution_size: Tuple[int, int] = (64, 64),
                  image_pixel: Tuple[float, float] = (1.0, 1.0),
                  psf_pixel: Tuple[float, float] = (1.0, 1.0),
                  psf_size: Tuple[int, int] = (25, 25),
                  n_sources: int = 6000,
                  sigma: float = 2.0,
                  timeout: float = 5.0):
    pos = make_random_positions(n_sources, convolution_size)
    psf = make_gaussian_psf_lut(psf_size, sigma)

    convolution = ConvolveLibrary(kernel,
                                  convolution_size,
                                  image_pixel,
                                  psf_pixel,
                                  debug=False)

    convolution.positions = pos
    convolution.psf = psf

    convolution.launch()
    t = Timer("convolution.launch()", globals={'convolution': convolution})
    number, elapsed_time = t.autorange()
    repeat = int(timeout / elapsed_time)

    if repeat < 1:
        results = np.asarray([elapsed_time]) / number
    else:
        results = np.asarray(t.repeat(repeat=repeat, number=number)) / number

    return TimeitProfile(mean=float(np.mean(results)),
                         std=float(np.std(results)),
                         repeat=results.size,
                         number=number)


def profile_kernel(kernel: str,
                   convolution_size: List[int],
                   psf_size: List[int],
                   n_sources: List[int],
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
                    result = timeit_kernel(kernel=kernel,
                                           convolution_size=(x, x),
                                           image_pixel=image_pixel,
                                           psf_size=(y, y),
                                           psf_pixel=psf_pixel,
                                           n_sources=z,
                                           sigma=sigma,
                                           timeout=timeout)
                except RuntimeError as e:
                    print(f"Error on {x} -> {y} -> {z}\n {e}")
                    continue

                new_row = {'timestamp': time.strftime("%m%d%Y-%H%M%S"),
                           'kernel': kernel,
                           'convolution_size': x,
                           'image_pixel': image_pixel,
                           'psf_size': y,
                           'psf_pixel': psf_pixel,
                           'n_sources': z,
                           'sigma': sigma,
                           'mean': result.mean,
                           'std': result.std,
                           'repeat': result.repeat,
                           'number': result.number,
                           'gpu': GPU_NAME,
                           'gpu_serial': GPU_SERIAL}

                df = df.append(new_row, ignore_index=True)
    return df


def test1(timeout: float = 5.0, plot: bool = True) -> pd.DataFrame:
    test1a = profile_kernel(kernel='d00',
                            convolution_size=[x * 2 + 2 for x in range(32)],
                            psf_size=[16],
                            n_sources=[3200],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='convolution_size', y='mean', data=test1a, height=8)
        print(np.polyfit(test1a['convolution_size'], test1a['mean'], 1))
        plt.show()

    test1b = profile_kernel(kernel='d00',
                            convolution_size=[32],
                            psf_size=[x * 2 + 2 for x in range(32)],
                            n_sources=[3200],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='psf_size', y='mean', data=test1b, height=8)
        print(np.polyfit(test1a['psf_size'], test1a['mean'], 1))
        plt.show()

    test1c = profile_kernel(kernel='d00',
                            convolution_size=[32],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='n_sources', y='mean', data=test1c, height=8)
        print(np.polyfit(test1a['n_sources'], test1a['mean'], 1))
        plt.show()

    return pd.concat([test1a, test1b, test1c], ignore_index=True)


def test2(timeout: float = 5.0, plot: bool = True) -> pd.DataFrame:
    test2a = profile_kernel(kernel='d01',
                            convolution_size=[64],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    test2b = profile_kernel(kernel='d02',
                            convolution_size=[64],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    data = pd.concat([test2a, test2b], ignore_index=True)

    if plot is True:
        sns.lmplot(x='n_sources', y='mean', hue='kernel', data=data, height=8)
        print(np.polyfit(data[data.kernel == 'd01']['n_sources'], data[data.kernel == 'd01']['mean'], 1))
        print(np.polyfit(data[data.kernel == 'd02']['n_sources'], data[data.kernel == 'd02']['mean'], 1))
        plt.show()

    return data


def test3(timeout: float = 5.0, plot: bool = True) -> pd.DataFrame:
    test3a = profile_kernel(kernel='v0_2',
                            convolution_size=[x * 2 + 2 for x in range(32)],
                            psf_size=[16],
                            n_sources=[3200],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='convolution_size', y='mean', data=test3a, height=8)
        print(np.polyfit(test3a['convolution_size'], test3a['mean'], 1))
        plt.show()

    test3b = profile_kernel(kernel='v0_2',
                            convolution_size=[32],
                            psf_size=[x * 2 + 2 for x in range(32)],
                            n_sources=[3200],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='psf_size', y='mean', data=test3b, height=8)
        print(np.polyfit(test3b['psf_size'], test3b['mean'], 1))
        plt.show()

    test3c = profile_kernel(kernel='v0_2',
                            convolution_size=[32],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    if plot is True:
        sns.lmplot(x='n_sources', y='mean', data=test3c, height=8)
        print(np.polyfit(test3c['n_sources'], test3c['mean'], 1))
        plt.show()

    return pd.concat([test3a, test3b, test3c], ignore_index=True)


def test4(timeout: float = 5.0, plot: bool = True) -> pd.DataFrame:
    test4a = profile_kernel(kernel='v0_2',
                            convolution_size=[64],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    test4b = profile_kernel(kernel='v0_3',
                            convolution_size=[64],
                            psf_size=[16],
                            n_sources=[x * 128 + 128 for x in range(32)],
                            timeout=timeout)

    data = pd.concat([test4a, test4b], ignore_index=True)

    if plot is True:
        sns.lmplot(x='n_sources', y='mean', hue='kernel', data=data, height=8)
        print(np.polyfit(data[data.kernel == 'v0_2']['n_sources'], data[data.kernel == 'v0_2']['mean'], 1))
        print(np.polyfit(data[data.kernel == 'v0_3']['n_sources'], data[data.kernel == 'v0_3']['mean'], 1))
        plt.show()

    return data


test1(timeout=0.5, plot=False).to_csv(f'test1_{time.strftime("%m%d%Y_%H%M%S")}.csv')
test2(timeout=0.5, plot=False).to_csv(f'test2_{time.strftime("%m%d%Y_%H%M%S")}.csv')
test3(timeout=0.5, plot=False).to_csv(f'test3_{time.strftime("%m%d%Y_%H%M%S")}.csv')
test4(timeout=0.5, plot=False).to_csv(f'test4_{time.strftime("%m%d%Y_%H%M%S")}.csv')
