import numpy as np
from catmu import (get_available_devices,
                   get_device_properties,
                   ConvolutionManagerGPU,
                   ConvolutionManagerCPU,
                   CatmuError)
from catmu.analysis_tools import make_gaussian_psf_lut, make_n_random_positions


def test_get_available_devices():
    assert get_available_devices() >= 0


# noinspection DuplicatedCode
def test_gpu_lut_convolution_2d():
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

    assert convolution.loop_counter == 1
    assert convolution.last_elapsed_time > 0
    assert convolution.active is True
    convolution


# noinspection DuplicatedCode
def test_gpu_lut_convolution_3d():
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


# noinspection DuplicatedCode
def test_gpu_expression_convolution_2d():
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


# noinspection DuplicatedCode
def test_cpu_expression_convolution_2d():
    n = 10
    image_size = (64, 64)
    n_sources = 800

    pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

    convolution1 = ConvolutionManagerCPU(open_mp=True,
                                         debug=True)

    convolution1.prepare_expression_psf(id_function=0, params=[1.0, 2.0, 0.0])
    results1 = convolution1.sync_convolve(positions=pos)

    convolution2 = ConvolutionManagerCPU(open_mp=False,
                                         debug=True)

    convolution2.prepare_expression_psf(id_function=0, params=[1.0, 2.0, 0.0])
    results2 = convolution2.sync_convolve(positions=pos)


# noinspection DuplicatedCode
def test_cpu_lut_convolution_2d():
    n = 2
    image_size = (32, 32)
    image_pixel_size = (1.0, 1.0)
    psf_pixel_size = (1.0, 1.0)
    psf_size = (10, 10)
    n_sources = 100
    sigma = 2.0

    pos = make_n_random_positions(n=n, n_sources=n_sources, convolution_size=image_size)

    psf = make_gaussian_psf_lut(psf_size=psf_size, sigma=sigma)

    convolution_cpu = ConvolutionManagerCPU(open_mp=True,
                                            debug=True)

    convolution_cpu.prepare_lut_psf(psf=psf,
                                    image_size=image_size,
                                    image_pixel_size=image_pixel_size,
                                    psf_pixel_size=psf_pixel_size)

    results = convolution_cpu.sync_convolve(positions=pos)

    convolution_cpu = ConvolutionManagerCPU(open_mp=False,
                                            debug=True)

    convolution_cpu.prepare_lut_psf(psf=psf,
                                    image_size=image_size,
                                    image_pixel_size=image_pixel_size,
                                    psf_pixel_size=psf_pixel_size)

    results = convolution_cpu.sync_convolve(positions=pos)


# noinspection DuplicatedCode
def test_convolution_multiple_contexts():
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

    assert not np.all(results1 == results2)
    assert np.all(results1 == results3)
    assert not np.all(results1 == results4)
    assert np.all(results2 == results4)

    convolution1.prepare_lut_psf(psf=psf2,
                                 image_size=image_size,
                                 image_pixel_size=image_pixel_size,
                                 psf_pixel_size=psf_pixel_size)

    convolution1.async_convolve(positions=pos)
    results1 = convolution1.sync_get_results()
    results3 = convolution1.sync_convolve(positions=pos)

    assert np.all(results1 == results2)
    assert np.all(results1 == results3)
    assert np.all(results1 == results4)
    assert np.all(results2 == results4)


def test_get_device_properties():
    prop = get_device_properties(0)
    print("\nDevice Properties\n")
    print(prop)
