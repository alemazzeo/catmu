from catmu.analysis_tools import (make_gaussian_psf_lut,
                                  make_n_random_positions)


def test_make_gaussian_psf_lut_2d():
    make_gaussian_psf_lut(psf_size=(25, 25), sigma=2.0)


def test_make_gaussian_psf_lut_3d():
    make_gaussian_psf_lut(psf_size=(10, 25, 25), sigma=2.0)


def test_make_n_random_positions_2d():
    make_n_random_positions(n=100, n_sources=6400, n_dim=2, convolution_size=(64, 64))


def test_make_n_random_positions_3d():
    make_n_random_positions(n=100, n_sources=6400, n_dim=3, convolution_size=(10, 64, 64))
