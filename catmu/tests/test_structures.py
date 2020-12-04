import numpy as np
from ctypes import c_void_p
from catmu.structures import Image, Positions, LutPSF, ExpressionPSF, DevConfig


def test_image_2d():
    fake_image = np.arange(100*32*64).reshape((100, 32, 64))
    image = Image()
    assert image.data is None

    image1 = Image(image=fake_image, pixel_size=(2.0, 4.0))
    image2 = Image()
    image2.set_data(image=fake_image, pixel_size=(2.0, 4.0))

    for img in [image1, image2]:
        assert img.data == fake_image.ctypes.data_as(c_void_p).value
        assert img.N == 100
        assert img.width == 64
        assert img.height == 32
        assert img.depth == 1
        assert img.pixel_width == 4.0
        assert img.pixel_height == 2.0
        assert img.pixel_depth == 1.0

    image3 = Image(image=fake_image)
    assert image3.pixel_width == 1.0
    assert image3.pixel_height == 1.0
    assert image3.pixel_depth == 1.0


def test_image_3d():
    fake_image = np.arange(100*10*32*64).reshape((100, 10, 32, 64))
    image = Image()
    assert image.data is None

    image1 = Image(image=fake_image, pixel_size=(8.0, 2.0, 4.0))
    image2 = Image()
    image2.set_data(image=fake_image, pixel_size=(8.0, 2.0, 4.0))

    for img in [image1, image2]:
        assert img.data == fake_image.ctypes.data_as(c_void_p).value
        assert img.N == 100
        assert img.width == 64
        assert img.height == 32
        assert img.depth == 10
        assert img.pixel_width == 4.0
        assert img.pixel_height == 2.0
        assert img.pixel_depth == 8.0


def test_positions_2d():
    fake_positions = np.arange(100*6400*2).reshape((100, 6400, 2))
    pos = Positions()
    assert pos.data is None

    pos1 = Positions(positions=fake_positions)
    pos2 = Positions()
    pos2.set_data(positions=fake_positions)

    for pos in [pos1, pos2]:
        assert pos.data == fake_positions.ctypes.data_as(c_void_p).value
        assert pos.N == 100
        assert pos.n == 6400
        assert pos.dim == 2


def test_positions_3d():
    fake_positions = np.arange(100*6400*3).reshape((100, 6400, 3))
    positions = Positions()
    assert positions.data is None

    pos1 = Positions(positions=fake_positions)
    pos2 = Positions()
    pos2.set_data(positions=fake_positions)

    for pos in [pos1, pos2]:
        assert pos.data == fake_positions.ctypes.data_as(c_void_p).value
        assert pos.N == 100
        assert pos.n == 6400
        assert pos.dim == 3


def test_lut_psf_2d():
    fake_psf = np.arange(55*111).reshape((55, 111))
    psf_lut1 = LutPSF(psf_data=fake_psf, pixel_size=(2.0, 4.0))
    assert psf_lut1.data == fake_psf.ctypes.data_as(c_void_p).value
    assert psf_lut1.dim == 2
    assert psf_lut1.width == 111
    assert psf_lut1.height == 55
    assert psf_lut1.depth == 1
    assert psf_lut1.pixel_width == 4.0
    assert psf_lut1.pixel_height == 2.0
    assert psf_lut1.pixel_depth == 1.0

    psf_lut2 = LutPSF(psf_data=fake_psf)
    assert psf_lut2.pixel_width == 1.0
    assert psf_lut2.pixel_height == 1.0
    assert psf_lut2.pixel_depth == 1.0


def test_lut_psf_3d():
    fake_psf = np.arange(11*55*111).reshape((11, 55, 111))
    psf_lut = LutPSF(psf_data=fake_psf, pixel_size=(0.5, 2.0, 4.0))
    assert psf_lut.data == fake_psf.ctypes.data_as(c_void_p).value
    assert psf_lut.dim == 3
    assert psf_lut.width == 111
    assert psf_lut.height == 55
    assert psf_lut.depth == 11
    assert psf_lut.pixel_width == 4.0
    assert psf_lut.pixel_height == 2.0
    assert psf_lut.pixel_depth == 0.5


def test_expression_psf_2d():
    expression = ExpressionPSF(id_function=0, params=[1.0, 2.0, 0.0])
    assert expression.id_function == 0
    assert np.all(expression.params_array == np.array([1.0, 2.0, 0.0]))
    assert expression.n_params == 3
    assert expression.params == expression.params_array.ctypes.data_as(c_void_p).value


def test_dev_config():
    config = DevConfig(device=0, block_size=8, n_streams=17)
    assert config.device == 0
    assert config.n_streams == 17
    assert config.block_size == 8
