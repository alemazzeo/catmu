"""
    catmu
    ~~~~~~~~~~~~~
    Convolution Accelerated by the Texture Mapping Unit of GPU.
    :copyright: 2020 by catmu Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from . import analysis_tools, tmu_simulator, wrapper


def compile_all():
    import subprocess
    subprocess.run(f'make -C {__file__.replace("__init__.py", "")}/cuda_sources all', shell=True)


def compile_with_debug():
    import subprocess
    subprocess.run(f'make -C {__file__.replace("__init__.py", "")}/cuda_sources debug', shell=True)


def compile_without_debug():
    import subprocess
    subprocess.run(f'make -C {__file__.replace("__init__.py", "")}/cuda_sources nodebug', shell=True)


__all__ = [
    analysis_tools,
    tmu_simulator,
    wrapper
]