import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="catmu",
    version="0.1.0",
    author="Alejandro Ezequiel Mazzeo",
    author_email="ale.exactas@gmail.com",
    description="Convolution Accelerated by a Texture Mapping Unit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/alemazzeo/catmu",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Typing :: Typed',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
