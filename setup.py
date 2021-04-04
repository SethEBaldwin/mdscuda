from setuptools import setup, find_packages

setup(
    packages = find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'numba>=0.51.0'
    ]
)
