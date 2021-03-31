from setuptools import setup, find_packages, Extension

setup(
    packages = find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.6.0',
        'numba>=53.0'
    ]
)
