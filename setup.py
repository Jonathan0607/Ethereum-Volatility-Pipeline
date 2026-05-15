from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'monte_carlo',
        ['src/cpp/monte_carlo.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14', '-Wall'],
    ),
]

setup(
    name='monte_carlo',
    ext_modules=ext_modules,
)
