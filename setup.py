from setuptools import setup

metadata = dict(
    name='fruits',
    version='0.5.0',
    author='alienkrieg',
    author_email='alienkrieg@gmail.com',
    description='Feature extRation Using ITerated Sums',
    packages=['fruits'],
    ext_package="",
    ext_modules=[],
    py_modules=[
                'fruits.__init__',
                'fruits.core',
                'fruits.features',
                'fruits.iterators',
                'fruits.preparateurs',
                'fruits.main',
                'fruits.accelerated',
               ],
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.19.2",
        "numba >= 0.52.0",
        "pytest >= 6.2.4",
    ],
    python_requires=">=3.6, <3.10",
)

if __name__ == "__main__":
    setup(**metadata)
