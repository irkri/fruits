from setuptools import setup

if __name__ == "__main__":
    setup(
        name='fruits',
        version='0.5.0',
        author='alienkrieg',
        author_email='alienkrieg@gmail.com',
        description='Feature extRation Using ITerated Sums',
        packages=['fruits'],
        ext_package="",
        ext_modules=[],
        package_dir={'fruits':'fruits'},
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
    )
