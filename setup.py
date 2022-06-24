from setuptools import setup

metadata = dict(
    name="fruits",
    version="0.9.3",
    author="alienkrieg",
    author_email="alienkrieg@gmail.com",
    description="Feature Extraction Using Iterated Sums",
    packages=[
        "fruits",
        "fruits.core",
        "fruits.preparation",
        "fruits.sieving",
        "fruits.signature",
        "fruits.words",
    ],
    long_description=open("README.md").read(),
    install_requires=[
        "numpy >= 1.19.2",
        "numba >= 0.52.0",
    ],
    python_requires=">=3.9, <3.10",
)

if __name__ == "__main__":
    setup(**metadata)
