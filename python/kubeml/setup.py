from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='kubeml',
    version='0.1.6rc2',
    description='Python tools for training Neural Networks with KubeML',
    author='Diego Albo Martínez',
    author_email="diego.albo.martinez@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=['kubeml'],
    install_requires=[
        'torch>=1.7',
        'redisai>=1.0.1',
        'pymongo>=3.11.1',
        'flask>=1.1.2'
    ],
    license="MIT",
)
