from setuptools import setup

setup(
    name='kubeml',
    version='0.1',
    description='Python tools for training Neural Networks with KubeML',
    author='Diego Albo Martínez',
    packages=['kubeml'],
    install_requires=[
        'torch>=1.7',
        'flask',
        'redisai>=1.0.1',
        'pymongo>=3.11.1',
        'flask>=1.1.2'
    ],
)
