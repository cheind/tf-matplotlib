# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='tfmpl',
    version=open('tfmpl/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Seamlessly integrate matplotlib figures tensorflow summaries.',    
    author='Christoph Heindl',
    url='https://github.com/cheind/tf-matplotlib',
    license='MIT',
    install_requires=required,
    packages=['tfmpl', 'tfmpl.samples', 'tfmpl.tests'],
    include_package_data=True,
    keywords='tensorflow matplotlib tensorboard'
)