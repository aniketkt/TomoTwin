#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: atekawade
"""

from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='tomo_twin',
    url='https://github.com/aniketkt/TomoTwin',
    author='Aniket Tekawade',
    author_email='atekawade@anl.gov',
    # Needed to actually package something
    packages= ['tomo_twin'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'scipy', 'h5py', 'matplotlib', \
                      'opencv-python', 'porespy', \
                      'ConfigArgParse', 'tqdm', 'ipython', 'ct-segnet'],
    version=open('VERSION').read().strip(),
    license='BSD',
    description='A simple digital twin for synchrotron tomography',
#     long_description=open('README.md').read(),
)


