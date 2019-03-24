#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Jean",
    author_email='12112853+jeanlst@users.noreply.github.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="Project assignment for SOEN-691. Falling in the category of Algorithm Implementation where we will implement different clustering algorithms and compare them in different dimensionalities and granularities.",
    entry_points={
        'console_scripts': [
            'soen691_clustering_project=soen691_clustering_project.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='soen691_clustering_project',
    name='soen691_clustering_project',
    packages=find_packages(include=['soen691_clustering_project']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jeanlst/soen691_clustering_project',
    version='0.1.0',
    zip_safe=False,
)
