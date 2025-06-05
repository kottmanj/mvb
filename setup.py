from setuptools import setup, find_packages
import os
import sys

def read_requirements(fname):
    with open('requirements.txt') as file:
        lines=file.readlines()
        requirements = [line.strip() for line in lines]
    return requirements

requirements = read_requirements('requirements.txt')

setup(
    name='mvb',
    install_requires=requirements,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': [os.path.join('src')]
    }
)
