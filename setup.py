from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(  name = 'shared operations',
        version = '0.1',
        description = 'Package providing operation for shared use by pystxm (COSMIC/ALS) and xi-cam.',
        author = 'Juliane Reinhardt',
        author_email = 'jreinhardt@lbl.gov',
        url = 'https://github.com/JulReinhardt/shared_operations.git',
        packages = ['shop', 'shop.correction'],
        python_requires='>=3'
        )
