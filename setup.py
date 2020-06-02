from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(  name = 'shared_operation',
        version = '0.1',
        description = 'Package providing operation for shared use by pystxm (COSMIC/ALS) and xi-cam.',
        author = 'Juliane Reinhardt',
        author_email = 'jreinhardt@lbl.gov',
        packages = find_packages(include=['shared_operations', 'shared_operations.*']),
        python_requires='>=3'
        )
