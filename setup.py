import re

import setuptools
import distutils.cmd
import distutils.log
import subprocess
import sys

from setuptools.command.test import test as TestCommand

# Getting version:
with open("tbats/__init__.py") as init_file:
    version = re.search("__version__ = \'(.*?)\'", init_file.read()).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args).append('test/'))
        sys.exit(errno)


class RTestCommand(TestCommand):
    description = 'Run comparison to R forecast package implementation of BATS and TBATS (REQUIRES R)'

    def run_tests(self):
        """Run command."""
        command = ['pytest', 'test_R']
        # if self.pylint_rcfile:
        #     command.append('--rcfile=%s' % self.pylint_rcfile)
        # command.append(os.getcwd())
        self.announce(
            'Running command: %s' % str(command),
            level=distutils.log.INFO)
        subprocess.check_call(command)

setuptools.setup(
    name='tbats',
    version=version,
    packages=setuptools.find_packages(exclude=('test', 'test_R')),
    url='https://github.com/intive-DataScience/tbats',
    license='MIT License',
    author='Grzegorz Skorupa (intive)',
    author_email='grzegorz.skorupa@intive.com',
    description='BATS and TBATS for time series forecasting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'pmdarima', 'scikit-learn'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['pip-tools', 'pytest', 'rpy2'],
    },
    cmdclass={
        'test': PyTest,
        'test_r': RTestCommand,
    },

)
