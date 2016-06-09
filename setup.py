#!/usr/bin/python
# adenine setup script

from distutils.core import setup

# Package Version
from adenine import __version__ as version

setup(
    name='adenine',
    version=version,

    description=('A Data ExploratioN pIpeliNE'),
    long_description=open('README.md').read(),
    author='Samuele Fiorini, Federico Tomasi',
    author_email='{samuele.fiorini, federico.tomasi}@dibris.unige.it',
    maintainer='Samuele Fiorini, Federico Tomasi',
    maintainer_email='{samuele.fiorini, federico.tomasi}@dibris.unige.it',
    url='https://github.com/slipguru/adenine',
    download_url = 'https://github.com/slipguru/adenine/tarball/0.1.2',
    classifiers=[
	'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license = 'FreeBSD',

    packages=['adenine', 'adenine.core', 'adenine.utils'],
    requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'sklearn (>=0.17)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)',
              'fastcluster (>=1.1.20)'],
    scripts=['scripts/ade_run.py','scripts/ade_analysis.py'],
)
