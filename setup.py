import sys
if sys.version_info < (3,):
    sys.exit('scdml requires Python >= 3.6')
from pathlib import Path

from setuptools import setup, find_packages


try:
    from scdml import __author__, __email__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = 'Jiaqili@zju.edu.cn'

setup(
    name='scdml',
    # version=__version__,
    # cmdclass=versioneer.get_cmdclass(),
    description='Single-Cell Analysis in Python.',
    long_description=Path('README.rst').read_text('utf-8'),
    url='https://github.com/JiaqiLiZju/sc_metric_learning',
    author=__author__,
    author_email=__email__,
    license='BSD',
    python_requires='>=3.6',
    install_requires=[
          'numpy',
          'scanpy',
          'scikit-learn',
          'tqdm',
          'torch',
          'torchvision',
          'pytorch_metric_learning',
          'captum',
    ],
    # extras_require=dict(
    #     louvain=['python-igraph', 'louvain>=0.6'],
    #     leiden=['python-igraph', 'leidenalg'],
    #     bbknn=['bbknn'],
    #     doc=['sphinx', 'sphinx_rtd_theme', 'sphinx_autodoc_typehints', 'scanpydoc'],
    #     test=['pytest>=4.4', 'dask[array]', 'fsspec', 'zappy', 'zarr'],
    # ),
    packages=find_packages(),
    # `package_data` does NOT work for source distributions!!!
    # you also need MANIFTEST.in
    # https://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute
    # package_data={'': '*.txt'},
    include_package_data=True,
    entry_points=dict(
        console_scripts=['scanpy=scanpy.cli:console_main'],
    ),
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
