from setuptools import setup
from Cython.Build import cythonize

sklearn_min_version = '0.17.1'
sklearn_req_str = "uplift requires sklearn >= {}.\n".format(sklearn_min_version)

try:
    import sklearn
    if sklearn.__version__ < sklearn_min_version:
        raise ImportError("Your installation of Scikit-learn {} is out-of-date.\n{}"
                          .format(sklearn.__version__, sklearn_req_str))
except ImportError:
    raise ImportError("Scikit-learn is not installed.\n{}"
                      .format(sklearn_req_str))

setup(name='uplift',
      maintainer='Paulius Sarka',
      maintainer_email='paulius.sarka@gmail.com',
      license='new BSD',
      version='0.1',
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: C',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Programming Language :: Python :: 3.4'],
      ext_modules=cythonize('uplift/tree/_criterion.pyx'))
