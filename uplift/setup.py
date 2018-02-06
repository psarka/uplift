import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('uplift', parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('preprocessing')

    # submodules which have their own setup.py
    # leave out "linear_model" and "utils" for now; add them after cblas below
    config.add_subpackage('ensemble')
    config.add_subpackage('externals')
    config.add_subpackage('tree')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
