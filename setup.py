import setuptools
from prototype import __version__


setuptools.setup(
    name='Spring-Prototype',
    version=__version__,
    author='Yuan Kun',
    author_email='yuankun@sensetime.com',
    description='Distributed General Image Classification Framework',
    url='http://gitlab.bj.sensetime.com/spring-ce/element/prototype.git',
    packages=setuptools.find_packages(),
    license='Internal',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: Internal',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ]
)
