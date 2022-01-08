try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

install_requires = [
    'numpy>=1.19.2',
    'dill>=0.3.4'
]

if __name__ == '__main__':
    setup(name = 'trchime',
          version = '1.0',
          author = 'yatorho',
          author_email = '3227068950@qq.com',
          maintainer = 'yatorho',
          maintainer_email = '3227068950@qq.com',
          platforms = ['Windows', 'Unix/Linux'],
          keywords = ['deep-learning', 'machine-learning', 'autograd'],
          packages = ['trchime',
                      'trchime.call',
                      'trchime.core',
                      'trchime.core.dtype',
                      'trchime.core.gather',
                      'trchime.core.initial',
                      'trchime.nn',
                      'trchime.random',
                      'trchime.training'],
          classifiers = ["Natural Language :: English",
                         "Operating System :: Microsoft :: Windows",
                         "Operating System :: Unix",
                         "Programming Language :: Python :: 3"],
          zip_safe = False,
          install_requires = install_requires,
          description = 'a tiny deep-learning framework')
