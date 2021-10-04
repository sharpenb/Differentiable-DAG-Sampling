from setuptools import setup, find_packages

setup(name='differentiable-dag-sampling',
      version='0.1',
      description='Differentiable DAG Sampling',
      author='Anonymous',
      author_email='anonymous@mail.com',
      packages=['src'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm',
                        'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2', 'seml'],
      zip_safe=False)
