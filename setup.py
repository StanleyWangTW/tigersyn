from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

VESION = '0.0.6'
DESCRIPTION = 'Processing MRI images based on deep-learning'
CLASSIFIERS = [
    'Intended Audience :: Developers', 'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: MIT License',
    "Operating System :: Microsoft :: Windows"
]

setup(name='tigersyn',
      url='https://github.com/StanleyWangTW/tigersyn',
      version=VESION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='X. S. Wang',
      author_email='',
      License='MIT',
      packages=find_packages(),
      include_package_data=True,
      classifiers=CLASSIFIERS,
      python_requires='>=3.7',
      install_requires=['numpy', 'nibabel', 'nilearn', 'onnxruntime'])
