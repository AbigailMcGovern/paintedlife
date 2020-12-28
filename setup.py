from setuptools import setup
from setuptools import find_packages

def get_install_required():
    with open("./requirements.txt", "r") as reqs:
        requirements = reqs.readlines()
    return [r.rstrip() for r in requirements]

DESCRIPTION = "Conway's Game of Life, napari style (paintbrush inc.)"

setup(
      name='paintedlife', 
      version='0.0.3', 
      description=DESCRIPTION, 
      long_description_content_type='text/markdown',
      author='Abigail S. McGovern',
      author_email='abigail_mcgovern@hotmail.com',
      url='https://github.com/abigailmcgovern/paintedlife',
      packages=find_packages(),
      install_requires=get_install_required(),
      python_requires='>=3.6',
      license='LICENSE.md',
      classifiers=[
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3 :: Only',
                   ]
      )