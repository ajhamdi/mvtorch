from setuptools import find_packages # ,setup
from distutils.core import setup


setup(
  name = 'mvtorch',         # How you named your package folder (MyLib)
  # packages=['mvtorch', 'mvtorch.models'],   # Chose the same as "name"
  version = '0.1.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A modular Pytroch library for multi-view research on 3D understanding and 3D generation.',   # Give a short description about your library
  author = 'Abdullah Hamdi',                   # Type in your name
  author_email = 'abdullah.hamdi@kaust.edu.sa',      # Type in your E-Mail
  packages=find_packages(include=("mvtorch*")),
  url = 'https://github.com/ajhamdi/mvtorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ajhamdi/mvtorch/archive/v0.1.tar.gz',    # I explain this later on
  keywords = ['pytorch', 'multi-view', '3d understanding',"nerfs"],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy', 'pandas', "trimesh", "imageio", "einops", "scipy", "matplotlib", "ptflops", "tensorboard", "h5py", "metric-learn", "timm",
      ],
  dependency_links=["git+https://github.com/openai/CLIP.git"],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',

  ],
)
