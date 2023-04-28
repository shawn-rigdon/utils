#!/usr/bin/env python

from distutils.core import setup

setup(name='Distutils',
      version='0.1.0',
      description='Python utils modules for the mundain',
      author='Shawn Rigdon',
      author_email='shawn.patrick.rigdon@gmail.com',
      url='https://github.com/shawn-rigdon/utils',
      packages=['srutils'],
      install_requires=[
          'opencv-python',
          'pyautogui',
          'scipy',
          'scikit-image',
          'numpy',
          ],
     )
