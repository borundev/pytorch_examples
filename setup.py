from setuptools import setup,find_packages


packages=[]
packages+=find_packages()

setup(name='pytorch_examples',
      version='0.1',
      description='Examples using pytorch and pytorch lightning',
      url='http://github.com/borundev/pytorch_lightning_examples',
      author='Borun D Chowdhury',
      author_email='borundev@gmail.com',
      license='MIT',
      packages=packages,
      install_requires=['torch',
                        'pytorch_lightning',
                        'pytorch_lightning_bolts',
                        'wandb',
                        'torchvision',
                        'sklearn',
                        'matplotlib',
                        'torchsummary'
                        ],
      zip_safe=False)