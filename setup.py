from setuptools import setup

setup(name='fitness_mcmc',
      version='1.0',
      description='A genotypic data anaylsis package to infer genotype fitnesses',
      url='http://github.com/phys201/fitness_mcmc',
      author='Eliot, Pavel, Lidiya',
      author_email='eliotfenton@g.harvard.edu,pzhelnin@g.harvard.edu,lidiya_ahmed@g.harvard.edu ',
      license='GPLv3',
      packages=['fitness_mcmc'],
      install_requires=['pymc3','numpy','arviz','matplotlib','seaborn'])
