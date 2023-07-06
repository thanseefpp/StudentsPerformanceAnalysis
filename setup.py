import os
from setuptools import find_packages, setup

REPO_DIR = os.path.dirname(os.path.realpath(__file__))

def getVersion():
  """
  Get version from local file.
  """
  with open(os.path.join(REPO_DIR, "VERSION"), "r") as versionFile:
    return versionFile.read().strip()

def parse_file(requirementFile):
  try:
    return [
      line.strip()
      for line in open(requirementFile).readlines()
      if not line.startswith("-")
    ]
  except IOError:
    return []

def findRequirements():
  """
  Read the requirements.txt file and parse into requirements for setup's
  install_requirements option.
  """
  requirementsPath = os.path.join(REPO_DIR, "requirements.txt")
  requirements = parse_file(requirementsPath)
  return requirements

if __name__ == "__main__":

  requirements = findRequirements()
  
  setup(
    name='EndToEndMLProject',
    version=getVersion(),
    description='Machine Learning End-to-End Project',
    author='Thanseefpp',
    author_email='thanseefpp@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
    )