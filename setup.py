from typing import List
from setuptools import find_packages,setup


def get_requirements(filename:str) -> List[str]:
    """
        This function returns a list of requirements
    """
    HYPHEN_E_DOT = "-e ."
    requirements = []
    with open(filename) as file:
        requirements.extend(file.readlines())
        requirements = [item.replace("\n"," ") for item in requirements]
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='EndToEndMLProject',
    version='0.0.1',
    description='Machine Learning End-to-End Project',
    author='Thanseefpp',
    author_email='thanseefpp@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    )