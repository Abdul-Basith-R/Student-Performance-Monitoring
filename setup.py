from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path, mode='r', encoding='utf-16') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='STUDENT PERFORMANCE MONITORING SYSTEM',
    packages=find_packages(),
    version='0.1.0',
    description='Monitoring System',
    author='Abdul-Basith-R',
    license='MIT',
    install_requires=get_requirements('requirements.txt'))
print(get_requirements('requirements.txt'))