import setuptools

with open('README.md') as file_object:
    long_description = file_object.read()

with open('requirements.txt') as file_object:
    install_requires = file_object.read()

setuptools.setup(
    name='Dialogue',
    version='0.0.1',
    description='Train & Inference Machines That Talk Soo Much',
    long_description=long_description,
    long_description_conttype='text/markdown',
    packages=setuptools.find_packages(exclude=['notebooks']),
    install_requires=install_requires,
    zip_safe=False,
)
