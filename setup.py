from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='promptimizer',
    version='0.3.0',
    description='Automatic Prompt Optimization Framework',
    author='Aleksander Allen',
    author_email='aleks.allen@engineer.com',
    packages=find_packages(),
    install_requires=requirements,  # Include dependencies from requirements.txt
    # Additional metadata about your package
    url='https://github.com/SvetimFM/Promptimizer',
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache 2.0',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    # If your package includes data files inside
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
    },
)
