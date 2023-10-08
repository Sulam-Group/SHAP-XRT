import os

from setuptools import find_packages, setup

PACKAGE_NAME = "hshap_shaplit"
AUTHOR = "Anonymous authors"
AUTHOR_EMAIL = "an.author@email.com"

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "hshap", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]
        print(VERSION)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
)
