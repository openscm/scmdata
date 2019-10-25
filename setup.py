import versioneer

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


PACKAGE_NAME = "scmdata"
AUTHOR = "Jared Lewis"
EMAIL = "jared.lewis@climate-energy-college.org"
URL = "https://github.com/lewisjared/scmdata"

DESCRIPTION = (
    "Simple data handling for Simple Climate Model data"
)
README = "README.rst"

SOURCE_DIR = "src"

REQUIREMENTS = ["numpy", "scipy"]
REQUIREMENTS_TESTS = ["codecov", "pytest-cov", "pytest>=4.0"]
REQUIREMENTS_DOCS = ["sphinx>=1.4,<2.1", "sphinx_rtd_theme"]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=38.6.0", "wheel>=0.31.0"]

requirements_dev = [
    *["flake8"],
    *REQUIREMENTS_TESTS,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_DEPLOY,
]

requirements_extras = {
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": requirements_dev,
}

with open(README, "r") as readme_file:
    README_TEXT = readme_file.read()


class ScmData(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": ScmData})

setup(
    name=PACKAGE_NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=README_TEXT,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license="3-Clause BSD License",
    classifiers=[  # full list at https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=["data", "simple climate model", "climate", "scm"],
    packages=find_packages(SOURCE_DIR),  # no exclude as only searching in `src`
    package_dir={"": SOURCE_DIR},
    # next line is only required if you have data files that have to be included
    # e.g. csvs which define certain conventions etc.
    # include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=requirements_extras,
    cmdclass=cmdclass,
)
