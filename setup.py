from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

import versioneer

PACKAGE_NAME = "scmdata"
DESCRIPTION = "Data handling for simple climate model data"
KEYWORDS = ["data", "simple climate model", "climate", "scm"]

AUTHORS = [
    ("Jared Lewis", "jared.lewis@climate-energy-college.org"),
    ("Zeb Nicholls", "zebedee.nicholls@climate-energy-college.org"),
]
EMAIL = "jared.lewis@climate-energy-college.org"
URL = "https://github.com/openscm/scmdata"
PROJECT_URLS = {
    "Bug Reports": "https://github.com/openscm/scmdata/issues",
    "Documentation": "https://scmdata.readthedocs.io/en/latest",
    "Source": "https://github.com/openscm/scmdata",
}
LICENSE = "3-Clause BSD License"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

REQUIREMENTS = [
    "cftime",
    "numpy",
    "openscm-units>=0.2,<0.3",
    "pandas>=1.0.4,<1.2",
    "pint-pandas==0.1.0",
    "python-dateutil",
    "tqdm",
    "xarray",
]
REQUIREMENTS_OPTIONAL = [
    "netCDF4",
    "scipy",
    "xlrd<=1.2.0",  # support reading excel
]
REQUIREMENTS_PLOTTING = ["matplotlib", "seaborn"]
REQUIREMENTS_NOTEBOOKS = (
    ["notebook", "ipywidgets"] + REQUIREMENTS_PLOTTING + REQUIREMENTS_OPTIONAL
)
REQUIREMENTS_TESTS = (
    ["codecov", "nbval", "pytest>=5.0.0, <6", "pytest-benchmark", "pytest-cov"]
    + REQUIREMENTS_PLOTTING
    + REQUIREMENTS_OPTIONAL
)
REQUIREMENTS_DOCS = ["sphinx>2.1", "sphinx_rtd_theme"]
REQUIREMENTS_DEPLOY = ["twine>=1.11.0", "setuptools>=41.2", "wheel>=0.31.0"]

REQUIREMENTS_DEV = [
    *[
        "bandit",
        "black==19.10b0",
        "black-nb",
        "flake8",
        "isort>=5",
        "nbdime",
        "notebook",
        "ipywidgets",
        "pydocstyle",
        "pylint",
    ],
    *REQUIREMENTS_OPTIONAL,
    *REQUIREMENTS_PLOTTING,
    *REQUIREMENTS_TESTS,
    *REQUIREMENTS_DOCS,
    *REQUIREMENTS_DEPLOY,
]

REQUIREMENTS_EXTRAS = {
    "notebooks": REQUIREMENTS_NOTEBOOKS,
    "optional": REQUIREMENTS_OPTIONAL,
    "plotting": REQUIREMENTS_PLOTTING,
    "docs": REQUIREMENTS_DOCS,
    "tests": REQUIREMENTS_TESTS,
    "deploy": REQUIREMENTS_DEPLOY,
    "dev": REQUIREMENTS_DEV,
}


SOURCE_DIR = "src"

PACKAGES = find_packages(SOURCE_DIR)  # no exclude as only searching in `src`
PACKAGE_DIR = {"": SOURCE_DIR}
PACKAGE_DATA = {"scmdata": ["data/*.csv"]}


README = "README.rst"

# Get the long description from the README file
with open(README, "r") as f:
    README_LINES = ["scmdata", "=======", ""]
    add_line = False
    for line in f:
        if line.strip() == ".. sec-begin-long-description":
            add_line = True
        elif line.strip() == ".. sec-end-long-description":
            break
        elif add_line:
            README_LINES.append(line.strip())

if len(README_LINES) < 3:
    raise RuntimeError("Insufficient description given")


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
    long_description="\n".join(README_LINES),
    long_description_content_type="text/x-rst",
    author=", ".join([author[0] for author in AUTHORS]),
    author_email=", ".join([author[1] for author in AUTHORS]),
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require=REQUIREMENTS_EXTRAS,
    cmdclass=cmdclass,
)
