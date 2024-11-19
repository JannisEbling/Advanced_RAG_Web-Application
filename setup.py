"""Setup file for the project."""
import os
import setuptools

# Constants
REPO_NAME = "template"
AUTHOR_USER_NAME = "JannisEbling"
AUTHOR_EMAIL = "jannis.ebling@outlook.de"
SRC_REPO = "src"
PYTHON_REQUIRES = ">=3.11"

# Read version from version.txt or set default
try:
    with open("version.txt", "r", encoding="utf-8") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    __version__ = "0.0.1"

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Template project - README.md not found"

# Read requirements
def get_requirements(filename="requirements.txt"):
    """Read requirements from file."""
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A template project structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Documentation": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}#readme",
        "Source Code": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    },
    package_dir={"": SRC_REPO},
    packages=setuptools.find_packages(where=SRC_REPO),
    python_requires=PYTHON_REQUIRES,
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    zip_safe=False,
)
