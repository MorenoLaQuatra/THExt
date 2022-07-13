from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='thext',
    version='1.0',
    description='THExt - Transformer-based Highlights Extraction',
    py_modules=["DatasetPlus", "Highlighter", "RedundancyManager", "SentenceRankerPlus"],
    packages=find_packages(include=['thext', 'thext.*']),
    classifiers={
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    },
    long_description = long_description,
    long_description_content_type = "text/markdown",
    install_requires = [
        "nltk",
        "spacy",
        "mlxtend",
        "numpy",
        "sentence-transformers",
        "torch",
        "multiprocess",
        "py-rouge",
        "scispacy",
    ],
    extras_require = {
        "dev" : [
            "pytest>=3.7",
        ],
    },
    url="https://github.com/MorenoLaQuatra/THExt",
    author="Moreno La Quatra",
    author_email="moreno.laquatra@gmail.com",
)