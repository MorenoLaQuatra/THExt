from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='THExt',
    version='0.0.1',
    description='THExt - Transformer-based Higlights Extraction',
    py_modules=["Dataset", "DatasetPlus", "Highlighter", "RedundancyManager", "SentenceRanker", "SentenceRankerPlus"],
    #package_dir={'':'THExt'},
    packages=find_packages(include=['THExt', 'THExt.*']),
    classifiers={
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved ::  GNU GPLv3",
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
    url="",
    author="Moreno La Quatra",
    author_email="moreno.laquatra@gmail.com",
)