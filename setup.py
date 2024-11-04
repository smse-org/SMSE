from setuptools import setup, find_packages

setup(
    name="Semantic Multi-Modal Search Engine",
    version="0.1-dev",
    author="Ahmed Saed",
    author_email="mail@ahmedsaed.me",
    description="A search engine that works across multiple modalities using semantic embeddings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmedsaed/smse",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
