from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="teascoFT",
    version="0.1.0",
    author="TeasCo",
    description=(
        "Knowledge Distillation + Cascade Architecture for "
        "fast, specialized text classification."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "examples*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "embeddings": ["sentence-transformers>=2.6.0"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
