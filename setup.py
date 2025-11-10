"""
Setup configuration for qn-ssa package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Read the long description from README
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qn-ssa",
    version="0.1.0",
    author="Mirco Tribastone",
    author_email="",
    description="Stochastic Simulation Algorithm for Queueing Networks using Gillespie's SSA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mirco-tribastone/qn-ssa",
    project_urls={
        "Bug Reports": "https://github.com/mirco-tribastone/qn-ssa/issues",
        "Source": "https://github.com/mirco-tribastone/qn-ssa",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.990",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="gillespie stochastic simulation queueing markov ssa ctmc",
)

