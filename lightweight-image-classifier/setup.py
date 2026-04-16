"""
Setup configuration for lightweight-image-classifier.

This file allows the package to be installed via:
    pip install -e .
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="lightweight-image-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A ready-to-use PyTorch lightweight image classification framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lightweight-image-classifier",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lic-train=scripts.train:main",
            "lic-validate=scripts.validate:main",
            "lic-inference=scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)