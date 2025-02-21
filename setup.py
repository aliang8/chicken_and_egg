from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Read README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chicken_and_egg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Anthony Liang",
    author_email="aliang80@usc.edu",
    description="A reinforcement learning package for exploring exploration strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliang80/chicken_and_egg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
