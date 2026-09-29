from setuptools import setup, find_packages

setup(
    name="lightagentx",
    version="0.1.0",
    description="A lightweight modular agentic AI framework built from scratch",
    author="Devansh",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
