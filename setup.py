#!/usr/bin/env python3
"""
Setup script for Wikipedia Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wikipedia-agent",
    version="1.0.0",
    author="Wikipedia Agent",
    description="A simple command line agent to answer natural language questions using Wikipedia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["wikipedia_agent"],
    install_requires=requirements,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "wikipedia-agent=wikipedia_agent:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)