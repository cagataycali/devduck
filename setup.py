#!/usr/bin/env python3
"""🦆 DevDuck setup - Extreme minimalist agent package"""
from setuptools import setup
import os


# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="devduck",
    version="0.1.0",
    description="🦆 Extreme minimalist self-adapting AI agent - one file, self-healing, runtime dependencies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="maxs",
    author_email="hey@devduck.dev",
    url="https://github.com/cagataycali/devduck",
    project_urls={
        "Bug Tracker": "https://github.com/cagataycali/devduck/issues",
        "Source Code": "https://github.com/cagataycali/devduck",
        "Documentation": "https://github.com/cagataycali/devduck#readme",
    },
    packages=["devduck", "devduck.tools"],
    install_requires=[
        "strands-agents",
        "strands-agents[ollama]",
        "strands-agents[openai]",
        "strands-agents[anthropic]",
        "strands-agents-tools",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "devduck=devduck:cli",
        ],
    },
    keywords=[
        "ai",
        "agent",
        "minimalist",
        "self-healing",
        "automation",
        "ollama",
        "strands",
    ],
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
    ],
)
