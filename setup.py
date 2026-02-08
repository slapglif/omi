"""
OMI - OpenClaw Memory Infrastructure
"""

from setuptools import setup, find_packages

setup(
    name="omi-openclaw",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    author="Hermes",
    author_email="Hermes@ai-smith.net",
    description="OpenClaw Memory Infrastructure - The seeking is the continuity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/slapglif/omi",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "ollama": ["ollama>=0.1.0"],
        "sqlite-vss": ["sqlite-vss>=1.1.0"],
    },
    entry_points={
        "console_scripts": [
            "omi=omi.cli:main",
        ],
    },
)
