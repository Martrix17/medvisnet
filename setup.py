from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).parent
long_description = (
    (this_dir / "README.md").read_text() if (this_dir / "README.md").exists() else ""
)

setup(
    name="medvisnet",
    version="0.1.0",
    author="Martrix17",
    description="Medical computer vision project with DL and DevOps.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[],
)
