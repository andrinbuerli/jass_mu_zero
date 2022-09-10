import subprocess
from pathlib import Path

from setuptools import setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


cwd = Path(__file__).parent.resolve()
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)

subprocess.check_call(["pip", "install", "-e", "."], cwd=cwd / "extern" / "jass-kit-py")
subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-kit-cpp")
subprocess.check_call(["cmake", "."], cwd=cwd / "extern" / "jass-kit-cpp")
subprocess.check_call(["make", "install"], cwd=cwd / "extern" / "jass-kit-cpp")
subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass-ml-cpp")
subprocess.check_call(["pip", "install", "-e", "."], cwd=cwd / "extern" / "jass-ml-py")
subprocess.check_call(["pip", "install", "-e", "."], cwd=cwd / "extern" / "jass_gym")

setup(
    name="jass_mu_zero",
    version="1.0",
    description="Jass MuZero implementation",
    url="tbd",
    packages=["jass_mu_zero"],
    install_requires=["wheel"] + parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "sjmz = jass_mu_zero.__main__:main",
        ],
    },
)
