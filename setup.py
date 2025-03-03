import os
import subprocess
from pathlib import Path

from setuptools import setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


try:
    if 'SKIP_EXTERN' not in os.environ:
        cwd = Path(__file__).parent.resolve()
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)

        os.system("export SKIP_EXTERN=1")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass_gym")
        os.system("unset SKIP_EXTERN")

        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-kit-py")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-kit-cpp")
        subprocess.check_call(["cmake", "."], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-kit-cpp")
        subprocess.check_call(["make", "install"], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-kit-cpp")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-ml-cpp")
        subprocess.check_call(["pip", "install", "."], cwd=cwd / "extern" / "jass_gym" / "extern" / "jass-ml-py")

except Exception as e:
    print("Failed to install extern modules...", e)

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
