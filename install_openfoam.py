#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


OPENFOAM_BASHRC = Path("/opt/openfoam14/etc/bashrc")


def run(command):
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def check_system():
    if platform.system() != "Linux":
        raise RuntimeError("Run this script inside Ubuntu. On Windows, open an Ubuntu WSL terminal first.")
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    if release.get("ID") != "ubuntu":
        raise RuntimeError("This installer supports Ubuntu 22.04, 24.04 and 26.04 only.")
    if release.get("VERSION_ID") not in {"22.04", "24.04", "26.04"}:
        raise RuntimeError("Use Ubuntu 22.04, 24.04 or 26.04 LTS.")
    architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    if architecture not in {"amd64", "arm64"}:
        raise RuntimeError("OpenFOAM packages require an amd64 or arm64 Ubuntu installation.")
    return release["VERSION_CODENAME"], architecture


def verify_installation():
    if not OPENFOAM_BASHRC.is_file():
        raise RuntimeError("OpenFOAM 14 was not found at /opt/openfoam14.")
    run(["bash", "-c", 'source /opt/openfoam14/etc/bashrc && foamRun -help && blockMesh -help && checkMesh -help && decomposePar -help && mpirun --version'])


def install():
    codename, architecture = check_system()
    if os.geteuid() == 0:
        raise RuntimeError("Run python3 install_openfoam.py as your normal user, without sudo.")
    if shutil.which("sudo") is None:
        raise RuntimeError("sudo is required to install Ubuntu packages.")
    print("Installing OpenFOAM Foundation 14 from the official repository.", flush=True)
    print("Ubuntu will ask for your sudo password. Typing it does not display characters.", flush=True)
    print("The download may take several minutes. You can rerun this script if interrupted.", flush=True)
    run(["sudo", "-v"])
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "ca-certificates", "curl", "software-properties-common"])
    run(["sudo", "add-apt-repository", "-y", "universe"])
    with tempfile.TemporaryDirectory(prefix="openfoam-install-") as temporary:
        key = Path(temporary) / "openfoam.asc"
        run(["curl", "--fail", "--show-error", "--location", "--retry", "3", "https://dl.openfoam.org/gpg.key", "--output", str(key)])
        run(["sudo", "install", "-m", "644", str(key), "/usr/share/keyrings/openfoam-foundation.asc"])
        source = Path(temporary) / "openfoam-foundation.list"
        source.write_text(f"deb [arch={architecture} signed-by=/usr/share/keyrings/openfoam-foundation.asc] http://dl.openfoam.org/ubuntu {codename} main\n")
        run(["sudo", "install", "-m", "644", str(source), "/etc/apt/sources.list.d/openfoam-foundation.list"])
    run(["sudo", "apt-get", "update"])
    run(["sudo", "apt-get", "install", "-y", "openfoam14", "openmpi-bin", "gmsh", "ffmpeg", "python3-matplotlib", "python3-numpy", "python3-paraview"])
    verify_installation()
    bashrc = Path.home() / ".bashrc"
    activation = "[ ! -f /opt/openfoam14/etc/bashrc ] || source /opt/openfoam14/etc/bashrc"
    contents = ""
    if bashrc.exists():
        contents = bashrc.read_text()
    if activation not in contents:
        with bashrc.open("a") as stream:
            stream.write("\n# OpenFOAM Foundation 14\n" + activation + "\n")
    print("\nOpenFOAM 14 is ready. New Bash terminals will load it automatically.")
    print("For this terminal, run: source /opt/openfoam14/etc/bashrc")


def main():
    parser = argparse.ArgumentParser(description="Install OpenFOAM Foundation 14 on Ubuntu or Ubuntu under WSL.")
    parser.add_argument("--check", action="store_true", help="Check an existing installation without changing the system.")
    arguments = parser.parse_args()
    try:
        if arguments.check:
            check_system()
            verify_installation()
            print("OpenFOAM 14 verification passed.")
        else:
            install()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"\nInstallation or verification failed: {error}", file=sys.stderr)
        print("Fix the reported error and run the same command again.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume installation.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
