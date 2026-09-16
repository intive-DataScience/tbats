#!/usr/bin/env python
"""Fail-closed validation for the exact wheel and sdist of a tagged release."""

import argparse
import hashlib
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


PROJECT_NAME = "tbats"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(contents, artifact, metadata_path):
    metadata = BytesParser().parsebytes(contents)
    if metadata["Name"] != PROJECT_NAME:
        raise ValueError(f"{artifact.name} {metadata_path} has Name {metadata['Name']!r}, expected {PROJECT_NAME!r}")
    return metadata


def validate_wheel(path, expected_version):
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = sorted(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        if len(metadata_paths) != 1:
            raise ValueError(f"{path.name} must contain exactly one wheel METADATA file, found {metadata_paths!r}")
        metadata = parse_metadata(wheel.read(metadata_paths[0]), path, metadata_paths[0])
    if metadata["Version"] != expected_version:
        raise ValueError(f"{path.name} has Version {metadata['Version']!r}, expected {expected_version!r}")


def validate_sdist(path, expected_version):
    metadata_path = f"{PROJECT_NAME}-{expected_version}/PKG-INFO"
    with tarfile.open(path, "r:gz") as sdist:
        members = [member for member in sdist.getmembers() if member.name == metadata_path]
        if len(members) != 1:
            raise ValueError(f"{path.name} must contain exactly one {metadata_path!r}, found {len(members)}")
        metadata_file = sdist.extractfile(members[0])
        if metadata_file is None:
            raise ValueError(f"{path.name} has no readable {metadata_path}")
        metadata = parse_metadata(metadata_file.read(), path, metadata_path)
    if metadata["Version"] != expected_version:
        raise ValueError(f"{path.name} has Version {metadata['Version']!r}, expected {expected_version!r}")


def validate(dist_path, expected_version):
    expected = {
        f"{PROJECT_NAME}-{expected_version}-py3-none-any.whl": validate_wheel,
        f"{PROJECT_NAME}-{expected_version}.tar.gz": validate_sdist,
    }
    if not dist_path.is_dir():
        raise ValueError(f"dist path is not a directory: {dist_path}")
    actual = {path.name: path for path in dist_path.iterdir()}
    for name, path in sorted(actual.items()):
        if path.is_symlink():
            raise ValueError(f"dist entry must not be a symlink: {name}")
    if set(actual) != set(expected):
        raise ValueError(f"{dist_path} must contain exactly {sorted(expected)!r}, found {sorted(actual)!r}")
    for name, validator in expected.items():
        if actual[name].is_symlink():
            raise ValueError(f"expected artifact must not be a symlink: {actual[name]}")
        if not actual[name].is_file():
            raise ValueError(f"expected artifact is not a file: {actual[name]}")
        validator(actual[name], expected_version)
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_path", type=Path)
    parser.add_argument("expected_version")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    artifacts = validate(args.dist_path, args.expected_version)
    wheel_name = f"{PROJECT_NAME}-{args.expected_version}-py3-none-any.whl"
    sdist_name = f"{PROJECT_NAME}-{args.expected_version}.tar.gz"
    values = {
        "version": args.expected_version,
        "wheel_filename": wheel_name,
        "wheel_sha256": sha256(artifacts[wheel_name]),
        "sdist_filename": sdist_name,
        "sdist_sha256": sha256(artifacts[sdist_name]),
    }
    for key in ("wheel_filename", "wheel_sha256", "sdist_filename", "sdist_sha256"):
        print(f"{key}={values[key]}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
