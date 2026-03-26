"""Pytest configuration: add scripts/ to sys.path so script modules are importable."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))



