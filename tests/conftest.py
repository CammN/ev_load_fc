"""Pytest configuration: add scripts/ to sys.path and register run_*.py modules
under their short names (e.g. 'run_training' → importable as 'training')."""
import sys
import importlib
import pathlib

SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Register each run_*.py script under its short alias so tests can do
# `import training` instead of `import run_training`.
_ALIASES = {
    "training": "run_training",
    "inference": "run_inference",
    "features": "run_features",
    "enrichment": "run_enrichment",
    "extraction": "run_extraction",
}

for _alias, _module_name in _ALIASES.items():
    if _alias not in sys.modules:
        try:
            sys.modules[_alias] = importlib.import_module(_module_name)
        except Exception:
            pass
