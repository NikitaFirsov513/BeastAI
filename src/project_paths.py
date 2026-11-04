"""Stable project locations shared by apps and experiment scripts."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPT_DIRS = tuple(sorted(p for p in (PROJECT_ROOT / "scripts").iterdir() if p.is_dir() and not p.name.startswith((".", "_"))))
CODE_DIRS = (SOURCE_DIR, *SCRIPT_DIRS, PROJECT_ROOT / "tests")

def configure_imports():
    for folder in reversed(CODE_DIRS):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))

def commands():
    return {p.stem: p for folder in CODE_DIRS for p in sorted(folder.glob("*.py")) if p.name != "project_paths.py"}

def command_path(name):
    key = name[:-3] if name.endswith(".py") else name
    try:
        return commands()[key]
    except KeyError:
        raise ValueError(f"Unknown command: {name}. Run python run.py --list.") from None
