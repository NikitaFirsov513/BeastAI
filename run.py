"""Run BeastAI apps, experiments and tests from a stable project root."""
from pathlib import Path
import os
import runpy
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from project_paths import command_path, commands, configure_imports

def main():
    configure_imports()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("BeastAI: python run.py COMMAND [arguments]")
        print("  waid serve --dataset sherbrooke   Start the local viewer")
        print("  screen_overlay                   Start the Windows overlay")
        print("  test                             Run existing regression tests")
        print("  --list                           List available scripts")
        return 0
    if args[0] == "--list":
        for name, path in sorted(commands().items(), key=lambda item: (str(item[1].parent), item[0])):
            print(f"{name:38} {path.relative_to(ROOT).as_posix()}")
        return 0
    os.chdir(ROOT)
    if args[0] == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        target = command_path(args[0])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    sys.argv = [str(target), *args[1:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
