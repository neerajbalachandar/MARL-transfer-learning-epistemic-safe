"""Run the lightweight test functions without requiring pytest."""

import importlib.util
import sys
import tempfile
from pathlib import Path


TEST_FILE = Path(__file__).with_name("test_transfer.py")
sys.path.insert(0, str(TEST_FILE.parents[1]))
spec = importlib.util.spec_from_file_location("test_transfer", TEST_FILE)
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)

for name in sorted(value for value in dir(tests) if value.startswith("test_")):
    test = getattr(tests, name)
    if name == "test_plotting_uses_actual_episode_records":
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    else:
        test()
    print(f"PASS {name}")
