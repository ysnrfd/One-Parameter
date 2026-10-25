"""Import bootstrap for the test suite.

``python -m unittest discover -s tests`` from inside the package directory, and
``python main.py selftest`` from its parent, both need the project root on
``sys.path`` before ``single_parameter_observatory`` can be imported.  Doing it
here keeps it out of every individual test module.
"""

from __future__ import annotations

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_TESTS_DIR)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

for _path in (_PROJECT_ROOT, _PACKAGE_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
