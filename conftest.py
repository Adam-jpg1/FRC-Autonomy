"""Make ``frc_perception`` importable when pytest runs from the repo root.

pytest inserts the rootdir into ``sys.path`` for rootdir-level conftest
files, so this file existing is enough; the explicit insert below keeps it
working when pytest is invoked from elsewhere with an explicit path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
