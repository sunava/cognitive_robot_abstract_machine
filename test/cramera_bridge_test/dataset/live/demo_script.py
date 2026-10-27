"""
A demo file the live command runs; it records the arguments it was given.
"""

import os
import sys
from pathlib import Path

Path(os.environ["CRAMERA_TEST_DEMO_OUTPUT"]).write_text(
    " ".join(sys.argv[1:] + [os.environ["CORAPLEX_VISUALIZATION"]]), encoding="utf-8"
)
