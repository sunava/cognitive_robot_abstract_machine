"""
Keep an isolated launcher child alive until it receives a shutdown signal.
"""

import os
from pathlib import Path
import signal
import sys

# %% process lifetime
if __name__ == "__main__":
    if "--password-file" in sys.argv:
        if os.environ.get("CRAMERA_LAUNCHER_TEST_GATEWAY_FAIL") == "1":
            sys.exit(3)
        Path(os.environ["CRAMERA_LAUNCHER_TEST_GATEWAY_READY"]).touch()
    while True:
        signal.pause()
