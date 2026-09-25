"""
Exercise the standalone rights installer without third-party site packages.
"""

import runpy
import sys
from pathlib import Path

# %% isolated packaging

namespace = runpy.run_path(sys.argv[1])
namespace["LaboratoryAssetRights"]().install(Path(sys.argv[2]))
