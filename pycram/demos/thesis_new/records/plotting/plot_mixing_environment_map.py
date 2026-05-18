#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PYCRAM_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PYCRAM_ROOT.parent
for source_root in [
    PYCRAM_ROOT,
    PYCRAM_ROOT / "src",
    WORKSPACE_ROOT / "krrood" / "src",
    WORKSPACE_ROOT / "semantic_digital_twin" / "src",
]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

PLOT_MODULE_PATH = Path(__file__).with_name("plot_cutting_environment_map.py")
PLOT_MODULE_SPEC = importlib.util.spec_from_file_location(
    "thesis_new_environment_map_plot", PLOT_MODULE_PATH
)
plot_module = importlib.util.module_from_spec(PLOT_MODULE_SPEC)
assert PLOT_MODULE_SPEC.loader is not None
PLOT_MODULE_SPEC.loader.exec_module(plot_module)


if __name__ == "__main__":
    plot_module.main(
        default_input_name="mix_all_bowls_results.csv",
        default_output_dir_name="mixing_environment_maps",
        task_label="Mixing",
    )
