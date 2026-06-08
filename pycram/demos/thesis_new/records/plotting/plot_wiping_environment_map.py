#!/usr/bin/env python3
from __future__ import annotations

from plot_environment_maps import main


if __name__ == "__main__":
    main(
        default_input_name="csv/raw_wiping_merged.csv",
        default_output_dir_name="environment_maps/wiping",
        default_environment_name="",
        task_label="Wiping",
    )
