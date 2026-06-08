#!/usr/bin/env python3
from __future__ import annotations

from plot_environment_maps import main


if __name__ == "__main__":
    main(
        default_input_name="csv/mixing_reuslt.csv",
        default_output_dir_name="environment_maps/mixing",
        task_label="Mixing",
    )
