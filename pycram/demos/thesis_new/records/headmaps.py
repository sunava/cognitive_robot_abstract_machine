import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# MPLBACKEND=Agg python headmaps.py
cut = pd.read_csv(SCRIPT_DIR / "cut_all_breads_results.csv", on_bad_lines="skip")
mix = pd.read_csv(SCRIPT_DIR / "mix_all_bowls_results.csv", on_bad_lines="skip")
wipe = pd.read_csv(SCRIPT_DIR / "wipe_all_spaces_results.csv", on_bad_lines="skip")

# Normalise environment name for wipe (world_name = 'map' -> 'apartment')
wipe["world_name"] = wipe["world_name"].replace("map", "apartment")

CUT_ROBOTS = [
    "pr2",
    "tiago",
    "rollin_justin",
    "armar7",
    "unitree_g1",
    "stretch",
    "hsrb",
]
MIX_ROBOTS = ["rollin_justin", "tiago", "armar7", "unitree_g1", "stretch", "hsrb"]
WIPE_ROBOTS = ["pr2", "tiago", "stretch", "hsrb"]
ROBOT_LABELS = {
    "pr2": "PR2",
    "tiago": "TIAGo",
    "rollin_justin": "Rollin' Justin",
    "armar7": "ARMAR-7",
    "unitree_g1": "Unitree G1",
    "stretch": "Stretch 3",
    "hsrb": "HSRB",
}
ENV_ORDER = ["apartment", "kitchen", "isr"]
ENV_LABELS = {"apartment": "Apt.", "kitchen": "Kitchen", "isr": "ISR"}


def make_matrix(df, robot_order, env_order):
    sr = df.groupby(["robot_name", "world_name"])["final_success"].mean()
    mat = pd.DataFrame(np.nan, index=robot_order, columns=env_order)
    for (robot, env), val in sr.items():
        if robot in mat.index and env in mat.columns:
            mat.loc[robot, env] = val
    return mat


cut_mat = make_matrix(cut, CUT_ROBOTS, ENV_ORDER)
mix_mat = make_matrix(mix, MIX_ROBOTS, ["apartment"])
wipe_mat = make_matrix(wipe, WIPE_ROBOTS, ["apartment"])

TEXT_PRIMARY = "#242421"
TEXT_SECONDARY = "#696862"
TEXT_TERTIARY = "#9a9890"

cmap = mcolors.LinearSegmentedColormap.from_list("oaat", ["#fce1dc", "#0f6446"], N=256)


def rate_to_color(rate):
    if np.isnan(rate):
        return "none"
    return cmap(float(rate))


def text_color(rate):
    if np.isnan(rate):
        return TEXT_TERTIARY
    return "white" if rate > 0.5 else TEXT_PRIMARY


def draw_heatmap_table(ax, mat, title):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.975,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="500",
        color=TEXT_PRIMARY,
    )

    left = 0.02
    right = 0.98
    top = 0.86
    bottom = 0.06
    header_h = 0.08
    row_gap = 0.014
    col_gap = 0.012
    label_w = 0.42
    cell_w = (right - left - label_w - col_gap * len(mat.columns)) / len(mat.columns)
    row_h = (top - bottom - header_h - row_gap * len(mat.index)) / len(mat.index)

    for j, env in enumerate(mat.columns):
        x = left + label_w + col_gap + j * (cell_w + col_gap)
        ax.text(
            x + cell_w / 2,
            top - header_h / 2,
            ENV_LABELS.get(env, env),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="400",
            color=TEXT_SECONDARY,
        )

    for i, robot in enumerate(mat.index):
        y = top - header_h - (i + 1) * row_h - i * row_gap
        ax.text(
            left + label_w - 0.018,
            y + row_h / 2,
            ROBOT_LABELS.get(robot, robot),
            ha="right",
            va="center",
            fontsize=11,
            fontweight="400",
            color=TEXT_SECONDARY,
        )

        for j, env in enumerate(mat.columns):
            rate = mat.loc[robot, env]
            x = left + label_w + col_gap + j * (cell_w + col_gap)
            if not np.isnan(rate):
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (x, y),
                        cell_w,
                        row_h,
                        boxstyle="round,pad=0.002,rounding_size=0.012",
                        linewidth=0,
                        facecolor=rate_to_color(rate),
                    )
                )

            label = "-" if np.isnan(rate) else f"{int(round(rate * 100))}%"
            ax.text(
                x + cell_w / 2,
                y + row_h / 2,
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="500",
                color=text_color(rate),
            )


fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"wspace": 0.24})
fig.patch.set_facecolor("white")

titles = ["Cutting", "Mixing", "Wiping"]
matrices = [cut_mat, mix_mat, wipe_mat]

for ax, mat, title in zip(axes, matrices, titles):
    draw_heatmap_table(ax, mat, title)

legend_ax = fig.add_axes([0.34, 0.035, 0.32, 0.06])
legend_ax.set_axis_off()
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.text(
    0.03, 0.5, "0%", ha="right", va="center", fontsize=11, color=TEXT_SECONDARY
)

bar_x = 0.06
bar_y = 0.32
bar_w = 0.42
bar_h = 0.22
steps = 12
for i in range(steps):
    x = bar_x + i * bar_w / steps
    legend_ax.add_patch(
        patches.Rectangle(
            (x, bar_y),
            bar_w / steps,
            bar_h,
            linewidth=0,
            facecolor=cmap(i / (steps - 1)),
        )
    )

legend_ax.text(
    bar_x + bar_w + 0.03,
    0.5,
    "100%",
    ha="left",
    va="center",
    fontsize=11,
    color=TEXT_SECONDARY,
)
legend_ax.text(
    0.68,
    0.5,
    "- not tested",
    ha="left",
    va="center",
    fontsize=11,
    color=TEXT_SECONDARY,
)

plt.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.17)
plt.savefig(SCRIPT_DIR / "oaat_results_heatmap.pdf", bbox_inches="tight", dpi=300)
plt.savefig(SCRIPT_DIR / "oaat_results_heatmap.png", bbox_inches="tight", dpi=300)
print("Saved: oaat_results_heatmap.pdf and oaat_results_heatmap.png")
