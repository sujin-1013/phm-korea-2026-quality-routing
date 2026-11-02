"""Presentation tables as TNR-styled images:
  fig_table_results.png  - S10 results: method / macro-F1 / active params / Large-normalized params
  fig_table_spec.png     - S8 dataset & model spec
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import figstyle  # noqa

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "presentation_assets"
OUT.mkdir(parents=True, exist_ok=True)
NAVY = figstyle.NAVY; ORANGE_FILL = "#fde3d2"; GREY = "#cdd6df"


def styled_table(fname, cols, rows, figsize, highlight=None, col_widths=None, left_align_col0=True,
                 header_height_factor=1.0):
    fig, ax = plt.subplots(figsize=figsize, dpi=150); ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                   colWidths=col_widths)
    tab.auto_set_font_size(False); tab.set_fontsize(15.5); tab.scale(1, 2.25)
    ncol = len(cols)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GREY); cell.set_linewidth(0.65)
        if r == 0:  # header
            cell.set_facecolor(NAVY); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
            if header_height_factor != 1.0:
                cell.set_height(cell.get_height() * header_height_factor)
        elif highlight is not None and r == highlight:
            cell.set_facecolor(ORANGE_FILL); cell.get_text().set_fontweight("bold")
        if left_align_col0 and c == 0:
            cell.get_text().set_ha("left"); cell.PAD = 0.04
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight"); plt.close(fig)
    print("saved", fname)


# ---- S10 results ----
styled_table(
    "fig_table_results.png",
    ["Method", "Macro-F1", "Active params", "Parameter ratio\n(Large=1)"],
    [["Small", "0.553", "133K", "0.25"],
     ["Medium", "0.611", "331K", "0.63"],
     ["Large", "0.618", "524K", "1.00"],
     ["Quality gate", "0.615", "266K", "0.51"]],
    figsize=(7.2, 2.35), highlight=4, col_widths=[0.24, 0.21, 0.28, 0.27],
    header_height_factor=1.28)

# ---- S8 dataset & model spec ----
styled_table(
    "fig_table_spec.png",
    ["Property", "Value"],
    [["Dataset", "Paderborn University (PU) bearing"],
     ["Classes", "3  (Normal / Inner race / Outer race)"],
     ["Signal", "Vibration (accelerometer), 64 kHz"],
     ["Window", "2048 samples (32 ms at 64 kHz), non-overlapping"],
     ["Training", "Clean signals only"],
     ["Test", "AWGN, per-window mixed SNR (clean → −10 dB)"],
     ["Model tiers", "Small 133K / Medium 331K / Large 524K"],
     ["Quality estimator", "0.13K, supervised by injected SNR"],
     ["Evaluation", "Macro-F1; avg. active params as efficiency proxy"]],
    figsize=(9.4, 4.2), col_widths=[0.26, 0.74])
