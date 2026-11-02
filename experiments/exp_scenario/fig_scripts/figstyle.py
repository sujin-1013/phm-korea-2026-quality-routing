"""Shared matplotlib style for ALL presentation figures — Times New Roman, unified.
Import this right after `import matplotlib.pyplot as plt` in each figure script."""
import matplotlib.pyplot as plt

FONT_AXIS = 19
FONT_MAJOR = 17
FONT_MINOR = 15

NAVY = "#08306B"
BLUE = "#2C7FB8"
SKY = "#9ECAE1"
ORANGE = "#D55E00"
GREY = "#9AA7B4"
GRID = "#E4EAF0"
SPINE = "#9AA7B4"
TEXT = "#333333"

LW_MAIN = 3.0
LW_SECONDARY = 2.6
LW_AUX = 1.6
MS_MAIN = 8
MS_SECONDARY = 7

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",       # math symbols match TNR look
    "axes.titlesize": FONT_AXIS,
    "axes.labelsize": FONT_AXIS,
    "xtick.labelsize": FONT_MAJOR,
    "ytick.labelsize": FONT_MAJOR,
    "legend.fontsize": FONT_MAJOR,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # embed real fonts in vector output
})
