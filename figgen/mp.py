"""Matplotlib helpers: LaTeX-like typography (Computer Modern), exact text width, panel letters, annotated heatmaps."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "overleaf", "build")
TEXTW = 6.5                                      # inches: the text width of the paper (1 in margins on letter paper)

C = dict(blue="#2F5D8A", green="#3E7B45", gold="#A9822A", orange="#C0642B", purple="#7A4E9C", teal="#2A8C8C", red="#B03A48", gray="#6B6B6B",
         light="#E4E4E4", ink="#222222", sky="#8DB3D6", pink="#D58A9A")
FAMC = {"ft": C["gray"], "ewc": C["orange"], "replay": C["red"], "lw": C["blue"], "lwf": C["sky"], "ogd": C["green"], "ad": C["purple"]}
MARK = {"ft": "o", "ewc": "^", "replay": "s", "lw": "o", "lwf": "o", "ogd": "D", "ad": "D"}
FAMN = {"ft": "fine-tuning", "ewc": "EWC", "replay": "replay", "lw": "layerwise, frozen", "lwf": "layerwise, free", "ogd": "gradient subspace", "ad": "task adapters"}
CMAP = plt.get_cmap("viridis")
DIV = LinearSegmentedColormap.from_list("div", ["#2F5D8A", "#8DB3D6", "#F0F0F0", "#E9B478", "#C0642B"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#F7F3EC", "#E9B478", "#C0642B", "#6E281E"])
BOLD = fm.FontProperties(fname=fm.findfont("cmb10"))


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.titlesize": 8.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.2, "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": "#E4E4E4", "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.3,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "savefig.bbox": None, "axes.axisbelow": True, "errorbar.capsize": 1.5,
    })


setup()


def new(h, nrows=1, ncols=3, w=TEXTW, widths=None, heights=None, hspace=0.62, wspace=0.55, left=0.08, right=0.985, top=0.88, bottom=0.19, **_ignored):
    """Figure of exact text width w (in) and height h (in) with explicit margins (fractions) so nothing moves between runs."""
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(nrows, ncols, left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace, width_ratios=widths, height_ratios=heights)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)])
    return fig, (axes[0] if nrows == 1 else axes)


def nolog_minor(ax):
    from matplotlib.ticker import NullFormatter
    for a in (ax.xaxis, ax.yaxis): a.set_minor_formatter(NullFormatter())


def panel(ax, letter, title=None, pad=4):
    """Bold panel letter at the top-left of an axes, followed by an optional title (both outside the plot area)."""
    t = ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-2, pad), textcoords="offset points", ha="left", va="bottom", fontproperties=BOLD, fontsize=8.5)
    if title:
        ax.annotate(title, xy=(1, 0), xycoords=t, xytext=(3, 0), textcoords="offset points", ha="left", va="bottom", fontsize=8.5)


def grid(ax, axis="y"):
    ax.grid(axis="both" if axis == "xy" else axis, alpha=1.0, zorder=0)


def legend(ax, loc="best", ncol=1, **kw):
    kw.setdefault("fontsize", 6.8)
    return ax.legend(loc=loc, ncol=ncol, **kw)


def label_bars(ax, xs, hs, fmt="{:.2f}", dy=0.0, fs=6, **kw):
    for x, h in zip(xs, hs):
        ax.text(x, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=fs, **kw)


def heat(ax, Z, xe, ye, vmin=None, vmax=None, cmap=CMAP, annotate=None, fs=6, edge="white", lw=0.4):
    """pcolormesh heatmap with optional per-cell annotation (white/ink text chosen by luminance). Z[i, j]: row i spans ye[i]..ye[i+1]."""
    Z = np.ma.masked_invalid(np.asarray(Z, float)); vmin = float(Z.min()) if vmin is None else vmin; vmax = float(Z.max()) if vmax is None else vmax
    m = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=edge, linewidth=lw, rasterized=False)
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z.mask[i, j]: continue
                rgb = cmap((Z[i, j] - vmin) / (vmax - vmin + 1e-12)); lum = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
                ax.text((xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2, annotate(Z[i, j]), ha="center", va="center", fontsize=fs, color="white" if lum < 0.55 else C["ink"])
    for s in ax.spines.values(): s.set_visible(False)
    return m


def cbar(fig, mappable, ax, label=None, ticks=None, width=0.045, pad=0.035, **_ignored):
    """Colorbar in an inset axes to the right of ax (does not change the layout of the other panels)."""
    cax = ax.inset_axes([1 + pad, 0, width, 1]); cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.5, width=0.4, length=2)
    if label: cb.set_label(label, fontsize=7)
    return cb


def fmt_pm(x, nd=2): return f"{x:.{nd}f}"


def save(fig, name, tight=False):
    from matplotlib.text import Text
    fig.canvas.draw(); bb = fig.bbox
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        e = t.get_window_extent()
        if e.x0 < bb.x0 - 1 or e.x1 > bb.x1 + 1 or e.y0 < bb.y0 - 1 or e.y1 > bb.y1 + 1: print(f"  [clip] {name}: {t.get_text()[:40]!r}")
    for a in fig.axes:
        if a.get_xscale() == "log" or a.get_yscale() == "log": nolog_minor(a)
    os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, name + ".pdf"), **({"bbox_inches": "tight", "pad_inches": 0.04} if tight else {}), metadata={"Creator": None, "Producer": None, "CreationDate": None, "Title": None, "Author": None, "Subject": None, "Keywords": None}); plt.close(fig)
