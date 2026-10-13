"""Minimal publication-style plotting layer that emits plain TikZ (no pgfplots): coordinates are mapped to centimetres in Python."""
import math
import numpy as np

PALETTE = {"blue": "2F5D8A", "green": "3E7B45", "gold": "A9822A", "orange": "C0642B", "purple": "7A4E9C", "teal": "2A8C8C", "red": "B03A48", "gray": "6B6B6B", "light": "E4E4E4", "ink": "222222", "sky": "8DB3D6"}
STYLE_TEX = "\n".join(f"\\definecolor{{c{k.capitalize()}}}{{HTML}}{{{v}}}" for k, v in PALETTE.items())
VIRIDIS = [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88), (253, 231, 37)]
DIVERGE = [(47, 93, 138), (141, 179, 214), (240, 240, 240), (233, 180, 120), (192, 100, 43)]


def cmap(t, table=VIRIDIS):
    t = min(max(float(t), 0.0), 1.0) * (len(table) - 1); i = min(int(t), len(table) - 2); f = t - i
    return tuple(int(round(table[i][k] * (1 - f) + table[i + 1][k] * f)) for k in range(3))


def rgb(c): return f"{{rgb,255:red,{c[0]};green,{c[1]};blue,{c[2]}}}"


def col(c):
    return f"c{c.capitalize()}" if c in PALETTE else c


def fmt(v, nd=None):
    if nd is None:
        s = f"{v:.6g}"
    else:
        s = f"{v:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


class Fig:
    def __init__(self, width=16.0, height=6.0):
        self.w, self.h = width, height; self.L = []
    def raw(self, s): self.L.append(s)
    def rect(self, x0, y0, x1, y1, draw="cGray", fill=None, lw=0.5, r=0, opacity=1):
        opts = [f"draw={draw}" if draw else "draw=none", f"line width={lw}pt"]
        if fill: opts.append(f"fill={fill}")
        if r: opts.append(f"rounded corners={r}pt")
        if opacity != 1: opts.append(f"opacity={opacity}")
        self.raw(f"\\path[{','.join(opts)}] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
    def text(self, x, y, s, anchor="center", size="scriptsize", color="cInk", rot=0, bold=False, extra=""):
        f = f"\\{size}" + ("\\bfseries" if bold else "")
        self.raw(f"\\node[anchor={anchor},font={f},text={color},inner sep=1pt{',rotate=' + str(rot) if rot else ''}{(',' + extra) if extra else ''}] at ({x:.3f},{y:.3f}) {{{s}}};")
    def line(self, pts, color="cInk", lw=0.8, dash=None, opacity=1, arrow=None):
        o = [f"draw={color}", f"line width={lw}pt"]
        if dash:
            a, b = dash.replace("pt", "").split(" on ")
            o.append(f"dash pattern=on {a}pt off {b}pt")
        if opacity != 1: o.append(f"opacity={opacity}")
        if arrow: o.append(arrow)
        self.raw(f"\\draw[{','.join(o)}] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in pts) + ";")
    def out(self):
        return "\\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]\n" + f"\\path (0,0) rectangle ({self.w:.3f},{self.h:.3f});\n" + "\n".join(self.L) + "\n\\end{tikzpicture}\n"


class Ax:
    def __init__(self, fig, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False, xcat=None):
        self.f, self.x0, self.y0, self.w, self.h = fig, x0, y0, w, h; self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog
    def X(self, x):
        a, b = self.xlim
        t = (math.log10(max(x, 1e-300)) - math.log10(a)) / (math.log10(b) - math.log10(a)) if self.xlog else (x - a) / (b - a)
        return self.x0 + t * self.w
    def Y(self, y):
        a, b = self.ylim
        t = (math.log10(max(y, 1e-300)) - math.log10(a)) / (math.log10(b) - math.log10(a)) if self.ylog else (y - a) / (b - a)
        return self.y0 + t * self.h
    def P(self, x, y): return (self.X(x), self.Y(y))
    def clip_begin(self): self.f.raw(f"\\begin{{scope}}\\clip ({self.x0:.3f},{self.y0:.3f}) rectangle ({self.x0 + self.w:.3f},{self.y0 + self.h:.3f});")
    def clip_end(self): self.f.raw("\\end{scope}")
    def frame(self, xticks=None, yticks=None, xtl=None, ytl=None, xlabel=None, ylabel=None, title=None, subtitle=None, grid="y", panel=None, box=True, xrot=0, tickfont="tiny", labfont="scriptsize", ylab_dx=0.0):
        f = self.f
        xt = xticks if xticks is not None else []; yt = yticks if yticks is not None else []
        if grid in ("y", "xy"):
            for v in yt: f.line([(self.x0, self.Y(v)), (self.x0 + self.w, self.Y(v))], "cLight", 0.4)
        if grid in ("x", "xy"):
            for v in xt: f.line([(self.X(v), self.y0), (self.X(v), self.y0 + self.h)], "cLight", 0.4)
        if box: f.rect(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h, "cGray", None, 0.5)
        for i, v in enumerate(xt):
            f.line([(self.X(v), self.y0), (self.X(v), self.y0 - 0.07)], "cGray", 0.4)
            lab = xtl[i] if xtl is not None else fmt(v)
            f.text(self.X(v), self.y0 - 0.10, lab, anchor="north" if not xrot else "north east", size=tickfont, rot=xrot, color="cInk")
        for i, v in enumerate(yt):
            f.line([(self.x0, self.Y(v)), (self.x0 - 0.07, self.Y(v))], "cGray", 0.4)
            lab = ytl[i] if ytl is not None else fmt(v)
            f.text(self.x0 - 0.10, self.Y(v), lab, anchor="east", size=tickfont, color="cInk")
        if xlabel: f.text(self.x0 + self.w / 2, self.y0 - (0.62 if not xrot else 0.95), xlabel, size=labfont)
        if ylabel: f.text(self.x0 - 0.80 + ylab_dx, self.y0 + self.h / 2, ylabel, size=labfont, rot=90)
        if title:
            pre = f"\\textbf{{({panel})}}\\ " if panel else ""
            f.text(self.x0 + self.w / 2, self.y0 + self.h + 0.28, pre + title, size="footnotesize", bold=True)
        if subtitle: f.text(self.x0 + self.w / 2, self.y0 + self.h + 0.02 + (0.0), subtitle, size="tiny", color="cGray", anchor="south")
        if panel and not title: f.text(self.x0 - 0.55, self.y0 + self.h + 0.28, f"\\textbf{{({panel})}}", anchor="east", size="small")
    def line(self, xs, ys, color="cBlue", lw=1.0, dash=None, opacity=1):
        self.clip_begin(); self.f.line([self.P(x, y) for x, y in zip(xs, ys)], color, lw, dash, opacity); self.clip_end()
    def marker(self, x, y, color="cBlue", kind="o", r=0.07, fill=True, opacity=1, lw=0.5, edge=None):
        px, py = self.P(x, y); edge = edge or color
        fillspec = f"fill={color}" if fill else "fill=white"
        op = f",opacity={opacity}" if opacity != 1 else ""
        if kind == "o": self.f.raw(f"\\path[draw={edge},{fillspec},line width={lw}pt{op}] ({px:.3f},{py:.3f}) circle ({r:.3f});")
        elif kind == "s": self.f.raw(f"\\path[draw={edge},{fillspec},line width={lw}pt{op}] ({px - r:.3f},{py - r:.3f}) rectangle ({px + r:.3f},{py + r:.3f});")
        elif kind == "d": self.f.raw(f"\\path[draw={edge},{fillspec},line width={lw}pt{op}] ({px:.3f},{py + r * 1.3:.3f}) -- ({px + r * 1.3:.3f},{py:.3f}) -- ({px:.3f},{py - r * 1.3:.3f}) -- ({px - r * 1.3:.3f},{py:.3f}) -- cycle;")
        elif kind == "t": self.f.raw(f"\\path[draw={edge},{fillspec},line width={lw}pt{op}] ({px:.3f},{py + r * 1.2:.3f}) -- ({px + r * 1.15:.3f},{py - r * 0.8:.3f}) -- ({px - r * 1.15:.3f},{py - r * 0.8:.3f}) -- cycle;")
    def scatter(self, xs, ys, color="cBlue", kind="o", r=0.05, opacity=0.6, edge="none"):
        self.clip_begin()
        for x, y in zip(xs, ys):
            px, py = self.P(x, y)
            if kind == "o": self.f.raw(f"\\fill[{color},opacity={opacity}] ({px:.3f},{py:.3f}) circle ({r:.3f});")
        self.clip_end()
    def band(self, xs, lo, hi, color="cBlue", opacity=0.18):
        self.clip_begin()
        pts = [self.P(x, h) for x, h in zip(xs, hi)] + [self.P(x, l) for x, l in zip(xs[::-1], lo[::-1])]
        self.f.raw(f"\\fill[{color},opacity={opacity}] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in pts) + " -- cycle;"); self.clip_end()
    def bars(self, xs, hs, width, color="cBlue", base=0, opacity=1, edge=True):
        self.clip_begin()
        for x, h in zip(xs, hs):
            x0, x1 = self.X(x) - width / 2, self.X(x) + width / 2; y0, y1 = self.Y(base), self.Y(h)
            self.f.raw(f"\\path[fill={color},opacity={opacity}{',draw=white,line width=0.3pt' if edge else ''}] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
        self.clip_end()
    def err(self, x, lo, hi, color="cInk", cap=0.06, lw=0.7):
        px = self.X(x); self.f.line([(px, self.Y(lo)), (px, self.Y(hi))], color, lw); self.f.line([(px - cap, self.Y(lo)), (px + cap, self.Y(lo))], color, lw); self.f.line([(px - cap, self.Y(hi)), (px + cap, self.Y(hi))], color, lw)
    def herr(self, y, lo, hi, color="cInk", cap=0.06, lw=0.7):
        py = self.Y(y); self.f.line([(self.X(lo), py), (self.X(hi), py)], color, lw); self.f.line([(self.X(lo), py - cap), (self.X(lo), py + cap)], color, lw); self.f.line([(self.X(hi), py - cap), (self.X(hi), py + cap)], color, lw)
    def hline(self, y, color="cGray", lw=0.6, dash="3pt on 2pt", label=None, lab_x=None, lab_anchor="south west", size="tiny", lab_color=None):
        self.clip_begin(); self.f.line([(self.x0, self.Y(y)), (self.x0 + self.w, self.Y(y))], color, lw, dash); self.clip_end()
        if label: self.f.text(self.X(lab_x) if lab_x is not None else self.x0 + 0.05, self.Y(y) + 0.02, label, anchor=lab_anchor, size=size, color=lab_color or color)
    def vline(self, x, color="cGray", lw=0.6, dash="3pt on 2pt", label=None, lab_y=None, lab_anchor="south west", size="tiny"):
        self.clip_begin(); self.f.line([(self.X(x), self.y0), (self.X(x), self.y0 + self.h)], color, lw, dash); self.clip_end()
        if label: self.f.text(self.X(x) + 0.03, self.Y(lab_y) if lab_y is not None else self.y0 + self.h - 0.1, label, anchor=lab_anchor, size=size, color=color)
    def text(self, x, y, s, anchor="center", size="tiny", color="cInk", rot=0, bold=False, extra=""): self.f.text(self.X(x), self.Y(y), s, anchor, size, color, rot, bold, extra)
    def rtext(self, fx, fy, s, anchor="center", size="tiny", color="cInk", bold=False): self.f.text(self.x0 + fx * self.w, self.y0 + fy * self.h, s, anchor, size, color, 0, bold)
    def heat(self, Z, xedges, yedges, vmin=None, vmax=None, table=VIRIDIS, annotate=None, textsize="tiny", nan_color="white", seams=False):
        Z = np.asarray(Z, float); vmin = np.nanmin(Z) if vmin is None else vmin; vmax = np.nanmax(Z) if vmax is None else vmax
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if np.isnan(v): continue
                c = cmap((v - vmin) / (vmax - vmin + 1e-12), table); x0, x1 = self.X(xedges[j]), self.X(xedges[j + 1]); y0, y1 = self.Y(yedges[i]), self.Y(yedges[i + 1])
                self.f.raw(f"\\path[fill={rgb(c)},draw={'white' if seams else rgb(c)},line width={0.3 if seams else 0.25}pt] ({x0:.3f},{y0:.3f}) rectangle ({x1:.3f},{y1:.3f});")
                if annotate:
                    tc = "white" if (c[0] * 0.3 + c[1] * 0.59 + c[2] * 0.11) < 140 else "cInk"
                    self.f.text((x0 + x1) / 2, (y0 + y1) / 2, annotate(v), size=textsize, color=tc)
    def colorbar(self, vmin, vmax, x, y, w, h, label=None, ticks=None, table=VIRIDIS, horizontal=False, nd=2):
        n = 40
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n; c = cmap((t0 + t1) / 2, table)
            if horizontal: self.f.raw(f"\\path[fill={rgb(c)}] ({x + t0 * w:.3f},{y:.3f}) rectangle ({x + t1 * w:.3f},{y + h:.3f});")
            else: self.f.raw(f"\\path[fill={rgb(c)}] ({x:.3f},{y + t0 * h:.3f}) rectangle ({x + w:.3f},{y + t1 * h:.3f});")
        self.f.rect(x, y, x + w, y + h, "cGray", None, 0.4)
        for v in (ticks if ticks is not None else [vmin, (vmin + vmax) / 2, vmax]):
            t = (v - vmin) / (vmax - vmin)
            if horizontal: self.f.text(x + t * w, y - 0.06, fmt(v, nd), anchor="north", size="tiny")
            else: self.f.text(x + w + 0.06, y + t * h, fmt(v, nd), anchor="west", size="tiny")
        if label:
            if horizontal: self.f.text(x + w / 2, y - 0.38, label, size="tiny")
            else: self.f.text(x + w + 0.55, y + h / 2, label, size="tiny", rot=90)
    def legend(self, items, x, y, anchor="north west", size="tiny", dx=0.0, ncol=1, rowh=0.30, boxw=0.5, fill="white", cw=0.105):
        """items: (label, color, kind) with kind in line / dash / box / o / s / d / t ; (x,y) in cm (absolute) of the anchor corner."""
        nrow = math.ceil(len(items) / ncol)
        widths = [max(len(re_strip(it[0])) for it in items[c * nrow:(c + 1) * nrow]) * cw + boxw + 0.25 for c in range(ncol)]
        W = sum(widths) + 0.1; Hh = nrow * rowh + 0.1
        x0 = x if "west" in anchor else x - W; y1 = y if "north" in anchor else y + Hh; y0 = y1 - Hh
        self.f.rect(x0, y0, x0 + W, y1, "cLight", fill, 0.4, r=2, opacity=0.92)
        cx = x0 + 0.08
        for c in range(ncol):
            for r, (lab, color, kind) in enumerate(items[c * nrow:(c + 1) * nrow]):
                yy = y1 - 0.05 - rowh * (r + 0.5)
                if kind == "line": self.f.line([(cx, yy), (cx + boxw, yy)], color, 1.1)
                elif kind == "dash": self.f.line([(cx, yy), (cx + boxw, yy)], color, 1.0, "3pt on 2pt")
                elif kind == "box": self.f.rect(cx, yy - 0.06, cx + boxw, yy + 0.06, None, color, 0)
                else:
                    r_ = 0.06; px = cx + boxw / 2
                    if kind == "o": self.f.raw(f"\\path[fill={color}] ({px:.3f},{yy:.3f}) circle ({r_});")
                    elif kind == "s": self.f.raw(f"\\path[fill={color}] ({px - r_:.3f},{yy - r_:.3f}) rectangle ({px + r_:.3f},{yy + r_:.3f});")
                    elif kind == "d": self.f.raw(f"\\path[fill={color}] ({px:.3f},{yy + r_ * 1.3:.3f}) -- ({px + r_ * 1.3:.3f},{yy:.3f}) -- ({px:.3f},{yy - r_ * 1.3:.3f}) -- ({px - r_ * 1.3:.3f},{yy:.3f}) -- cycle;")
                    elif kind == "t": self.f.raw(f"\\path[fill={color}] ({px:.3f},{yy + r_ * 1.2:.3f}) -- ({px + r_ * 1.15:.3f},{yy - r_ * 0.8:.3f}) -- ({px - r_ * 1.15:.3f},{yy - r_ * 0.8:.3f}) -- cycle;")
                self.f.text(cx + boxw + 0.08, yy, lab, anchor="west", size=size)
            cx += widths[c]


def re_strip(s):
    import re
    s = re.sub(r"\\[a-zA-Z]+", "x", s); s = re.sub(r"[\$\{\}\^_]", "", s); return s
