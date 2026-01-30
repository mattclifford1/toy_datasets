# terminal_plots.py
"""Render Matplotlib figures in the terminal.

This is meant to be a *drop-in* replacement for `plt.show()`:

    from data_loaders.terminal_plots import enable_terminal_show
    plotter = enable_terminal_show()
    ...
    plt.show()  # renders in terminal
    plotter.disable()  # restore original matplotlib show

On macOS terminals, ASCII ramps usually look rough. This module defaults to a
text renderer that shows scatter points as characters and prints axis/title
labels as text (no braille dots).

If you're using iTerm2, you can optionally render the actual PNG inline using
its OSC 1337 protocol (looks best) by passing `mode="iterm2"`.

Dependencies: pillow (PIL)
    pip install pillow
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import functools
from typing import Literal, Optional

import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("terminal_plots requires pillow: pip install pillow") from e


Mode = Literal["ascii", "iterm2", "text", "auto"]


# --- Rasterize Matplotlib figure to a PIL image --------------------------------

def _fig_to_pil(fig, *, dpi: int = 150) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# --- ASCII  ------------------------------------------------------------

_ASCII_RAMP = " .:-=+*#%@"  # light -> dark


def _pil_to_ascii(img: Image.Image, *, cols: int, rows: int) -> str:
    g = img.convert("L").resize((cols, rows))
    arr = list(g.getdata())
    n = len(_ASCII_RAMP) - 1
    lines = []
    for r in range(rows):
        row = arr[r * cols : (r + 1) * cols]
        lines.append("".join(_ASCII_RAMP[int(v / 255 * n)] for v in row))
    return "\n".join(lines)


# --- Text plot renderer (scatter + axes labels) --------------------------------

def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _format_tick(value: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000 or abs_value < 0.01:
        return f"{value:.1e}"
    return f"{value:.2g}"


def _get_suptitle(fig) -> str:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is None:
        return ""
    text = suptitle.get_text()
    return text.strip()


def _render_axes_text(ax, *, cols: int, rows: int, figure_title: str = "") -> str:
    title = ax.get_title()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    header = []
    if figure_title:
        header.append(figure_title.center(cols))
    if title:
        header.append(title.center(cols))
    if xlabel or ylabel:
        header.append(f"x: {xlabel} | y: {ylabel}".strip())

    header_lines = len(header)
    footer_lines = 2

    plot_height = max(6, rows - header_lines - footer_lines)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1

    yticks = _linspace(ymin, ymax, 5)
    ylabels = [_format_tick(v) for v in yticks]
    max_y_label = max(len(s) for s in ylabels) if ylabels else 0
    left_margin = max(6, max_y_label + 1)

    plot_width = max(10, cols - left_margin)
    grid = [[" "] * plot_width for _ in range(plot_height)]

    for r in range(plot_height):
        grid[r][0] = "|"
    for c in range(plot_width):
        grid[plot_height - 1][c] = "-"
    grid[plot_height - 1][0] = "+"

    def place_point(x: float, y: float, ch: str) -> None:
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return
        col = int(round((x - xmin) / (xmax - xmin) * (plot_width - 1)))
        row = int(round((ymax - y) / (ymax - ymin) * (plot_height - 1)))
        if 0 <= row < plot_height and 0 <= col < plot_width:
            existing = grid[row][col]
            if existing in (" ", "|", "-", "+"):
                grid[row][col] = ch
            elif existing != ch:
                grid[row][col] = "*"

    def place_line(x0: float, y0: float, x1: float, y1: float, ch: str) -> None:
        if xmax == xmin or ymax == ymin:
            return
        col0 = int(round((x0 - xmin) / (xmax - xmin) * (plot_width - 1)))
        row0 = int(round((ymax - y0) / (ymax - ymin) * (plot_height - 1)))
        col1 = int(round((x1 - xmin) / (xmax - xmin) * (plot_width - 1)))
        row1 = int(round((ymax - y1) / (ymax - ymin) * (plot_height - 1)))

        dcol = abs(col1 - col0)
        drow = abs(row1 - row0)
        steps = max(dcol, drow, 1)
        for step in range(steps + 1):
            t = step / steps
            col = int(round(col0 + (col1 - col0) * t))
            row = int(round(row0 + (row1 - row0) * t))
            if 0 <= row < plot_height and 0 <= col < plot_width:
                existing = grid[row][col]
                if existing in (" ", "|", "-", "+"):
                    grid[row][col] = ch
                elif existing != ch:
                    grid[row][col] = "*"

    marker_cycle = ["o", "x", "+", "*", "#"]
    collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
    for idx, collection in enumerate(collections):
        ch = marker_cycle[idx % len(marker_cycle)]
        offsets = collection.get_offsets()
        try:
            for x, y in offsets:
                place_point(float(x), float(y), ch)
        except Exception:
            continue

    for line in ax.lines:
        xdata = line.get_xdata(orig=False)
        ydata = line.get_ydata(orig=False)
        if len(xdata) < 2:
            for x, y in zip(xdata, ydata):
                place_point(float(x), float(y), ".")
            continue
        marker = line.get_marker()
        if marker in ("o", "x", "+", "*"):
            ch = marker
        else:
            ch = "."
        for i in range(len(xdata) - 1):
            place_line(float(xdata[i]), float(ydata[i]), float(xdata[i + 1]), float(ydata[i + 1]), ch)

    y_tick_rows = []
    for v in yticks:
        row = int(round((ymax - v) / (ymax - ymin) * (plot_height - 1)))
        y_tick_rows.append(row)

    lines = []
    for r in range(plot_height):
        if r in y_tick_rows:
            v = yticks[y_tick_rows.index(r)]
            label = _format_tick(v).rjust(left_margin - 1)
        else:
            label = " " * (left_margin - 1)
        lines.append(label + " " + "".join(grid[r]))

    xticks = _linspace(xmin, xmax, 5)
    xtick_cols = [
        int(round((v - xmin) / (xmax - xmin) * (plot_width - 1))) for v in xticks
    ]
    tick_line = [" "] * (left_margin + plot_width)
    for c in xtick_cols:
        if 0 <= c < plot_width:
            tick_line[left_margin + c] = "+"
    lines.append("".join(tick_line))

    label_line = [" "] * (left_margin + plot_width)
    for v, c in zip(xticks, xtick_cols):
        label = _format_tick(v)
        start = left_margin + c - len(label) // 2
        start = max(left_margin, min(start, left_margin + plot_width - len(label)))
        for i, ch in enumerate(label):
            label_line[start + i] = ch
    lines.append("".join(label_line))

    return "\n".join(header + lines)


# --- iTerm2 inline image (best overall if supported) ---------------------------

def _is_iterm2() -> bool:
    return os.environ.get("TERM_PROGRAM") == "iTerm.app" or "ITERM_SESSION_ID" in os.environ


def _print_iterm2_inline_png(png_bytes: bytes, *, width_cols: Optional[int] = None) -> None:
    """Print PNG bytes inline using iTerm2 OSC 1337 protocol."""
    b64 = base64.b64encode(png_bytes).decode("ascii")

    # iTerm supports size hints like width=NN (cells) or width=NNpx.
    # We'll use cell width if provided; otherwise let iTerm decide.
    width_part = f";width={int(width_cols)}" if width_cols else ""

    # OSC 1337;File=...:base64
    # Use inline=1 so it renders in the scrollback.
    payload = f"\033]1337;File=inline=1{width_part};preserveAspectRatio=1:{b64}\a"
    print(payload, end="")


# --- Public API ----------------------------------------------------------------

class TerminalPlotter:
    """Enable/disable a terminal renderer for matplotlib via a show() monkey-patch."""

    def __init__(
        self,
        *,
        mode: Mode = "auto",
        width: Optional[int] = None,
        height: Optional[int] = None,
        clear: bool = True,
        close: bool = True,
        dpi: int = 150,
        margin_cols: int = 2,
    ) -> None:
        self.mode = mode
        self.width = width
        self.height = height
        self.clear = clear
        self.close = close
        self.dpi = dpi
        self.margin_cols = margin_cols
        self._original_show = None
        self._patched_show = None

    def enable(self) -> "TerminalPlotter":
        if self._original_show is None:
            self._original_show = plt.show
            @functools.wraps(self._original_show)
            def terminal_show(*args, **kwargs):
                return self._terminal_show(*args, **kwargs)

            self._patched_show = terminal_show
            plt.show = terminal_show
        return self

    def disable(self) -> None:
        if self._original_show is not None:
            plt.show = self._original_show
            self._original_show = None
            self._patched_show = None

    def __enter__(self) -> "TerminalPlotter":
        return self.enable()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.disable()
        return False

    def _terminal_show(self, *args, **kwargs) -> None:
        fnums = plt.get_fignums()
        if not fnums:
            return

        term_cols, term_rows = shutil.get_terminal_size((120, 40))

        # Default sizing: leave a little margin so lines don't wrap.
        cols = self.width if self.width is not None else max(20, term_cols - self.margin_cols)
        rows = self.height if self.height is not None else max(10, term_rows - 2)

        chosen: Mode
        if self.mode == "auto":
            chosen = "iterm2" if _is_iterm2() else "text"
        else:
            chosen = self.mode

        if self.clear:
            print("\033[2J\033[H", end="")

        for i, n in enumerate(fnums):
            if i:
                print("\n")

            fig = plt.figure(n)

            if chosen == "iterm2":
                if not _is_iterm2():
                    # Fall back gracefully
                    chosen_local: Mode = "text"
                else:
                    pil = _fig_to_pil(fig, dpi=self.dpi)
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    _print_iterm2_inline_png(buf.getvalue(), width_cols=cols)
                    print()  # newline after the image
                    chosen_local = "iterm2"

                if chosen_local == "iterm2":
                    pass
                else:
                    # Render via text fallback
                    axes = fig.axes or []
                    if axes:
                        figure_title = _get_suptitle(fig)
                        rows_per = rows if len(axes) == 1 else max(10, rows // len(axes))
                        for ax_idx, ax in enumerate(axes):
                            if ax_idx:
                                print("\n")
                            fig_title = figure_title if ax_idx == 0 else ""
                            print(_render_axes_text(ax, cols=cols, rows=rows_per, figure_title=fig_title))

            elif chosen == "ascii":
                pil = _fig_to_pil(fig, dpi=self.dpi)
                print(_pil_to_ascii(pil, cols=cols, rows=rows))

            elif chosen == "text":
                axes = fig.axes or []
                if not axes:
                    return
                figure_title = _get_suptitle(fig)
                rows_per = rows if len(axes) == 1 else max(10, rows // len(axes))
                for ax_idx, ax in enumerate(axes):
                    if ax_idx:
                        print("\n")
                    fig_title = figure_title if ax_idx == 0 else ""
                    print(_render_axes_text(ax, cols=cols, rows=rows_per, figure_title=fig_title))

            else:
                raise ValueError(f"Unknown mode: {chosen}")

            if self.close:
                plt.close(fig)


def enable_terminal_show(
    *,
    mode: Mode = "auto",
    width: Optional[int] = None,
    height: Optional[int] = None,
    clear: bool = True,
    close: bool = True,
    dpi: int = 150,
    margin_cols: int = 2,
) -> TerminalPlotter:
    """Monkey-patch `matplotlib.pyplot.show()` to render figures in the terminal.

    Returns a TerminalPlotter instance so you can call .disable() later.
    """
    return TerminalPlotter(
        mode=mode,
        width=width,
        height=height,
        clear=clear,
        close=close,
        dpi=dpi,
        margin_cols=margin_cols,
    ).enable()

def terminal_show():
    """Shortcut to enable terminal plotting with default settings."""
    plot_env = enable_terminal_show()
    plt.show()
    # reset terminal plot to previous state
    plot_env.disable()


if __name__ == "__main__":
    import numpy as np

    plotter = enable_terminal_show(mode="auto")

    x = np.linspace(0, 2 * np.pi, 300)
    plt.plot(x, np.sin(x))
    plt.title("Sine wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)

    plt.show()

    # scatter plot with x and o markers
    plt.scatter([1,2,3],[1,4,9], marker='x', c='red')
    plt.scatter([1.5,2.5],[2,6], marker='o', c='blue')
    plt.show()
    plotter.disable()
