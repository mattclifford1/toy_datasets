# terminal_plots.py
"""Render Matplotlib figures in the terminal.

This is meant to be a *drop-in* replacement for `plt.show()`:

    from data_loaders.terminal_plots import enable_terminal_show
    plotter = enable_terminal_show()
    ...
    plt.show()  # renders in terminal
    plotter.disable()  # restore original matplotlib show

On macOS terminals, ASCII ramps usually look rough. This module defaults to a
Unicode *braille* renderer (2x4 dots per character) which is dramatically
sharper.

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


Mode = Literal["braille", "ascii", "iterm2", "auto"]


# --- Rasterize Matplotlib figure to a PIL image --------------------------------

def _fig_to_pil(fig, *, dpi: int = 150) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


# --- Braille renderer (best looking text-mode option) --------------------------

# Braille dot bit positions (Unicode braille patterns):
# Dots are numbered:
#   1 4
#   2 5
#   3 6
#   7 8
# And bits map to: 1<<0 .. 1<<7
_BRAILLE_BITS = {
    (0, 0): 0,  # dot 1
    (0, 1): 1,  # dot 2
    (0, 2): 2,  # dot 3
    (0, 3): 6,  # dot 7
    (1, 0): 3,  # dot 4
    (1, 1): 4,  # dot 5
    (1, 2): 5,  # dot 6
    (1, 3): 7,  # dot 8
}


def _pil_to_braille(img: Image.Image, *, cols: int, rows: int) -> str:
    """Convert an image to braille characters.

    Each terminal character represents a 2x4 pixel block.
    """
    # Resize to pixel grid matching braille resolution
    px_w, px_h = cols * 2, rows * 4
    g = img.convert("L").resize((px_w, px_h))

    # Dither down to 1-bit (helps a lot for plots)
    bw = g.convert("1", dither=Image.FLOYDSTEINBERG)
    pix = bw.load()

    out_lines = []
    for r in range(rows):
        line_chars = []
        y0 = r * 4
        for c in range(cols):
            x0 = c * 2
            bits = 0
            for dx in (0, 1):
                for dy in (0, 1, 2, 3):
                    # In mode "1", white is 255, black is 0.
                    # We want a dot for "ink" (black).
                    if pix[x0 + dx, y0 + dy] == 0:
                        bits |= 1 << _BRAILLE_BITS[(dx, dy)]
            line_chars.append(chr(0x2800 + bits))
        out_lines.append("".join(line_chars))

    return "\n".join(out_lines)


# --- ASCII fallback ------------------------------------------------------------

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
            chosen = "iterm2" if _is_iterm2() else "braille"
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
                    chosen_local: Mode = "braille"
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
                    # Render via braille fallback
                    pil = _fig_to_pil(fig, dpi=self.dpi)
                    print(_pil_to_braille(pil, cols=cols, rows=rows))

            elif chosen == "braille":
                pil = _fig_to_pil(fig, dpi=self.dpi)
                print(_pil_to_braille(pil, cols=cols, rows=rows))

            elif chosen == "ascii":
                pil = _fig_to_pil(fig, dpi=self.dpi)
                print(_pil_to_ascii(pil, cols=cols, rows=rows))

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

    plotter.disable()

    # plotter = enable_terminal_show()          # defaults to mode="auto" (braille unless iTerm2)
    # plotter = enable_terminal_show(mode="braille")  # force braille
    # plotter = enable_terminal_show(mode="iterm2")   # if you're on iTerm2: best quality (inline PNG)

    # scatter plot with x and o markers
    plt.scatter([1,2,3],[1,4,9], marker='x', c='red')
    plt.scatter([1.5,2.5],[2,6], marker='o', c='blue')
    plt.show()
    plotter.disable()
