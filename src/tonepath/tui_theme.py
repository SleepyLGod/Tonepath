"""Theme palettes for the Tonepath terminal interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TonepathPalette:
    """Named color palette used by TUI rich text renderers."""

    key: str
    label: str
    dark: bool
    primary: str
    secondary: str
    accent: str
    success: str
    warning: str
    muted: str
    text: str
    background: str
    surface: str
    panel: str


WARM_LINE = TonepathPalette(
    key="warmline",
    label="Warmline",
    dark=True,
    primary="#d8a657",
    secondary="#6fb7a6",
    accent="#c9825c",
    success="#8ebf7f",
    warning="#e0b45f",
    muted="#a7afa5",
    text="#e6e0cf",
    background="#101311",
    surface="#151914",
    panel="#171b18",
)

MIDNIGHT = TonepathPalette(
    key="midnight",
    label="Midnight",
    dark=True,
    primary="#8fb8ff",
    secondary="#6fd3c7",
    accent="#c6a5ff",
    success="#8bd99a",
    warning="#f0c36a",
    muted="#a5b0c2",
    text="#e6ecf5",
    background="#0d1117",
    surface="#111827",
    panel="#151d2b",
)

HIGH_CONTRAST = TonepathPalette(
    key="high-contrast",
    label="High Contrast",
    dark=True,
    primary="#ffd166",
    secondary="#70e0c2",
    accent="#ff8a65",
    success="#9cff9c",
    warning="#ffdf70",
    muted="#d0d0d0",
    text="#ffffff",
    background="#000000",
    surface="#0b0b0b",
    panel="#121212",
)

SOLARIZED_DARK = TonepathPalette(
    key="solarized-dark",
    label="Solarized Dark",
    dark=True,
    primary="#b58900",
    secondary="#2aa198",
    accent="#cb4b16",
    success="#859900",
    warning="#dc322f",
    muted="#93a1a1",
    text="#eee8d5",
    background="#002b36",
    surface="#073642",
    panel="#0b3440",
)

SOLARIZED_LIGHT = TonepathPalette(
    key="solarized-light",
    label="Solarized Light",
    dark=False,
    primary="#b58900",
    secondary="#2aa198",
    accent="#cb4b16",
    success="#859900",
    warning="#dc322f",
    muted="#657b83",
    text="#586e75",
    background="#fdf6e3",
    surface="#eee8d5",
    panel="#f5efdc",
)

CATPPUCCIN_MOCHA = TonepathPalette(
    key="catppuccin-mocha",
    label="Catppuccin Mocha",
    dark=True,
    primary="#f5c2e7",
    secondary="#94e2d5",
    accent="#cba6f7",
    success="#a6e3a1",
    warning="#f9e2af",
    muted="#a6adc8",
    text="#cdd6f4",
    background="#1e1e2e",
    surface="#313244",
    panel="#181825",
)

CATPPUCCIN_LATTE = TonepathPalette(
    key="catppuccin-latte",
    label="Catppuccin Latte",
    dark=False,
    primary="#8839ef",
    secondary="#179299",
    accent="#ea76cb",
    success="#40a02b",
    warning="#df8e1d",
    muted="#6c6f85",
    text="#4c4f69",
    background="#eff1f5",
    surface="#e6e9ef",
    panel="#dce0e8",
)

DRACULA = TonepathPalette(
    key="dracula",
    label="Dracula",
    dark=True,
    primary="#bd93f9",
    secondary="#8be9fd",
    accent="#ff79c6",
    success="#50fa7b",
    warning="#ffb86c",
    muted="#6272a4",
    text="#f8f8f2",
    background="#282a36",
    surface="#44475a",
    panel="#343746",
)

JUKEBOX = TonepathPalette(
    key="jukebox",
    label="Jukebox",
    dark=True,
    primary="#a6e22e",
    secondary="#66d9ef",
    accent="#e6db74",
    success="#a6e22e",
    warning="#fd971f",
    muted="#8f9b7a",
    text="#f8f8f2",
    background="#050705",
    surface="#10140d",
    panel="#0b0f09",
)

PALETTES: tuple[TonepathPalette, ...] = (
    WARM_LINE,
    MIDNIGHT,
    HIGH_CONTRAST,
    SOLARIZED_DARK,
    SOLARIZED_LIGHT,
    CATPPUCCIN_MOCHA,
    CATPPUCCIN_LATTE,
    DRACULA,
    JUKEBOX,
)
PALETTE_BY_KEY = {palette.key: palette for palette in PALETTES}
DEFAULT_THEME = WARM_LINE.key


def normalize_theme(value: str | None) -> str:
    """Return a valid theme key, falling back to the default theme."""

    key = (value or DEFAULT_THEME).strip().lower()
    return key if key in PALETTE_BY_KEY else DEFAULT_THEME


def next_theme(value: str | None) -> str:
    """Return the next theme key in the stable TUI theme cycle."""

    current = normalize_theme(value)
    keys = [palette.key for palette in PALETTES]
    index = keys.index(current)
    return keys[(index + 1) % len(keys)]
