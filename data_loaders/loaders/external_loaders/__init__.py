from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# MIMIC data location
# ---------------------------------------------------------------------------
# MIMIC CSVs are NOT shipped with this package due to PhysioNet licensing.
# The loaders look for them in the following order:
#   1. an explicit ``data_path`` argument passed to the loader
#   2. the ``MIMIC_DATA_DIR`` environment variable
#   3. ``DEFAULT_MIMIC_DATA_DIR`` below
# The chosen directory is expected to contain ``MIMIC-III/`` and ``MIMIC-IV/``
# sub-folders holding the preprocessed CSV files.
DEFAULT_MIMIC_DATA_DIR = os.path.expanduser('~/datasets')

# Shown when a requested MIMIC file cannot be found.
DOWNLOAD_INSTRUCTIONS = (
    "MIMIC data is not shipped with this package due to PhysioNet licensing.\n"
    "Download it (credentialed access required), then point the loader at the\n"
    "directory containing the 'MIMIC-III/' and 'MIMIC-IV/' folders via either\n"
    "the `data_path` argument or the MIMIC_DATA_DIR environment variable:\n"
    "  - MIMIC-III: https://physionet.org/content/mimiciii/\n"
    "  - MIMIC-IV:  https://physionet.org/content/mimiciv/\n"
    "Or email Matt Clifford <matt.clifford@bristol.ac.uk> for the preprocessed CSVs."
)


def resolve_mimic_dir(data_path: str | None = None) -> str:
    """Resolve the base directory that holds the MIMIC-III / MIMIC-IV folders.

    Parameters
    ----------
    data_path : str or None, default=None
        Explicit base directory. When None, falls back to the ``MIMIC_DATA_DIR``
        environment variable and then to :data:`DEFAULT_MIMIC_DATA_DIR`.

    Returns
    -------
    str
        Absolute base directory (``~`` expanded).
    """
    if data_path is not None:
        return os.path.expanduser(data_path)
    return os.path.expanduser(os.environ.get('MIMIC_DATA_DIR', DEFAULT_MIMIC_DATA_DIR))


def mimic_file(version: str, filename: str, data_path: str | None = None) -> str:
    """Return the absolute path to a MIMIC CSV, or raise if it is missing.

    Parameters
    ----------
    version : str
        ``'MIMIC-III'`` or ``'MIMIC-IV'`` (the sub-folder name).
    filename : str
        CSV file name within that sub-folder.
    data_path : str or None, default=None
        Base directory override, forwarded to :func:`resolve_mimic_dir`.

    Returns
    -------
    str
        Absolute path to the requested file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist, with download instructions.
    """
    base = resolve_mimic_dir(data_path)
    path = os.path.join(base, version, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {version} data file:\n    {path}\n\n{DOWNLOAD_INSTRUCTIONS}"
        )
    return path
