"""Convert CMIP5-style custom CMOR tables to a CMIP6-style custom table.

Example usage: `python convert-cmip5-to-cmip6.py esmvalcore/cmor/tables/cmip5-custom/*.dat`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def read(file: Path) -> dict[str, str]:
    """Read a CMIP5-style custom CMOR table file into a `dict`."""
    result = {}
    for line in file.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith("!"):
            continue
        key, value = [elem.strip() for elem in line.split(":", 1)]
        result[key] = value
    return result


def translate(files: Iterable[Path]) -> dict[str, Any]:
    """Read in CMIP5-style custom CMOR table files and return a CMIP6-style custom table."""
    result = {
        "Header": {
            "table_id": "Table custom",
            "generic_levels": "olevel",
        },
        "variable_entry": {},
    }
    for file in files:
        # Skip the coordinates file and use standard CMIP6 coordinates instead.
        if "coordinates" in file.name:
            continue
        variable = read(file)
        variable_entry = variable.pop("variable_entry")
        # Remove the "SOURCE" key which has no meaning in CMOR tables.
        variable.pop("SOURCE", None)
        # Some files are missing `out_name`, assume it is the same as the entry.
        if "out_name" not in variable:
            variable["out_name"] = variable_entry
        # Use a CMIP6 pressure levels coordinate.
        if "plevs" in variable["dimensions"]:
            variable["dimensions"] = variable["dimensions"].replace(
                "plevs",
                "plev19",
            )
        result["variable_entry"][variable_entry] = variable
    return result


if __name__ == "__main__":
    table_files = [Path(p) for p in sys.argv[1:]]
    print(json.dumps(translate(table_files), indent=4, sort_keys=True))  # noqa: T201
