import json
import sys
from pathlib import Path

import rich.pretty

SKIP = {
    "SOURCE",
}


def read(file: Path) -> dict:
    result = {}
    for line in file.read_text().split("\n"):
        if not line or line.startswith("!"):
            continue
        key, value = [elem.strip() for elem in line.split(":", 1)]
        if key in SKIP:
            continue
        result[key] = value
    return result


if __name__ == "__main__":
    filenames = sys.argv[1:]
    result = {
        "Header": {
            "table_id": "Table custom",
            "generic_levels": "olevel",
        },
        "variable_entry": {},
    }
    for filename in filenames:
        if "coordinates" in filename:
            continue
        variable = read(Path(filename))
        variable_entry = variable.pop("variable_entry")
        if "plevs" in variable["dimensions"]:
            variable["dimensions"] = variable["dimensions"].replace(
                "plevs",
                "plev19",
            )
        if "out_name" not in variable:
            variable["out_name"] = variable_entry
        result["variable_entry"][variable_entry] = variable
    print(json.dumps(result, indent=4, sort_keys=True))
    # rich.pretty.pprint(result)
