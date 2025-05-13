import typing


def merge_intake_search_history(
    search_history: list[dict[str, list[typing.Any]]],
) -> dict[str, typing.Any]:
    """Create a facet mapping from an intake-esm search history.

    This takes an intake-esm search history, which typically looks something like
    ```python
    [
        {'variable_id': ['tos']},
        {'table_id': ['Omon']},
        {'experiment_id': ['historical']},
        {'member_id': ['r1i1p1f1']},
        {'source_id': ['ACCESS-ESM1-5']},
        {'grid_label': ['gn']},
        {'version': ['v.*']},
    ]
    ```
    and turns it into something like
    ```python
    {
        'variable_id': 'tos',
        'table_id': 'Omon',
        'experiment_id': 'historical',
        'member_id': 'r1i1p1f1',
        'source_id': 'ACCESS-ESM1-5',
        'grid_label': 'gn',
        'version': 'v.*',
    }
    ```

    Notes
    -----
    This function is really quite ugly & could probably be improved.
    """
    merged: dict[str, typing.Any] = {}

    for entry in search_history:
        for key, value in entry.items():
            if key in merged:
                if isinstance(merged[key], list):
                    merged[key].extend(value)
                else:
                    merged[key] = [merged[key]] + value
            else:
                merged[key] = value

    for key, val in merged.items():
        if isinstance(val, list) and len(val) == 1:
            merged[key] = val[0]

    return merged
