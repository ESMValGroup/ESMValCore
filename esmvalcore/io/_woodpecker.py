"""Fix data with `woodpecker <https://github.com/roocs/woodpecker>`__.

If a woodpecker recipe matches the data, woodpecker is responsible for fixing
it and ESMValCore does not apply its own fixes. This allows for a gradual
transition from ESMValCore fixes to woodpecker fixes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import woodpecker.recipe
from iris.mesh import MeshCoord, MeshXY
from woodpecker.io.backends import XarrayInput
from woodpecker.recipes.loaders import RECIPE_PATH_ENV

if TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr
    from iris.cube import Cube, CubeList
    from woodpecker.recipes.models import Recipe
    from woodpecker.stores.base import RecipeStore

logger = logging.getLogger(__name__)

PROVENANCE_ATTRIBUTE = "woodpecker_provenance"
"""Global attribute that marks data as fixed by woodpecker.

This is the attribute that woodpecker writes when it is asked to embed
provenance metadata. ESMValCore does not apply its own fixes to data that has
this attribute.
"""


def merge_provenance(values: Iterable[str]) -> str:
    """Merge values of the ``woodpecker_provenance`` attribute.

    When data from several files is combined, e.g. by concatenation, the
    provenance of each file is kept. The result is valid JSON: the record
    itself if all files have the same record, otherwise a list of the
    distinct records. Values that are already lists, from an earlier merge,
    are flattened.
    """
    records: dict[str, Any] = {}
    for value in values:
        parsed = json.loads(value)
        for record in parsed if isinstance(parsed, list) else [parsed]:
            records.setdefault(json.dumps(record, sort_keys=True), record)
    merged = list(records.values())
    return json.dumps(
        merged[0] if len(merged) == 1 else merged,
        sort_keys=True,
    )


def is_fixed_by_woodpecker(cube: Cube) -> bool:
    """Check if ``cube`` contains data that has been fixed by woodpecker."""
    return PROVENANCE_ATTRIBUTE in cube.attributes


def _masked_invalid(
    array: np.ndarray | da.Array,
) -> np.ndarray | da.Array | None:
    """Mask NaN and infinite values, keeping lazy arrays lazy.

    Returns :obj:`None` if ``array`` is not lazy and has no values to mask,
    so it does not need to be replaced.
    """
    if array.dtype.kind != "f":
        return None
    if isinstance(array, da.Array):
        return da.ma.masked_invalid(array)
    invalid = ~np.isfinite(np.ma.getdata(array))
    if not invalid.any():
        return None
    return np.ma.masked_where(invalid, array)


def mask_invalid(cubes: CubeList) -> None:
    """Mask NaN and infinite values in ``cubes`` in place.

    Woodpecker fixes work on :class:`xarray.Dataset` objects, which represent
    missing values as NaN. When the data is converted to Iris cubes, these
    are only masked if the variable still has a ``_FillValue`` in its
    encoding, which fixes may remove, e.g. when they compute new data.
    ESMValCore expects missing values to be masked.

    This masks the data, the points and bounds of the auxiliary coordinates
    (including the coordinates of a mesh), the cell measures and the
    ancillary variables. Dimension coordinates cannot be masked, so they are
    not changed. Cubes loaded from the same file share their mesh, so each
    mesh is masked only once; the mesh coordinates of the cubes follow the
    mesh.
    """
    meshes: list[MeshXY] = []
    for cube in cubes:
        if (data := _masked_invalid(cube.core_data())) is not None:
            cube.data = data
        coords = [c for c in cube.aux_coords if not isinstance(c, MeshCoord)]
        if cube.mesh is not None and not any(
            cube.mesh is mesh for mesh in meshes
        ):
            meshes.append(cube.mesh)
            coords.extend(cube.mesh.coords())
        for coord in coords:
            if (points := _masked_invalid(coord.core_points())) is not None:
                coord.points = points
            if (
                coord.has_bounds()
                and (bounds := _masked_invalid(coord.core_bounds()))
                is not None
            ):
                coord.bounds = bounds
        for variable in [*cube.cell_measures(), *cube.ancillary_variables()]:
            if (data := _masked_invalid(variable.core_data())) is not None:
                variable.data = data


_RECIPE_CATALOGS: dict[str | None, RecipeStore] = {}


def _recipe_catalog() -> RecipeStore:
    """Return the catalog of woodpecker recipes.

    The catalog is cached, and reloaded when the environment variable that
    woodpecker uses to find extra recipes changes.
    """
    recipe_path = os.environ.get(RECIPE_PATH_ENV)
    if recipe_path not in _RECIPE_CATALOGS:
        _RECIPE_CATALOGS[recipe_path] = woodpecker.recipe.create_recipe_store(
            "catalog",
            None,
        )
    return _RECIPE_CATALOGS[recipe_path]


def find_recipe(dataset: xr.Dataset, name: str) -> Recipe | None:
    """Find the woodpecker recipe that matches the data, if any.

    Only the global attributes of ``dataset`` and ``name`` are used to find
    the recipe, so ``dataset`` may be a dataset without any variables that
    only has the global attributes of the data.

    Parameters
    ----------
    dataset:
        The data to find a recipe for.
    name:
        The name of the data. Woodpecker uses the recipe whose path patterns
        match this name.

    Returns
    -------
    :
        The matching recipe, or :obj:`None` if no recipe matches.

    Raises
    ------
    ValueError
        If more than one recipe matches.
    """
    try:
        recipes = _recipe_catalog().lookup(dataset, path=name)
    except Exception as exc:
        exc.add_note(f"While finding a woodpecker recipe for {name}")
        raise
    if len(recipes) > 1:
        msg = (
            f"Multiple woodpecker recipes match {name}: "
            f"{', '.join(r.id for r in recipes)}."
        )
        raise ValueError(msg)
    if not recipes:
        logger.debug("No woodpecker recipe matches %s", name)
        return None
    return recipes[0]


def apply_recipe(dataset: xr.Dataset, recipe: Recipe, name: str) -> list[str]:
    """Fix ``dataset`` in place with a woodpecker recipe.

    The dataset gets the global attribute ``woodpecker_provenance``, so
    ESMValCore does not apply its own fixes to it. This also happens if the
    recipe does not change the data.

    Parameters
    ----------
    dataset:
        The data to fix.
    recipe:
        The woodpecker recipe to apply.
    name:
        The name of the data.

    Returns
    -------
    :
        The ids of the fixes that changed the data.
    """
    try:
        result = woodpecker.recipe.apply(
            XarrayInput(payload=dataset, name=name),
            recipe,
            dry_run=False,
        )
    except Exception as exc:
        exc.add_note(f"While fixing {name} with woodpecker")
        raise

    applied = [
        preview["fix_id"]
        for preview in result.stats["preview"]
        if preview["changed"]
    ]
    logger.debug(
        "Applied woodpecker recipe %s to %s, fixes that changed the data: %s",
        recipe.id,
        name,
        ", ".join(applied) if applied else "none",
    )

    # Woodpecker only writes this attribute when asked to embed provenance
    # metadata, which its Python API does not support yet, and only when the
    # data changed. Use the same keys as woodpecker does, plus the recipe.
    dataset.attrs[PROVENANCE_ATTRIBUTE] = json.dumps(
        {
            "run_id": "",
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "source": name,
            "recipe_id": recipe.id,
            "applied_fix_ids": applied,
        },
        sort_keys=True,
    )
    return applied


def apply_fixes(dataset: xr.Dataset, name: str) -> list[str]:
    """Fix ``dataset`` in place with woodpecker.

    If a woodpecker recipe matches the data, it is applied and the dataset
    gets the global attribute ``woodpecker_provenance``, so ESMValCore does not
    apply its own fixes to it. This also happens if the recipe does not change
    the data. If no recipe matches, the data is not changed.

    Parameters
    ----------
    dataset:
        The data to fix.
    name:
        The name of the data. Woodpecker uses the recipe whose path patterns
        match this name.

    Returns
    -------
    :
        The ids of the fixes that changed the data.
    """
    recipe = find_recipe(dataset, name)
    if recipe is None:
        return []
    return apply_recipe(dataset, recipe, name)
