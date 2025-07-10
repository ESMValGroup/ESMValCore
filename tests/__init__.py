"""Provides testing capabilities for :mod:`esmvaltool` package."""

import unittest
from unittest import mock

import numpy as np
from cf_units import Unit
from iris.coords import (
    AncillaryVariable,
    AuxCoord,
    CellMeasure,
    CellMethod,
    DimCoord,
)
from iris.cube import Cube

from esmvalcore.preprocessor import PreprocessorFile as PreprocessorFileBase


def assert_array_equal(a, b):
    """Assert that array a equals array b."""
    np.testing.assert_array_equal(a, b)
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        np.testing.assert_array_equal(a.mask, b.mask)


class Test(unittest.TestCase):
    """Provides esmvaltool specific testing functionality."""

    def _remove_testcase_patches(self):
        """Remove per-testcase patches installed by :meth:`patch`."""
        # Remove all patches made, ignoring errors.
        for patch in self.testcase_patches:
            patch.stop()

        # Reset per-test patch control variable.
        self.testcase_patches.clear()

    def patch(self, *args, **kwargs):
        """Install a patch to be removed automatically after the current test.

        The patch is created with :func:`unittest.mock.patch`.

        Parameters
        ----------
        args : list
            The parameters to be passed to :func:`unittest.mock.patch`.
        kwargs : dict
            The keyword parameters to be passed to :func:`unittest.mock.patch`.

        Returns
        -------
            The substitute mock instance returned by
            :func:`unittest.patch.start`.
        """
        # Make the new patch and start it.
        patch = unittest.mock.patch(*args, **kwargs)
        start_result = patch.start()

        # Create the per-testcases control variable if it does not exist.
        # NOTE: this mimics a setUp method, but continues to work when a
        # subclass defines its own setUp.
        if not hasattr(self, "testcase_patches"):
            self.testcase_patches = {}

        # When installing the first patch, schedule remove-all at cleanup.
        if not self.testcase_patches:
            self.addCleanup(self._remove_testcase_patches)

        # Record the new patch and start object for reference.
        self.testcase_patches[patch] = start_result

        # Return patch replacement object.
        return start_result

    def assert_array_equal(self, a, b):
        assert_array_equal(a, b)


class PreprocessorFile(mock.Mock):
    """Mocked PreprocessorFile."""

    def __init__(self, cubes, filename, attributes, settings=None, **kwargs):
        """Initialize with cubes."""
        super().__init__(spec=PreprocessorFileBase, **kwargs)
        self.cubes = cubes
        self.filename = filename
        self.attributes = attributes
        if settings is None:
            self.settings = {}
        else:
            self.settings = settings
        self.mock_ancestors = set()
        self.wasderivedfrom = mock.Mock(side_effect=self.mock_ancestors.add)
        self.copy_provenance = mock.Mock(return_value=self)

    group = PreprocessorFileBase.group


def create_realistic_4d_cube():
    """Create a realistic 4D cube."""
    time = DimCoord(
        [11.0, 12.0],
        standard_name="time",
        units=Unit("hours since 1851-01-01", calendar="360_day"),
    )
    plev = DimCoord([50000], standard_name="air_pressure", units="Pa")
    lat = DimCoord([0.0, 1.0], standard_name="latitude", units="degrees_north")
    lon = DimCoord(
        [0.0, 20.0, 345.0],
        standard_name="longitude",
        units="degrees_east",
    )

    aux_2d_data = np.arange(2 * 3).reshape(2, 3)
    aux_2d_bounds = np.stack(
        (aux_2d_data - 1, aux_2d_data, aux_2d_data + 1),
        axis=-1,
    )
    aux_2d = AuxCoord(aux_2d_data, var_name="aux_2d")
    aux_2d_with_bnds = AuxCoord(
        aux_2d_data,
        bounds=aux_2d_bounds,
        var_name="aux_2d_with_bnds",
    )
    aux_time = AuxCoord(["Jan", "Jan"], var_name="aux_time")
    aux_lon = AuxCoord([0, 1, 2], var_name="aux_lon")

    cell_area = CellMeasure(
        np.arange(2 * 2 * 3).reshape(2, 2, 3) + 10,
        standard_name="cell_area",
        units="m2",
        measure="area",
    )
    type_var = AncillaryVariable(
        [["sea", "land", "lake"], ["lake", "sea", "land"]],
        var_name="type",
        units="no_unit",
    )

    return Cube(
        np.ma.masked_inside(
            np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3),
            1,
            3,
        ),
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
        cell_methods=[CellMethod("mean", "time")],
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
        aux_coords_and_dims=[
            (aux_2d, (0, 3)),
            (aux_2d_with_bnds, (0, 3)),
            (aux_time, 0),
            (aux_lon, 3),
        ],
        cell_measures_and_dims=[(cell_area, (0, 2, 3))],
        ancillary_variables_and_dims=[(type_var, (0, 3))],
        attributes={"test": 1},
    )
