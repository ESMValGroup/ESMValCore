"""Integration tests for :func:`esmvalcore.preprocessor._io.load`."""

import os
import tempfile
import unittest
import warnings
from importlib.resources import files as importlib_files
from pathlib import Path

import iris
import ncdata
import numpy as np
import xarray as xr
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from esmvalcore.preprocessor._io import _get_attr_from_field_coord, load


def _create_sample_cube():
    coord = DimCoord([1, 2], standard_name="latitude", units="degrees_north")
    cube = Cube([1, 2], var_name="sample", dim_coords_and_dims=((coord, 0),))
    return cube


class TestLoad(unittest.TestCase):
    """Tests for :func:`esmvalcore.preprocessor.load`."""

    def setUp(self):
        """Start tests."""
        self.temp_files = []

    def tearDown(self):
        """Finish tests."""
        for temp_file in self.temp_files:
            os.remove(temp_file)

    def _save_cube(self, cube):
        descriptor, temp_file = tempfile.mkstemp(".nc")
        os.close(descriptor)
        iris.save(cube, temp_file)
        self.temp_files.append(temp_file)
        return temp_file

    def test_load(self):
        """Test loading multiple files."""
        cube = _create_sample_cube()
        temp_file = self._save_cube(cube)

        cubes = load(temp_file)
        cube = cubes[0]
        self.assertEqual(1, len(cubes))
        self.assertEqual(temp_file, cube.attributes["source_file"])
        self.assertTrue((cube.data == np.array([1, 2])).all())
        self.assertTrue(
            (cube.coord("latitude").points == np.array([1, 2])).all()
        )

    def test_load_grib(self):
        """Test loading a grib file."""
        grib_path = (
            Path(importlib_files("tests"))
            / "sample_data"
            / "iris-sample-data"
            / "polar_stereo.grib2"
        )
        cubes = load(grib_path)

        assert len(cubes) == 1
        cube = cubes[0]
        assert cube.standard_name == "air_temperature"
        assert cube.units == "K"
        assert cube.shape == (200, 247)
        assert "source_file" in cube.attributes

    def test_load_cube(self):
        """Test loading an Iris Cube."""
        cube = _create_sample_cube()
        cubes = load(cube)
        assert cubes == CubeList([cube])

    def test_load_cubes(self):
        """Test loading an Iris CubeList."""
        cube = _create_sample_cube()
        cubes = load(CubeList([cube]))
        assert cubes == CubeList([cube])

    def test_load_xarray_dataset(self):
        """Test loading an xarray.Dataset."""
        dataset = xr.Dataset(
            data_vars={"tas": ("time", [1, 2])},
            coords={"time": [0, 1]},
            attrs={"test": 1},
        )

        cubes = load(dataset)

        assert len(cubes) == 1
        cube = cubes[0]
        assert cube.var_name == "tas"
        assert cube.standard_name is None
        assert cube.long_name is None
        assert cube.units == "unknown"
        assert len(cube.coords()) == 1
        assert cube.coords()[0].var_name == "time"
        assert cube.attributes["test"] == 1

    def test_load_ncdata(self):
        """Test loading an xarray.Dataset."""
        dataset = ncdata.NcData(
            dimensions=(ncdata.NcDimension("time", 2),),
            variables=(ncdata.NcVariable("tas", ("time",), [0, 1]),),
        )

        cubes = load(dataset)

        assert len(cubes) == 1
        cube = cubes[0]
        assert cube.var_name == "tas"
        assert cube.standard_name is None
        assert cube.long_name is None
        assert cube.units == "unknown"
        assert not cube.coords()

    def test_callback_fix_lat_units(self):
        """Test callback for fixing units."""
        cube = _create_sample_cube()
        temp_file = self._save_cube(cube)

        cubes = load(temp_file)
        cube = cubes[0]
        self.assertEqual(1, len(cubes))
        self.assertTrue((cube.data == np.array([1, 2])).all())
        self.assertTrue(
            (cube.coord("latitude").points == np.array([1, 2])).all()
        )
        self.assertEqual(cube.coord("latitude").units, "degrees_north")

    def test_get_attr_from_field_coord_none(self):
        """Test ``_get_attr_from_field_coord``."""
        attr = _get_attr_from_field_coord(
            unittest.mock.sentinel.ncfield, None, "attr"
        )
        assert attr is None

    @unittest.mock.patch("iris.load_raw", autospec=True)
    def test_fail_empty_cubes(self, mock_load_raw):
        """Test that ValueError is raised when cubes are empty."""
        mock_load_raw.return_value = CubeList([])
        msg = "Can not load cubes from myfilename"
        with self.assertRaises(ValueError, msg=msg):
            load("myfilename")

    @staticmethod
    def load_with_warning(*_, **__):
        """Mock load with a warning."""
        warnings.warn(
            "This is a custom expected warning",
            category=UserWarning,
            stacklevel=2,
        )
        return CubeList([Cube(0)])

    @unittest.mock.patch("iris.load_raw", autospec=True)
    def test_do_not_ignore_warnings(self, mock_load_raw):
        """Test do not ignore specific warnings."""
        mock_load_raw.side_effect = self.load_with_warning
        ignore_warnings = [{"message": "non-relevant warning"}]

        # Warning is not ignored -> assert warning has been issued
        with self.assertWarns(UserWarning):
            cubes = load("myfilename", ignore_warnings=ignore_warnings)

        # Check output
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].attributes, {"source_file": "myfilename"})

    @unittest.mock.patch("iris.load_raw", autospec=True)
    def test_ignore_warnings(self, mock_load_raw):
        """Test ignore specific warnings."""
        mock_load_raw.side_effect = self.load_with_warning
        ignore_warnings = [{"message": "This is a custom expected warning"}]

        # Warning is ignored -> assert warning has not been issued
        with self.assertRaises(AssertionError):
            with self.assertWarns(UserWarning):
                cubes = load("myfilename", ignore_warnings=ignore_warnings)

        # Check output
        self.assertEqual(len(cubes), 1)
        self.assertEqual(cubes[0].attributes, {"source_file": "myfilename"})
