"""Test add_supplementary_variables and remove_supplementary_variables.

Integration tests for the
:func:`esmvalcore.preprocessor._supplementary_vars` module.
"""

import dask.array as da
import iris
import iris.fileformats
import numpy as np
import pytest

from esmvalcore.preprocessor._supplementary_vars import (
    add_ancillary_variable,
    add_cell_measure,
    add_supplementary_variables,
    find_matching_coord_dims,
    get_data_dims,
    remove_supplementary_variables,
)


class Test:
    """Test class."""

    @pytest.fixture(autouse=True)
    def setUp(self):
        """Assemble a stock cube."""
        fx_area_data = np.ones((3, 3))
        fx_volume_data = np.ones((3, 3, 3))
        self.new_cube_data = np.empty((3, 3))
        self.new_cube_data[:] = 200.0
        self.new_cube_3D_data = np.empty((3, 3, 3))
        self.new_cube_3D_data[:] = 200.0
        crd_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        self.lons = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="longitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_east",
            coord_system=crd_sys,
        )
        self.lats = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="latitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_north",
            coord_system=crd_sys,
        )
        self.depth = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="depth",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="m",
            long_name="ocean depth coordinate",
        )
        self.monthly_times = iris.coords.DimCoord(
            [
                15.5,
                45,
                74.5,
                105,
                135.5,
                166,
                196.5,
                227.5,
                258,
                288.5,
                319,
                349.5,
            ],
            standard_name="time",
            var_name="time",
            bounds=[
                [0, 31],
                [31, 59],
                [59, 90],
                [90, 120],
                [120, 151],
                [151, 181],
                [181, 212],
                [212, 243],
                [243, 273],
                [273, 304],
                [304, 334],
                [334, 365],
            ],
            units="days since 1950-01-01 00:00:00",
        )
        self.yearly_times = iris.coords.DimCoord(
            [182.5, 547.5],
            standard_name="time",
            bounds=[[0, 365], [365, 730]],
            units="days since 1950-01-01 00:00",
        )
        self.coords_spec = [(self.lats, 0), (self.lons, 1)]
        self.fx_area = iris.cube.Cube(
            fx_area_data,
            dim_coords_and_dims=self.coords_spec,
        )
        self.fx_volume = iris.cube.Cube(
            fx_volume_data,
            dim_coords_and_dims=[
                (self.depth, 0),
                (self.lats, 1),
                (self.lons, 2),
            ],
        )
        self.monthly_volume = iris.cube.Cube(
            np.ones((12, 3, 3, 3)),
            dim_coords_and_dims=[
                (self.monthly_times, 0),
                (self.depth, 1),
                (self.lats, 2),
                (self.lons, 3),
            ],
        )
        self.cube = iris.cube.Cube(
            self.new_cube_3D_data,
            dim_coords_and_dims=[
                (self.depth, 0),
                (self.lats, 1),
                (self.lons, 2),
            ],
        )
        self.plev = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="air_pressure",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="Pa",
        )
        self.lats_no_metadata = iris.coords.DimCoord(
            [0, 1.5, 3],
            standard_name="latitude",
            bounds=[[0, 1], [1, 2], [2, 3]],
            units="degrees_north",
        )
        self.ancillary_cube_plev = iris.cube.Cube(
            np.ones(3),
            dim_coords_and_dims=[(self.plev, 0)],
        )
        self.ancillary_cube_lat_plev = iris.cube.Cube(
            np.ones((3, 3)),
            dim_coords_and_dims=[(self.lats, 0), (self.plev, 1)],
        )
        self.ancillary_cube_lat_lon = iris.cube.Cube(
            np.ones((3, 3)),
            dim_coords_and_dims=[(self.lats, 0), (self.lons, 1)],
        )
        self.ancillary_cube_lat_no_metadata = iris.cube.Cube(
            np.ones(3),
            dim_coords_and_dims=[(self.lats_no_metadata, 0)],
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("var_name", ["areacella", "areacello"])
    def test_add_cell_measure_area(self, var_name, lazy):
        """Test add area fx variables as cell measures."""
        if lazy:
            self.fx_area.data = self.fx_area.lazy_data()
            self.new_cube_data = da.array(self.new_cube_data).rechunk((1, 2))
        self.fx_area.var_name = var_name
        self.fx_area.standard_name = "cell_area"
        self.fx_area.units = "m2"
        cube = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.coords_spec,
        )

        cube = add_supplementary_variables(cube, [self.fx_area])

        assert cube.has_lazy_data() is lazy
        assert cube.cell_measures(self.fx_area.standard_name)
        cell_measure = cube.cell_measure(self.fx_area.standard_name)
        assert cell_measure.has_lazy_data() is lazy
        if lazy:
            assert cell_measure.lazy_data().chunks == cube.lazy_data().chunks

    @pytest.mark.parametrize("lazy", [True, False])
    def test_add_cell_measure_volume(self, lazy):
        """Test add volume as cell measure."""
        if lazy:
            self.fx_volume.data = self.fx_volume.lazy_data()
            self.new_cube_3D_data = da.array(self.new_cube_3D_data).rechunk(
                (1, 2, 3),
            )
        self.fx_volume.var_name = "volcello"
        self.fx_volume.standard_name = "ocean_volume"
        self.fx_volume.units = "m3"
        cube = iris.cube.Cube(
            self.new_cube_3D_data,
            dim_coords_and_dims=[
                (self.depth, 0),
                (self.lats, 1),
                (self.lons, 2),
            ],
        )

        cube = add_supplementary_variables(cube, [self.fx_volume])

        assert cube.has_lazy_data() is lazy
        assert cube.cell_measures(self.fx_volume.standard_name)
        cell_measure = cube.cell_measure(self.fx_volume.standard_name)
        assert cell_measure.has_lazy_data() is lazy
        if lazy:
            assert cell_measure.lazy_data().chunks == cube.lazy_data().chunks

    def test_no_cell_measure(self):
        """Test no cell measure is added."""
        cube = iris.cube.Cube(
            self.new_cube_3D_data,
            dim_coords_and_dims=[
                (self.depth, 0),
                (self.lats, 1),
                (self.lons, 2),
            ],
        )
        cube = add_supplementary_variables(cube, [])
        assert cube.cell_measures() == []

    @pytest.mark.parametrize("lazy", [True, False])
    def test_add_ancillary_vars(self, lazy):
        """Test adding ancillary variables."""
        if lazy:
            self.fx_area.data = self.fx_area.lazy_data()
            self.new_cube_data = da.array(self.new_cube_data).rechunk((1, 2))
        self.fx_area.var_name = "sftlf"
        self.fx_area.standard_name = "land_area_fraction"
        self.fx_area.units = "%"
        cube = iris.cube.Cube(
            self.new_cube_data,
            dim_coords_and_dims=self.coords_spec,
        )

        cube = add_supplementary_variables(cube, [self.fx_area])

        assert cube.has_lazy_data() is lazy
        assert cube.ancillary_variables(self.fx_area.standard_name)
        anc_var = cube.ancillary_variable(self.fx_area.standard_name)
        assert anc_var.has_lazy_data() is lazy
        if lazy:
            assert anc_var.lazy_data().chunks == cube.lazy_data().chunks

    def test_wrong_shape(self, monkeypatch):
        """Test variable is not added if it's not broadcastable to cube."""
        volume_data = np.ones((2, 3, 3, 3))
        volume_cube = iris.cube.Cube(
            volume_data,
            dim_coords_and_dims=[
                (self.yearly_times, 0),
                (self.depth, 1),
                (self.lats, 2),
                (self.lons, 3),
            ],
        )
        volume_cube.standard_name = "ocean_volume"
        volume_cube.var_name = "volcello"
        volume_cube.units = "m3"
        data = np.ones((12, 3, 3, 3))
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (self.monthly_times, 0),
                (self.depth, 1),
                (self.lats, 2),
                (self.lons, 3),
            ],
        )
        cube.var_name = "thetao"
        with pytest.raises(iris.exceptions.CannotAddError):
            add_supplementary_variables(cube, [volume_cube])

    def test_remove_supplementary_vars(self):
        """Test supplementary variables are removed from cube."""
        cube = iris.cube.Cube(
            self.new_cube_3D_data,
            dim_coords_and_dims=[
                (self.depth, 0),
                (self.lats, 1),
                (self.lons, 2),
            ],
        )
        self.fx_area.var_name = "areacella"
        self.fx_area.standard_name = "cell_area"
        self.fx_area.units = "m2"
        add_cell_measure(cube, self.fx_area, measure="area")
        assert cube.cell_measure(self.fx_area.standard_name) is not None
        self.fx_area.var_name = "sftlf"
        self.fx_area.standard_name = "land_area_fraction"
        self.fx_area.units = "%"
        add_ancillary_variable(cube, self.fx_area)
        assert cube.ancillary_variable(self.fx_area.standard_name) is not None
        cube = remove_supplementary_variables(cube)
        assert cube.cell_measures() == []
        assert cube.ancillary_variables() == []

    def test_add_ancillary_vars_errors(self):
        """Test errors when adding ancillary variable."""
        # Ancillary var not an iris.cube.Cube or iris.coords.AncillaryVariable
        msg = "ancillary_cube should be either an iris"
        with pytest.raises(ValueError, match=msg):
            add_ancillary_variable(
                self.cube,
                np.ones(self.new_cube_3D_data.shape),
            )
        # Ancillary var as iris.cube.Cube without matching dimensions
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            add_ancillary_variable(
                self.cube,
                self.ancillary_cube_plev,
            )

    def test_get_data_dims_no_match(self):
        """Test get_data_dims matching function w/ no match."""
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            get_data_dims(
                self.cube,
                self.ancillary_cube_plev,
            )

    def test_get_data_dims_one_match(self):
        """Test get_data_dims matching function w/ only one coordinate match."""
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            _ = get_data_dims(
                self.cube,
                self.ancillary_cube_lat_plev,
            )

    def test_get_data_dims_match(self):
        """Test get_data_dims matching function w/ both coordinates match."""
        assert get_data_dims(
            self.cube,
            self.ancillary_cube_lat_lon,
        ) == [1, 2]

    def test_get_data_dims_match_no_metadata(self):
        """Test get_data_dims matching function w/ coordinate w/o metadata."""
        assert get_data_dims(
            self.cube,
            self.ancillary_cube_lat_no_metadata,
        ) == [1]

    def test_find_matching_coord_dims_no_match(self):
        """Test find_matching_coord_dims function w/ no match."""
        assert (
            find_matching_coord_dims(
                self.plev,
                self.cube,
            )
            is None
        )

    def test_find_matching_coord_dims_match(self):
        """Test find_matching_coord_dims function w/ match."""
        assert find_matching_coord_dims(
            self.lats,
            self.cube,
        ) == (1,)
