"""Tests for the ACCESS-ESM on-the-fly CMORizer."""

from pathlib import Path

import dask.array as da
import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.cmor._fixes.access.access_esm1_5
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import CoordinateInfo, get_var_info
from esmvalcore.dataset import Dataset

time_coord = DimCoord(
    [15, 45],
    standard_name="time",
    var_name="time",
    units=Unit("days since 1851-01-01", calendar="noleap"),
    attributes={"test": 1, "time_origin": "will_be_removed"},
)

time_ocn_coord = DimCoord(
    list(range(1, 13)),
    standard_name="time",
    var_name="time",
    long_name="time",
    units=Unit("days since 0000-01-01", calendar="noleap"),
    attributes={"calendar_type": "GREGORIAN", "cartesian_axis": "T"},
)

lat_ocn_coord = DimCoord(
    np.linspace(-90, 210, 300),
    standard_name="latitude",
    long_name="tcell latitude",
    var_name="yt_ocean",
    units="degrees_N",
    attributes={
        "cartesian_axis": "Y",
    },
)

lon_ocn_coord = DimCoord(
    np.linspace(-90, 270, 360),
    standard_name="longitude",
    long_name="tcell longitude",
    var_name="xt_ocean",
    units="degrees_E",
    attributes={
        "cartesian_axis": "X",
    },
)

depth_ocn_coord = DimCoord(
    [0, 1],
    long_name="tcell zstar depth",
    var_name="st_ocean",
    units="meter",
    attributes={
        "cartesian_axis": "Z",
        "edges": "st_edges_ocean",
        "positive": "down",
    },
)

lat_ocn_aux_coord = AuxCoord(
    np.tile(
        np.concatenate(
            (np.linspace(80.5, 359.5, 280), np.linspace(0.5, 79.5, 80)),
        ),
        (300, 1),
    ),
    standard_name="latitude",
    long_name="tracer latitude",
    var_name="geolat_t",
    attributes={
        "valid_range": "[-91. 91]",
    },
)

lon_ocn_aux_coord = AuxCoord(
    np.tile(
        np.concatenate(
            (np.linspace(80.5, 359.5, 280), np.linspace(0.5, 79.5, 80)),
        ),
        (300, 1),
    ),
    standard_name="longitude",
    long_name="tracer longitude",
    var_name="geolon_t",
    attributes={
        "valid_range": "[-281. 361]",
    },
)

lat_coord = DimCoord(
    [0, 10],
    standard_name="latitude",
    var_name="lat",
    units="degrees",
)
lon_coord = DimCoord(
    [-180, 0],
    standard_name="longitude",
    var_name="lon",
    units="degrees",
)
coord_spec_3d = [
    (time_coord, 0),
    (lat_coord, 1),
    (lon_coord, 2),
]


@pytest.fixture
def cubes_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / "access_native.nc"
    return iris.load(str(nc_path))


def _get_fix(mip, frequency, short_name, fix_name):
    """Load a fix from :mod:`esmvalcore.cmor._fixes.access.access_esm1_5`."""
    dataset = Dataset(
        project="ACCESS",
        dataset="ACCESS-ESM1-5",
        mip=mip,
        short_name=short_name,
    )
    extra_facets = dataset._get_extra_facets()
    extra_facets["frequency"] = frequency
    extra_facets["exp"] = "amip"
    vardef = get_var_info(project="ACCESS", mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.access.access_esm1_5, fix_name)
    return cls(vardef, extra_facets=extra_facets, session={}, frequency="")


def get_fix(mip, frequency, short_name):
    """Load a variable fix from esmvalcore.cmor._fixes.access.access_esm1_5."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, frequency, short_name, fix_name)


def get_fix_allvar(mip, frequency, short_name):
    """Load a AllVar fix from esmvalcore.cmor._fixes.access.access_esm1_5."""
    return _get_fix(mip, frequency, short_name, "AllVars")


def fix_metadata(cubes, mip, frequency, short_name):
    """Fix metadata of cubes."""
    fix = get_fix(mip, frequency, short_name)
    return fix.fix_metadata(cubes)


def check_tas_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tas"
    assert cube.standard_name == "air_temperature"
    assert cube.long_name == "Near-Surface Air Temperature"
    assert cube.units == "K"
    assert "positive" not in cube.attributes
    return cube


def check_tos_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "tos"
    assert cube.standard_name == "sea_surface_temperature"
    assert cube.long_name == "Sea Surface Temperature"
    assert cube.units == "degC"
    return cube


def check_so_metadata(cubes):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "so"
    assert cube.standard_name == "sea_water_salinity"
    assert cube.long_name == "Sea Water Salinity"
    assert cube.units == Unit(0.001)
    return cube


def check_pr_metadata(cubes):
    """Check pr metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "pr"
    assert cube.standard_name == "precipitation_flux"
    assert cube.long_name == "Precipitation"
    assert cube.units == "kg m-2 s-1"
    assert "positive" not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords("time", dim_coords=True)
    time = cube.coord("time", dim_coords=True)
    assert time.var_name == "time"
    assert time.standard_name == "time"
    assert time.bounds.shape == (1, 2)
    assert time.attributes == {}


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords("latitude", dim_coords=True)
    lat = cube.coord("latitude", dim_coords=True)
    assert lat.var_name == "lat"
    assert lat.standard_name == "latitude"
    assert lat.units == "degrees_north"
    assert lat.attributes == {}


def check_ocn_lat(cube):
    """Check latitude coordinate of ocean variable cube."""


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords("longitude", dim_coords=True)
    lon = cube.coord("longitude", dim_coords=True)
    assert lon.var_name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.units == "degrees_east"
    assert lon.attributes == {}


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords("height")
    height = cube.coord("height")
    assert height.var_name == "height"
    assert height.standard_name == "height"
    assert height.units == "m"
    assert height.attributes == {"positive": "up"}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def check_ocean_dim_coords(cube):
    """Check dim_coords of ocean variables."""
    assert (cube.dim_coords[-2].points == np.arange(300)).all()
    assert cube.dim_coords[-2].standard_name is None
    assert cube.dim_coords[-2].var_name == "j"
    assert cube.dim_coords[-2].long_name == "cell index along second dimension"
    assert cube.dim_coords[-2].attributes == {}

    assert (cube.dim_coords[-1].points == np.arange(360)).all()
    assert cube.dim_coords[-1].standard_name is None
    assert cube.dim_coords[-1].var_name == "i"
    assert cube.dim_coords[-1].long_name == "cell index along first dimension"
    assert cube.dim_coords[-1].attributes == {}


def check_ocean_aux_coords(cube):
    """Check aux_coords of ocean variables."""
    assert cube.aux_coords[-2].shape == (300, 360)
    assert cube.aux_coords[-2].dtype == np.dtype("float64")
    assert cube.aux_coords[-2].standard_name == "latitude"
    assert cube.aux_coords[-2].long_name == "latitude"
    assert cube.aux_coords[-2].var_name == "lat"
    assert cube.aux_coords[-2].attributes == {}

    assert cube.aux_coords[-1].shape == (300, 360)
    assert (cube.aux_coords[-1].points < 360).all()
    assert (cube.aux_coords[-1].points > 0).all()
    assert cube.aux_coords[-1].standard_name == "longitude"
    assert cube.aux_coords[-1].long_name == "longitude"
    assert cube.aux_coords[-1].var_name == "lon"
    assert cube.aux_coords[-1].attributes == {}


def assert_plev_metadata(cube):
    """Assert plev metadata is correct."""
    assert cube.coord("air_pressure").standard_name == "air_pressure"
    assert cube.coord("air_pressure").var_name == "plev"
    assert cube.coord("air_pressure").units == "Pa"
    assert cube.coord("air_pressure").attributes == {"positive": "down"}


def test_only_time(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix_allvar("Amon", "mon", "pr")

    coord_info = CoordinateInfo("time")
    coord_info.standard_name = "time"
    monkeypatch.setattr(fix.vardef, "coordinates", {"time": coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_pr_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check time metadata
    assert cube.coords("time")
    new_time_coord = cube.coord("time", dim_coords=True)
    assert new_time_coord.var_name == "time"
    assert new_time_coord.standard_name == "time"


def test_only_latitude(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix_allvar("Amon", "mon", "pr")

    coord_info = CoordinateInfo("latitude")
    coord_info.standard_name = "latitude"
    monkeypatch.setattr(fix.vardef, "coordinates", {"latitude": coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_pr_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check latitude metadata
    assert cube.coords("latitude", dim_coords=True)
    new_lat_coord = cube.coord("latitude")
    assert new_lat_coord.var_name == "lat"
    assert new_lat_coord.standard_name == "latitude"
    assert new_lat_coord.units == "degrees_north"


def test_only_longitude(monkeypatch, cubes_2d):
    """Test fix."""
    fix = get_fix_allvar("Amon", "mon", "pr")

    coord_info = CoordinateInfo("longitude")
    coord_info.standard_name = "longitude"
    monkeypatch.setattr(fix.vardef, "coordinates", {"longitude": coord_info})

    cubes = cubes_2d
    fixed_cubes = fix.fix_metadata(cubes)

    # Check cube metadata
    cube = check_pr_metadata(fixed_cubes)

    # Check cube data
    assert cube.shape == (1, 145, 192)

    # Check longitude metadata
    assert cube.coords("longitude", dim_coords=True)
    new_lon_coord = cube.coord("longitude")
    assert new_lon_coord.var_name == "lon"
    assert new_lon_coord.standard_name == "longitude"
    assert new_lon_coord.units == "degrees_east"


def test_get_tas_fix():
    """Test getting of fix 'tas'."""
    fix = Fix.get_fixes("ACCESS", "ACCESS_ESM1_5", "Amon", "tas")
    assert fix == [
        esmvalcore.cmor._fixes.access.access_esm1_5.Tas(
            vardef={},
            extra_facets={},
            session={},
            frequency="",
        ),
        esmvalcore.cmor._fixes.access.access_esm1_5.AllVars(
            vardef={},
            extra_facets={},
            session={},
            frequency="",
        ),
        GenericFix(None),
    ]


def test_tas_fix(cubes_2d):
    """Test fix 'tas'."""
    fix_tas = get_fix("Amon", "mon", "tas")
    fix_allvar = get_fix_allvar("Amon", "mon", "tas")
    fixed_cubes = fix_tas.fix_metadata(cubes_2d)
    fixed_cubes = fix_allvar.fix_metadata(fixed_cubes)
    fixed_cube = check_tas_metadata(fixed_cubes)

    check_time(fixed_cube)
    check_lat(fixed_cube)
    check_lon(fixed_cube)
    check_heightxm(fixed_cube, 2)

    assert fixed_cube.shape == (1, 145, 192)


def test_hus_fix():
    """Test fix 'hus'."""
    time_coord = DimCoord(
        [15, 45],
        standard_name="time",
        var_name="time",
        units=Unit("days since 1851-01-01", calendar="noleap"),
        attributes={"test": 1, "time_origin": "will_be_removed"},
    )
    plev_coord_rev = DimCoord(
        [250, 500, 850],
        var_name="pressure",
        units="Pa",
    )
    lat_coord_rev = DimCoord(
        [10, -10],
        standard_name="latitude",
        var_name="lat",
        units="degrees",
    )
    lon_coord = DimCoord(
        [-180, 0],
        standard_name="longitude",
        var_name="lon",
        units="degrees",
    )
    coord_spec_4d = [
        (time_coord, 0),
        (plev_coord_rev, 1),
        (lat_coord_rev, 2),
        (lon_coord, 3),
    ]
    cube_4d = Cube(
        da.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2),
        standard_name="specific_humidity",
        long_name="Specific Humidity",
        var_name="fld_s30i205",
        units="1",
        dim_coords_and_dims=coord_spec_4d,
        attributes={},
    )
    cubes_4d = CubeList([cube_4d])

    fix = get_fix_allvar("Amon", "mon", "hus")
    fixed_cubes = fix.fix_metadata(cubes_4d)
    fixed_cube = fixed_cubes[0]
    assert_plev_metadata(fixed_cube)

    assert fixed_cube.shape == (2, 3, 2, 2)


def test_rsus_fix():
    """Test fix 'rsus'."""
    time_coord = DimCoord(
        [15, 45],
        standard_name="time",
        var_name="time",
        units=Unit("days since 1851-01-01", calendar="noleap"),
        attributes={"test": 1, "time_origin": "will_be_removed"},
    )
    lat_coord = DimCoord(
        [0, 10],
        standard_name="latitude",
        var_name="lat",
        units="degrees",
    )
    lon_coord = DimCoord(
        [-180, 0],
        standard_name="longitude",
        var_name="lon",
        units="degrees",
    )
    coord_spec_3d = [
        (time_coord, 0),
        (lat_coord, 1),
        (lon_coord, 2),
    ]
    cube_3d_1 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s01i235",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )
    cube_3d_2 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s01i201",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )
    cubes_3d = CubeList([cube_3d_1, cube_3d_2])

    cube_result = cubes_3d[0] - cubes_3d[1]

    fix = get_fix("Amon", "mon", "rsus")
    fixed_cubes = fix.fix_metadata(cubes_3d)
    np.testing.assert_allclose(fixed_cubes[0].data, cube_result.data)


def test_rlus_fix():
    """Test fix 'rlus'."""
    cube_3d_1 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s02i207",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )
    cube_3d_2 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s02i201",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )
    cube_3d_3 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s03i332",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )
    cube_3d_4 = Cube(
        da.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        var_name="fld_s02i205",
        units="W m-2",
        dim_coords_and_dims=coord_spec_3d,
        attributes={},
    )

    cubes_3d = CubeList([cube_3d_1, cube_3d_2, cube_3d_3, cube_3d_4])

    cube_result = cubes_3d[0] - cubes_3d[1] + cubes_3d[2] - cubes_3d[3]

    fix = get_fix("Amon", "mon", "rlus")
    fixed_cubes = fix.fix_metadata(cubes_3d)
    np.testing.assert_allclose(fixed_cubes[0].data, cube_result.data)


def test_tos_fix(test_data_path):
    """Test fix 'tos'."""
    coord_dim = [
        (time_ocn_coord, 0),
        (lat_ocn_coord, 1),
        (lon_ocn_coord, 2),
    ]

    coord_aux = [
        (lat_ocn_aux_coord, (1, 2)),
        (lon_ocn_aux_coord, (1, 2)),
    ]

    cube_tos = Cube(
        da.arange(12 * 300 * 360, dtype=np.float32).reshape(12, 300, 360),
        var_name="sst",
        units=Unit("degrees K"),
        dim_coords_and_dims=coord_dim,
        aux_coords_and_dims=coord_aux,
        attributes={},
    )

    grid_path = f"{test_data_path}/access_ocean_grid.nc"
    cubes_tos = CubeList([cube_tos])
    fix_tos = get_fix("Omon", "mon", "tos")
    fix_allvar = get_fix_allvar("Omon", "mon", "tos")
    fix_tos.extra_facets["ocean_grid_path"] = grid_path
    fixed_cubes = fix_tos.fix_metadata(cubes_tos)
    fixed_cubes = fix_allvar.fix_metadata(fixed_cubes)
    fixed_cube = check_tos_metadata(fixed_cubes)

    check_ocean_dim_coords(fixed_cube)
    check_ocean_aux_coords(fixed_cube)
    assert fixed_cube.shape == (12, 300, 360)


def test_so_fix(test_data_path):
    """Test fix 'so'."""
    coord_dim = [
        (time_ocn_coord, 0),
        (depth_ocn_coord, 1),
        (lat_ocn_coord, 2),
        (lon_ocn_coord, 3),
    ]

    coord_aux = [
        (lat_ocn_aux_coord, (2, 3)),
        (lon_ocn_aux_coord, (2, 3)),
    ]

    cube_so = Cube(
        da.arange(12 * 2 * 300 * 360, dtype=np.float32).reshape(
            12,
            2,
            300,
            360,
        ),
        var_name="salt",
        units="unknown",
        dim_coords_and_dims=coord_dim,
        aux_coords_and_dims=coord_aux,
        attributes={
            "invalid_units": "psu",
        },
    )

    grid_path = f"{test_data_path}/access_ocean_grid.nc"
    facet = "ocean_grid_path"
    cubes_so = CubeList([cube_so])
    fix_so = get_fix("Omon", "mon", "so")
    fix_allvar = get_fix_allvar("Omon", "mon", "so")
    fix_so.extra_facets[facet] = grid_path
    fixed_cubes = fix_so.fix_metadata(cubes_so)
    fixed_cubes = fix_allvar.fix_metadata(fixed_cubes)
    fixed_cube = check_so_metadata(fixed_cubes)

    check_ocean_dim_coords(fixed_cube)
    check_ocean_aux_coords(fixed_cube)
    assert fixed_cube.shape == (12, 2, 300, 360)


def test_get_path_from_facet_false(test_data_path):
    """Test get_path_from_facet."""
    facet = "ocean_grid_path"
    fix = get_fix("Omon", "mon", "so")
    fix.extra_facets[facet] = test_data_path
    test_filepath = Path(fix.extra_facets[facet])
    msg = f"'{test_filepath}' given by facet '{facet}' does not exist"

    with pytest.raises(FileNotFoundError, match=msg):
        fix._get_path_from_facet(facet)
