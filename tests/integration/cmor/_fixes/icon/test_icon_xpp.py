"""Test the ICON-XPP on-the-fly CMORizer."""

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

import esmvalcore.cmor._fixes.icon.icon_xpp
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor._fixes.icon._base_fixes import AllVarsBase, IconFix
from esmvalcore.cmor._fixes.icon.icon_xpp import (
    AllVars,
    Clwvi,
    Evspsbl,
    Gpp,
    Hfls,
    Hfss,
    Rlut,
    Rlutcs,
    Rsutcs,
    Rtmt,
    Rtnt,
    Zg,
)
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info
from esmvalcore.dataset import Dataset


@pytest.fixture(autouse=True)
def tmp_cache_dir(monkeypatch, tmp_path):
    """Use temporary path as cache directory for all tests in this module."""
    monkeypatch.setattr(IconFix, "CACHE_DIR", tmp_path)


@pytest.fixture
def cubes_atm_2d(test_data_path):
    """2D sample cubes."""
    nc_path = test_data_path / "icon_xpp_atm_2d.nc"
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_atm_3d(test_data_path):
    """3D sample cubes."""
    nc_path = test_data_path / "icon_xpp_atm_3d.nc"
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_ocean_3d(test_data_path):
    """3D sample cubes."""
    nc_path = test_data_path / "icon_xpp_ocean_3d.nc"
    return iris.load(str(nc_path))


@pytest.fixture
def cubes_regular_grid():
    """Cube with regular grid."""
    time_coord = DimCoord(
        [0],
        var_name="time",
        standard_name="time",
        units="days since 1850-01-01",
    )
    lat_coord = DimCoord(
        [0.0, 1.0],
        var_name="clat",
        standard_name="latitude",
        long_name="latitude",
        units="degrees_north",
    )
    lon_coord = DimCoord(
        [-1.0, 1.0],
        var_name="clon",
        standard_name="longitude",
        long_name="longitude",
        units="degrees_east",
    )
    cube = Cube(
        [[[0.0, 1.0], [2.0, 3.0]]],
        var_name="t_2m",
        units="K",
        dim_coords_and_dims=[(time_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )
    return CubeList([cube])


def _get_fix(mip, short_name, fix_name, session=None):
    """Load a fix from esmvalcore.cmor._fixes.icon.icon_xpp."""
    dataset = Dataset(
        project="ICON",
        dataset="ICON-XPP",
        mip=mip,
        short_name=short_name,
    )
    extra_facets = dataset._get_extra_facets()
    extra_facets["frequency"] = "mon"
    extra_facets["exp"] = "amip"
    vardef = get_var_info(project="ICON", mip=mip, short_name=short_name)
    cls = getattr(esmvalcore.cmor._fixes.icon.icon_xpp, fix_name)
    return cls(vardef, extra_facets=extra_facets, session=session)


def get_fix(mip, short_name, session=None):
    """Load a variable fix from esmvalcore.cmor._fixes.icon.icon_xpp."""
    fix_name = short_name[0].upper() + short_name[1:]
    return _get_fix(mip, short_name, fix_name, session=session)


def get_allvars_fix(mip, short_name, session=None):
    """Load the AllVars fix from esmvalcore.cmor._fixes.icon.icon_xpp."""
    return _get_fix(mip, short_name, "AllVars", session=session)


def fix_metadata(cubes, mip, short_name, session=None):
    """Fix metadata of cubes."""
    fix = get_fix(mip, short_name, session=session)
    cubes = fix.fix_metadata(cubes)
    fix = get_allvars_fix(mip, short_name, session=session)
    return fix.fix_metadata(cubes)


def fix_data(cube, mip, short_name, session=None):
    """Fix data of cube."""
    fix = get_fix(mip, short_name, session=session)
    cube = fix.fix_data(cube)
    fix = get_allvars_fix(mip, short_name, session=session)
    return fix.fix_data(cube)


def check_ta_metadata(cubes):
    """Check ta metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == "ta"
    assert cube.standard_name == "air_temperature"
    assert cube.long_name == "Air Temperature"
    assert cube.units == "K"
    assert "positive" not in cube.attributes
    return cube


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


def check_siconc_metadata(cubes, var_name, long_name):
    """Check tas metadata."""
    assert len(cubes) == 1
    cube = cubes[0]
    assert cube.var_name == var_name
    assert cube.standard_name == "sea_ice_area_fraction"
    assert cube.long_name == long_name
    assert cube.units == "%"
    assert "positive" not in cube.attributes
    return cube


def check_time(cube):
    """Check time coordinate of cube."""
    assert cube.coords("time", dim_coords=True)
    time = cube.coord("time", dim_coords=True)
    assert time.var_name == "time"
    assert time.standard_name == "time"
    assert time.long_name == "time"
    assert time.units == Unit(
        "days since 1850-01-01",
        calendar="proleptic_gregorian",
    )
    np.testing.assert_allclose(time.points, [54770.5])
    np.testing.assert_allclose(time.bounds, [[54755.0, 54786.0]])
    assert time.attributes == {}


def check_model_level_metadata(cube):
    """Check metadata of model_level coordinate."""
    assert cube.coords("model level number", dim_coords=True)
    height = cube.coord("model level number", dim_coords=True)
    assert height.var_name == "model_level"
    assert height.standard_name is None
    assert height.long_name == "model level number"
    assert height.units == "no unit"
    assert height.attributes == {"positive": "up"}
    return height


def check_air_pressure_metadata(cube):
    """Check metadata of air_pressure coordinate."""
    assert cube.coords("air_pressure", dim_coords=False)
    plev = cube.coord("air_pressure", dim_coords=False)
    assert plev.var_name == "plev"
    assert plev.standard_name == "air_pressure"
    assert plev.long_name == "pressure"
    assert plev.units == "Pa"
    assert plev.attributes == {"positive": "down"}
    return plev


def check_height(cube):
    """Check height coordinate of cube."""
    height = check_model_level_metadata(cube)
    np.testing.assert_array_equal(height.points, np.arange(47))
    assert height.bounds is None

    plev = check_air_pressure_metadata(cube)
    assert cube.coord_dims("air_pressure") == (0, 1, 2)

    np.testing.assert_allclose(
        plev.points[0, :4, 0],
        [100566.234, 99652.07, 97995.77, 95686.08],
    )
    assert plev.bounds is None


def check_heightxm(cube, height_value):
    """Check scalar heightxm coordinate of cube."""
    assert cube.coords("height")
    height = cube.coord("height")
    assert height.var_name == "height"
    assert height.standard_name == "height"
    assert height.long_name == "height"
    assert height.units == "m"
    assert height.attributes == {"positive": "up"}
    np.testing.assert_allclose(height.points, [height_value])
    assert height.bounds is None


def check_lat(cube):
    """Check latitude coordinate of cube."""
    assert cube.coords("latitude", dim_coords=False)
    lat = cube.coord("latitude", dim_coords=False)
    assert lat.var_name == "lat"
    assert lat.standard_name == "latitude"
    assert lat.long_name == "latitude"
    assert lat.units == "degrees_north"
    assert lat.attributes == {}
    np.testing.assert_allclose(
        lat.points,
        [-45.0, -45.0, -45.0, -45.0, 45.0, 45.0, 45.0, 45.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        lat.bounds,
        [
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
        ],
        rtol=1e-5,
    )
    return lat


def check_lon(cube):
    """Check longitude coordinate of cube."""
    assert cube.coords("longitude", dim_coords=False)
    lon = cube.coord("longitude", dim_coords=False)
    assert lon.var_name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.long_name == "longitude"
    assert lon.units == "degrees_east"
    assert lon.attributes == {}
    np.testing.assert_allclose(
        lon.points,
        [225.0, 315.0, 45.0, 135.0, 225.0, 315.0, 45.0, 135.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        lon.bounds,
        [
            [0.0, 270.0, 180.0],
            [0.0, 0.0, 270.0],
            [0.0, 90.0, 0.0],
            [0.0, 180.0, 90.0],
            [180.0, 270.0, 0.0],
            [270.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [90.0, 180.0, 0.0],
        ],
        rtol=1e-5,
    )
    return lon


def check_lat_lon(cube):
    """Check latitude, longitude and mesh of cube."""
    lat = check_lat(cube)
    lon = check_lon(cube)

    # Check that latitude and longitude are mesh coordinates
    assert cube.coords("latitude", mesh_coords=True)
    assert cube.coords("longitude", mesh_coords=True)

    # Check dimensional coordinate describing the mesh
    assert cube.coords(
        "first spatial index for variables stored on an unstructured grid",
        dim_coords=True,
    )
    i_coord = cube.coord(
        "first spatial index for variables stored on an unstructured grid",
        dim_coords=True,
    )
    assert i_coord.var_name == "i"
    assert i_coord.standard_name is None
    assert i_coord.long_name == (
        "first spatial index for variables stored on an unstructured grid"
    )
    assert i_coord.units == "1"
    np.testing.assert_allclose(i_coord.points, [0, 1, 2, 3, 4, 5, 6, 7])
    assert i_coord.bounds is None

    assert len(cube.coord_dims(lat)) == 1
    assert cube.coord_dims(lat) == cube.coord_dims(lon)
    assert cube.coord_dims(lat) == cube.coord_dims(i_coord)

    # Check the mesh itself
    assert cube.location == "face"
    mesh = cube.mesh
    check_mesh(mesh)


def check_mesh(mesh):  # noqa: PLR0915
    """Check the mesh."""
    assert mesh is not None
    assert mesh.var_name is None
    assert mesh.standard_name is None
    assert mesh.long_name is None
    assert mesh.units == "unknown"
    assert mesh.attributes == {}
    assert mesh.cf_role == "mesh_topology"
    assert mesh.topology_dimension == 2

    # Check face coordinates
    assert len(mesh.coords(location="face")) == 2

    mesh_face_lat = mesh.coord(location="face", axis="y")
    assert mesh_face_lat.var_name == "lat"
    assert mesh_face_lat.standard_name == "latitude"
    assert mesh_face_lat.long_name == "latitude"
    assert mesh_face_lat.units == "degrees_north"
    assert mesh_face_lat.attributes == {}
    np.testing.assert_allclose(
        mesh_face_lat.points,
        [-45.0, -45.0, -45.0, -45.0, 45.0, 45.0, 45.0, 45.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        mesh_face_lat.bounds,
        [
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [-90.0, 0.0, 0.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 90.0],
        ],
        rtol=1e-5,
    )

    mesh_face_lon = mesh.coord(location="face", axis="x")
    assert mesh_face_lon.var_name == "lon"
    assert mesh_face_lon.standard_name == "longitude"
    assert mesh_face_lon.long_name == "longitude"
    assert mesh_face_lon.units == "degrees_east"
    assert mesh_face_lon.attributes == {}
    np.testing.assert_allclose(
        mesh_face_lon.points,
        [225.0, 315.0, 45.0, 135.0, 225.0, 315.0, 45.0, 135.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        mesh_face_lon.bounds,
        [
            [0.0, 270.0, 180.0],
            [0.0, 0.0, 270.0],
            [0.0, 90.0, 0.0],
            [0.0, 180.0, 90.0],
            [180.0, 270.0, 0.0],
            [270.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
            [90.0, 180.0, 0.0],
        ],
        rtol=1e-5,
    )

    # Check node coordinates
    assert len(mesh.coords(location="node")) == 2

    mesh_node_lat = mesh.coord(location="node", axis="y")
    assert mesh_node_lat.var_name == "nlat"
    assert mesh_node_lat.standard_name == "latitude"
    assert mesh_node_lat.long_name == "node latitude"
    assert mesh_node_lat.units == "degrees_north"
    assert mesh_node_lat.attributes == {}
    np.testing.assert_allclose(
        mesh_node_lat.points,
        [-90.0, 0.0, 0.0, 0.0, 0.0, 90.0],
        rtol=1e-5,
    )
    assert mesh_node_lat.bounds is None

    mesh_node_lon = mesh.coord(location="node", axis="x")
    assert mesh_node_lon.var_name == "nlon"
    assert mesh_node_lon.standard_name == "longitude"
    assert mesh_node_lon.long_name == "node longitude"
    assert mesh_node_lon.units == "degrees_east"
    assert mesh_node_lon.attributes == {}
    np.testing.assert_allclose(
        mesh_node_lon.points,
        [0.0, 180.0, 270.0, 0.0, 90, 0.0],
        rtol=1e-5,
    )
    assert mesh_node_lon.bounds is None

    # Check connectivity
    assert len(mesh.connectivities()) == 1
    conn = mesh.connectivity()
    assert conn.var_name is None
    assert conn.standard_name is None
    assert conn.long_name is None
    assert conn.units == "unknown"
    assert conn.attributes == {}
    assert conn.cf_role == "face_node_connectivity"
    assert conn.start_index == 1
    assert conn.location_axis == 0
    assert conn.shape == (8, 3)
    np.testing.assert_array_equal(
        conn.indices,
        [
            [1, 3, 2],
            [1, 4, 3],
            [1, 5, 4],
            [1, 2, 5],
            [2, 3, 6],
            [3, 4, 6],
            [4, 5, 6],
            [5, 2, 6],
        ],
    )


def check_typesi(cube):
    """Check scalar typesi coordinate of cube."""
    assert cube.coords("area_type")
    typesi = cube.coord("area_type")
    assert typesi.var_name == "type"
    assert typesi.standard_name == "area_type"
    assert typesi.long_name == "Sea Ice area type"
    assert typesi.units.is_no_unit()
    np.testing.assert_array_equal(typesi.points, ["sea_ice"])
    assert typesi.bounds is None


# Test fix for all variables


def test_allvars_fix():
    """Test fix for all variables."""
    assert issubclass(AllVars, AllVarsBase)
    assert AllVars.fix_file is AllVarsBase.fix_file
    assert AllVars.fix_metadata is AllVarsBase.fix_metadata
    assert AllVars.fix_data is AllVarsBase.fix_data
    assert AllVars.DEFAULT_PFULL_VAR_NAME == "pres"


# Test ch4Clim (for time dimension time2)


def test_get_ch4clim_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "ch4Clim")
    assert fix == [AllVars(None), GenericFix(None)]


def test_ch4clim_fix(cubes_regular_grid):
    """Test fix."""
    cube = cubes_regular_grid[0]
    cube.var_name = "ch4Clim"
    cube.units = "mol mol-1"
    cube.coord("time").units = "no_unit"
    cube.coord("time").attributes["invalid_units"] = "day as %Y%m%d.%f"
    cube.coord("time").points = [18500201.0]
    cube.coord("time").long_name = "wrong_time_name"

    fix = get_allvars_fix("Amon", "ch4Clim")
    fixed_cubes = fix.fix_metadata(cubes_regular_grid)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "ch4Clim"
    assert cube.standard_name == "mole_fraction_of_methane_in_air"
    assert cube.long_name == "Mole Fraction of CH4"
    assert cube.units == "mol mol-1"
    assert "positive" not in cube.attributes

    time_coord = cube.coord("time")
    assert time_coord.var_name == "time"
    assert time_coord.standard_name == "time"
    assert time_coord.long_name == "time"
    assert time_coord.units == Unit(
        "days since 1850-01-01",
        calendar="proleptic_gregorian",
    )
    np.testing.assert_allclose(time_coord.points, [15.5])
    np.testing.assert_allclose(time_coord.bounds, [[0.0, 31.0]])


# Test clwvi (for extra fix)


def test_get_clwvi_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "clwvi")
    assert fix == [Clwvi(None), AllVars(None), GenericFix(None)]


def test_clwvi_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList(
        [cubes_regular_grid[0].copy(), cubes_regular_grid[0].copy()],
    )
    cubes[0].var_name = "tqc_dia"
    cubes[1].var_name = "tqi_dia"
    cubes[0].units = "kg m**-2"
    cubes[1].units = "kg m**-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "clwvi")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "clwvi"
    assert cube.standard_name == (
        "atmosphere_mass_content_of_cloud_condensed_water"
    )
    assert cube.long_name == "Condensed Water Path"
    assert cube.units == "kg m-2"
    assert "positive" not in cube.attributes

    np.testing.assert_allclose(cube.data, [[[0.0, 2.0], [4.0, 6.0]]])


# Test evspsbl (for extra fix)


def test_get_evspsbl_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "evspsbl")
    assert fix == [Evspsbl(None), AllVars(None), GenericFix(None)]


def test_evspsbl_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "qhfl_s"
    cubes[0].units = "kg m-2 s-1"

    fixed_cubes = fix_metadata(cubes, "Amon", "evspsbl")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "evspsbl"
    assert cube.standard_name == "water_evapotranspiration_flux"
    assert cube.long_name == (
        "Evaporation Including Sublimation and Transpiration"
    )
    assert cube.units == "kg m-2 s-1"
    assert "positive" not in cube.attributes

    fixed_cube = fix_data(cube, "Amon", "evspsbl")

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test gpp (for extra fix)


def test_get_gpp_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Lmon", "gpp")
    assert fix == [Gpp(None), AllVars(None), GenericFix(None)]


def test_gpp_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "assimi_gross_assimilation_box"
    cubes[0].units = "mol m-2 s-1"

    fixed_cubes = fix_metadata(cubes, "Lmon", "gpp")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "gpp"
    assert cube.standard_name == (
        "gross_primary_productivity_of_biomass_expressed_as_carbon"
    )
    assert cube.long_name == (
        "Carbon Mass Flux out of Atmosphere Due to Gross Primary Production on Land [kgC m-2 s-1]"
    )
    assert cube.units == "kg m-2 s-1"
    assert "positive" not in cube.attributes
    assert "invalid_units" not in cube.attributes

    fixed_cube = fix_data(cube, "Lmon", "gpp")

    np.testing.assert_allclose(
        fixed_cube.data,
        [
            [
                [0.0, 1.0 * 44.0095 / 1000],
                [2.0 * 44.0095 / 1000, 3.0 * 44.0095 / 1000],
            ],
        ],
    )


# Test hfls (for extra fix)


def test_get_hfls_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "hfls")
    assert fix == [Hfls(None), AllVars(None), GenericFix(None)]


def test_hfls_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "lhfl_s"
    cubes[0].units = "W m-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "hfls")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "hfls"
    assert cube.standard_name == "surface_upward_latent_heat_flux"
    assert cube.long_name == "Surface Upward Latent Heat Flux"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    fixed_cube = fix_data(cube, "Amon", "hfls")

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test hfss (for extra fix)


def test_get_hfss_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "hfss")
    assert fix == [Hfss(None), AllVars(None), GenericFix(None)]


def test_hfss_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "shfl_s"
    cubes[0].units = "W m-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "hfss")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "hfss"
    assert cube.standard_name == "surface_upward_sensible_heat_flux"
    assert cube.long_name == "Surface Upward Sensible Heat Flux"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    fixed_cube = fix_data(cube, "Amon", "hfss")

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test rlut (for extra fix)


def test_get_rlut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rlut")
    assert fix == [Rlut(None), AllVars(None), GenericFix(None)]


def test_rlut_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "thb_t"
    cubes[0].units = "W m-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "rlut")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rlut"
    assert cube.standard_name == "toa_outgoing_longwave_flux"
    assert cube.long_name == "TOA Outgoing Longwave Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    fixed_cube = fix_data(cube, "Amon", "rlut")

    np.testing.assert_allclose(fixed_cube.data, [[[0.0, -1.0], [-2.0, -3.0]]])


# Test rlutcs (for extra fix)


def test_get_rlutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rlutcs")
    assert fix == [Rlutcs(None), AllVars(None), GenericFix(None)]


def test_rlutcs_fix(cubes_atm_3d):
    """Test fix."""
    cube = cubes_atm_3d.extract_cube(NameConstraint(var_name="temp"))
    cube.var_name = "lwflx_up_clr"
    cube.units = "W m-2"
    cube.data = np.arange(1 * 47 * 8, dtype=np.float32).reshape(1, 47, 8)
    cubes = CubeList([cube])

    fixed_cubes = fix_metadata(cubes, "Amon", "rlutcs")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rlutcs"
    assert cube.standard_name == (
        "toa_outgoing_longwave_flux_assuming_clear_sky"
    )
    assert cube.long_name == "TOA Outgoing Clear-Sky Longwave Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    fixed_cube = fix_data(cube, "Amon", "rlutcs")

    assert fixed_cube.dtype == cube.dtype
    np.testing.assert_allclose(
        fixed_cube.data,
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
    )


# Test rsdt and rsut (for positive attribute)


def test_get_rsdt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rsdt")
    assert fix == [AllVars(None), GenericFix(None)]


def test_rsdt_fix(cubes_atm_2d):
    """Test fix."""
    fix = get_allvars_fix("Amon", "rsdt")
    fixed_cubes = fix.fix_metadata(cubes_atm_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rsdt"
    assert cube.standard_name == "toa_incoming_shortwave_flux"
    assert cube.long_name == "TOA Incident Shortwave Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "down"

    check_time(cube)
    check_lat_lon(cube)


def test_get_rsut_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rsut")
    assert fix == [AllVars(None), GenericFix(None)]


def test_rsut_fix(cubes_atm_2d):
    """Test fix."""
    fix = get_allvars_fix("Amon", "rsut")
    fixed_cubes = fix.fix_metadata(cubes_atm_2d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rsut"
    assert cube.standard_name == "toa_outgoing_shortwave_flux"
    assert cube.long_name == "TOA Outgoing Shortwave Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    check_time(cube)
    check_lat_lon(cube)


# Test rsutcs (for extra fix)


def test_get_rsutcs_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rsutcs")
    assert fix == [Rsutcs(None), AllVars(None), GenericFix(None)]


def test_rsutcs_fix(cubes_atm_3d):
    """Test fix."""
    cube = cubes_atm_3d.extract_cube(NameConstraint(var_name="temp"))
    cube.var_name = "swflx_up_clr"
    cube.units = "W m-2"
    cube.data = np.arange(1 * 47 * 8, dtype=np.float32).reshape(1, 47, 8)
    cubes = CubeList([cube])

    fixed_cubes = fix_metadata(cubes, "Amon", "rsutcs")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rsutcs"
    assert cube.standard_name == (
        "toa_outgoing_shortwave_flux_assuming_clear_sky"
    )
    assert cube.long_name == "TOA Outgoing Clear-Sky Shortwave Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "up"

    fixed_cube = fix_data(cube, "Amon", "rsutcs")

    assert fixed_cube.dtype == cube.dtype
    np.testing.assert_allclose(
        fixed_cube.data,
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
    )


# Test rtnt (for extra fix)


def test_get_rtnt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rtnt")
    assert fix == [Rtnt(None), AllVars(None), GenericFix(None)]


def test_rtnt_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList(
        [cubes_regular_grid[0].copy(), cubes_regular_grid[0].copy()],
    )
    cubes[0].var_name = "sob_t"
    cubes[1].var_name = "thb_t"
    cubes[0].units = "W m-2"
    cubes[1].units = "W m-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "rtnt")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rtnt"
    assert cube.standard_name is None
    assert cube.long_name == "TOA Net downward Total Radiation"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "down"

    np.testing.assert_allclose(cube.data, [[[0.0, 2.0], [4.0, 6.0]]])


# Test rtmt (for extra fix)


def test_get_rtmt_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "rtmt")
    assert fix == [Rtmt(None), AllVars(None), GenericFix(None)]


def test_rtmt_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList(
        [cubes_regular_grid[0].copy(), cubes_regular_grid[0].copy()],
    )
    cubes[0].var_name = "sob_t"
    cubes[1].var_name = "thb_t"
    cubes[0].units = "W m-2"
    cubes[1].units = "W m-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "rtmt")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "rtmt"
    assert cube.standard_name == (
        "net_downward_radiative_flux_at_top_of_atmosphere_model"
    )
    assert cube.long_name == "Net Downward Radiative Flux at Top of Model"
    assert cube.units == "W m-2"
    assert cube.attributes["positive"] == "down"

    np.testing.assert_allclose(cube.data, [[[0.0, 2.0], [4.0, 6.0]]])


# Test siconc (for extra_facets, removal of lev coord and  typesi coordinate)


def test_get_siconc_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "SImon", "siconc")
    assert fix == [AllVars(None), GenericFix(None)]


def test_siconc_fix(cubes_ocean_3d):
    """Test fix."""
    cubes = CubeList(
        [cubes_ocean_3d.extract_cube(NameConstraint(var_name="to")).copy()],
    )
    cubes[0].var_name = "conc"
    cubes[0].units = None

    # Add lev coord to test removal of it
    cubes[0] = cubes[0][:, [0], :]
    cubes[0].remove_coord("depth")
    cubes[0].add_dim_coord(DimCoord(0.0, var_name="lev"), 1)

    fix = get_allvars_fix("SImon", "siconc")
    fixed_cubes = fix.fix_metadata(cubes)

    cube = check_siconc_metadata(
        fixed_cubes,
        "siconc",
        "Sea-Ice Area Percentage (Ocean Grid)",
    )
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

    assert cube.shape == (1, 8)
    assert not cube.coords(var_name="lev")

    assert cube.dtype == np.float32
    np.testing.assert_allclose(
        cube.data,
        [
            [
                18660.58,
                18646.307,
                18668.656,
                18668.893,
                18651.273,
                18642.248,
                18647.305,
                18664.15,
            ],
        ],
    )


# Test siconca (for extra_facets and typesi coordinate)


def test_get_siconca_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "SImon", "siconca")
    assert fix == [AllVars(None), GenericFix(None)]


def test_siconca_fix(cubes_atm_2d):
    """Test fix."""
    fix = get_allvars_fix("SImon", "siconca")
    fixed_cubes = fix.fix_metadata(cubes_atm_2d)

    cube = check_siconc_metadata(
        fixed_cubes,
        "siconca",
        "Sea-Ice Area Percentage (Atmospheric Grid)",
    )
    check_time(cube)
    check_lat_lon(cube)
    check_typesi(cube)

    np.testing.assert_allclose(
        cube.data,
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]],
    )


# Test ta (for height and plev coordinate)


def test_get_ta_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "ta")
    assert fix == [AllVars(None), GenericFix(None)]


def test_ta_fix(cubes_atm_3d):
    """Test fix."""
    fix = get_allvars_fix("Amon", "ta")
    fixed_cubes = fix.fix_metadata(cubes_atm_3d)

    cube = check_ta_metadata(fixed_cubes)
    check_time(cube)
    check_height(cube)
    check_lat_lon(cube)

    assert cube.dtype == np.float32
    assert cube.shape == (1, 47, 8)


# Test tas (for height2m coordinate)


def test_get_tas_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "tas")
    assert fix == [AllVars(None), GenericFix(None)]


def test_tas_fix(cubes_atm_2d):
    """Test fix."""
    fix = get_allvars_fix("Amon", "tas")
    fixed_cubes = fix.fix_metadata(cubes_atm_2d)

    cube = check_tas_metadata(fixed_cubes)
    check_time(cube)
    check_lat_lon(cube)
    check_heightxm(cube, 2.0)

    assert cube.dtype == np.float32
    assert cube.shape == (1, 8)
    np.testing.assert_allclose(
        cube.data,
        [
            [
                266.02856,
                265.08435,
                264.6843,
                266.6293,
                262.27255,
                262.97803,
                260.04846,
                263.80975,
            ],
        ],
    )


# Test thetao (for depth coordinate)


def test_get_thetao_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Omon", "thetao")
    assert fix == [AllVars(None), GenericFix(None)]


def test_thetao_fix(cubes_ocean_3d):
    """Test fix."""
    fix = get_allvars_fix("Omon", "thetao")

    fixed_cubes = fix.fix_metadata(cubes_ocean_3d)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "thetao"
    assert cube.standard_name == "sea_water_potential_temperature"
    assert cube.long_name == "Sea Water Potential Temperature"
    assert cube.units == "degC"
    assert "positive" not in cube.attributes

    depth_coord = cube.coord("depth")
    assert depth_coord.has_bounds()

    assert cube.dtype == np.float32
    assert cube.shape == (1, 47, 8)


def test_thetao_fix_already_bounds(cubes_ocean_3d):
    """Test fix."""
    cube = cubes_ocean_3d.extract_cube(NameConstraint(var_name="to"))
    cube.coord("depth").guess_bounds()
    bounds = cube.coord("depth").bounds.copy()
    bounds[0, 0] = -1000.0
    cube.coord("depth").bounds = bounds
    cubes = CubeList([cube])

    fix = get_allvars_fix("Omon", "thetao")

    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "thetao"
    assert cube.standard_name == "sea_water_potential_temperature"
    assert cube.long_name == "Sea Water Potential Temperature"
    assert cube.units == "degC"
    assert "positive" not in cube.attributes

    depth_coord = cube.coord("depth")
    assert depth_coord.has_bounds()
    np.testing.assert_allclose(depth_coord.bounds[0, 0], -1000.0)

    assert cube.dtype == np.float32
    assert cube.shape == (1, 47, 8)


def test_thetao_fix_no_bounds(cubes_ocean_3d):
    """Test fix."""
    cube = cubes_ocean_3d.extract_cube(NameConstraint(var_name="to"))
    cubes = CubeList([cube])

    fix = get_allvars_fix("Omon", "thetao")

    fixed_cubes = fix.fix_metadata(cubes)

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "thetao"
    assert cube.standard_name == "sea_water_potential_temperature"
    assert cube.long_name == "Sea Water Potential Temperature"
    assert cube.units == "degC"
    assert "positive" not in cube.attributes

    depth_coord = cube.coord("depth")
    assert not depth_coord.has_bounds()

    assert cube.dtype == np.float32
    assert cube.shape == (1, 47, 8)


# Test zg (for extra fix)


def test_get_zg_fix():
    """Test getting of fix."""
    fix = Fix.get_fixes("ICON", "ICON-XPP", "Amon", "zg")
    assert fix == [Zg(None), AllVars(None), GenericFix(None)]


def test_zg_fix(cubes_regular_grid):
    """Test fix."""
    cubes = CubeList([cubes_regular_grid[0].copy()])
    cubes[0].var_name = "geopot"
    cubes[0].units = "m2 s-2"

    fixed_cubes = fix_metadata(cubes, "Amon", "zg")

    assert len(fixed_cubes) == 1
    cube = fixed_cubes[0]
    assert cube.var_name == "zg"
    assert cube.standard_name == "geopotential_height"
    assert cube.long_name == "Geopotential Height"
    assert cube.units == "m"
    assert "positive" not in cube.attributes

    np.testing.assert_allclose(
        cube.data,
        [[[0.0, 0.10197162], [0.20394324, 0.30591486]]],
    )
