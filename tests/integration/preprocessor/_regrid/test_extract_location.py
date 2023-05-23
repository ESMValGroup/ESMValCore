"""Integration tests for :func:`esmvalcore.preprocessor.extract_location."""
import ssl
from unittest.mock import patch

import iris
import iris.fileformats
import numpy as np
import pytest
from iris.coords import CellMethod, DimCoord

from esmvalcore.preprocessor import extract_location


@pytest.fixture(autouse=True)
def mocked_geopy_geocoders_nominatim(mocker):
    """Mock :class:`geopy.geocoders.Nominatim`.

    See https://github.com/ESMValGroup/ESMValCore/issues/1982.
    """
    mocked_nominatim = mocker.patch(
        'esmvalcore.preprocessor._regrid.Nominatim', autospec=True)
    geolocation_penacaballera = mocker.Mock(latitude=40.3442754,
                                            longitude=-5.8606859)
    mocked_nominatim.return_value.geocode.side_effect = (
        lambda x: geolocation_penacaballera if x == 'Peñacaballera' else None)


@pytest.fixture
def test_cube():
    """Create a 3d synthetic test cube."""
    shape = (3, 45, 36)
    data = np.arange(np.prod(shape)).reshape(shape)
    z, y, x = shape

    # Create the cube.
    cm = CellMethod(method='mean',
                    coords='time',
                    intervals='20 minutes',
                    comments=None)
    kwargs = dict(standard_name='air_temperature',
                  long_name='Air Temperature',
                  var_name='ta',
                  units='K',
                  attributes=dict(cube='attribute'),
                  cell_methods=(cm, ))
    cube = iris.cube.Cube(data, **kwargs)

    # Create a synthetic test latitude coordinate.
    data = np.linspace(-90, 90, y)
    cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    kwargs = dict(standard_name='latitude',
                  long_name='Latitude',
                  var_name='lat',
                  units='degrees_north',
                  attributes=dict(latitude='attribute'),
                  coord_system=cs)
    ycoord = DimCoord(data, **kwargs)
    ycoord.guess_bounds()
    cube.add_dim_coord(ycoord, 1)

    # Create a synthetic test longitude coordinate.
    data = np.linspace(0, 360, x)
    kwargs = dict(standard_name='longitude',
                  long_name='Longitude',
                  var_name='lon',
                  units='degrees_east',
                  attributes=dict(longitude='attribute'),
                  coord_system=cs)
    xcoord = DimCoord(data, **kwargs)
    xcoord.guess_bounds()
    cube.add_dim_coord(xcoord, 2)
    return cube


def test_extract_successful(test_cube):
    """Test only town name."""
    point = extract_location(test_cube,
                             scheme='nearest',
                             location='Peñacaballera')
    assert point.shape == (3, )
    np.testing.assert_equal(point.data, [1186, 2806, 4426])


def test_non_existing_location(test_cube):
    """Test town plus region plus country."""
    msg = "Requested location Minas Tirith,Gondor can not be found"
    with pytest.raises(ValueError, match=msg):
        extract_location(test_cube,
                         scheme='nearest',
                         location='Minas Tirith,Gondor')


def test_no_location_parameter(test_cube):
    """Test if no location supplied."""
    msg = "Location needs to be specified."
    with pytest.raises(ValueError, match=msg):
        extract_location(test_cube, scheme='nearest', location=None)


def test_no_scheme_parameter(test_cube):
    """Test if no scheme supplied."""
    msg = "Interpolation scheme needs to be specified."
    with pytest.raises(ValueError, match=msg):
        extract_location(test_cube,
                         scheme=None,
                         location='Calvitero,Candelario')


@patch("esmvalcore.preprocessor._regrid.ssl.create_default_context")
def test_create_default_ssl_context_raises_exception(mock_create, test_cube):
    """Test the original way 'extract_location' worked before adding the
    default SSL context, see
    https://github.com/ESMValGroup/ESMValCore/issues/2012 for more
    information."""
    mock_create.side_effect = ssl.SSLSyscallError
    extract_location(test_cube, scheme="nearest", location="Peñacaballera")
    mock_create.assert_called_once()
