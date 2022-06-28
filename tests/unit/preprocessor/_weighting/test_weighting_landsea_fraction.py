"""Unit tests for :mod:`esmvalcore.preprocessor._weighting`."""

import iris
import iris.fileformats
import numpy as np
import pytest
from cf_units import Unit

import esmvalcore.preprocessor._weighting as weighting

crd_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
LON_3 = iris.coords.DimCoord([0, 1.5, 3],
                             standard_name='longitude',
                             bounds=[[0, 1], [1, 2], [2, 3]],
                             units='degrees_east',
                             coord_system=crd_sys)
LON_4 = iris.coords.DimCoord([0, 1.5, 2.5, 3.5],
                             standard_name='longitude',
                             bounds=[[0, 1], [1, 2], [2, 3],
                                     [3, 4]],
                             units='degrees_east',
                             coord_system=crd_sys)

CUBE_SFTLF = iris.cube.Cube(
    [10.0, 0.0, 100.0],
    var_name='sftlf',
    standard_name='land_area_fraction',
    units=Unit('%'),
    dim_coords_and_dims=[(LON_3, 0), ]
)
CUBE_SFTOF = iris.cube.Cube(
    [100.0, 0.0, 50.0, 70.0],
    var_name='sftof',
    standard_name='sea_area_fraction',
    units=Unit('%'),
    dim_coords_and_dims=[(LON_4, 0), ]
)
CUBE_3 = iris.cube.Cube(
    [10.0, 20.0, 0.0],
    var_name='dim3',
    dim_coords_and_dims=[(LON_3, 0), ]
)
CUBE_4 = iris.cube.Cube(
    [1.0, 2.0, -1.0, 2.0],
    var_name='dim4',
    dim_coords_and_dims=[(LON_4, 0), ]
)

CUBE_ANCILLARY_3 = CUBE_3.copy()
CUBE_ANCILLARY_3.add_ancillary_variable(CUBE_SFTLF, (0))

CUBE_ANCILLARY_4 = CUBE_4.copy()
CUBE_ANCILLARY_4.add_ancillary_variable(CUBE_SFTOF, (0))

FRAC_SFTLF = np.array([0.1, 0.0, 1.0])
FRAC_SFTOF = np.array([0.0, 1.0, 0.5, 0.3])

LAND_FRACTION = [
    (CUBE_3, None, [
        'Ancillary variables land/sea area fraction not found in cube. '
        'Check fx_file availability.']),
    (CUBE_4, None, [
        'Ancillary variables land/sea area fraction not found in cube. '
        'Check fx_file availability.']),
    (CUBE_ANCILLARY_3, FRAC_SFTLF, []),
    (CUBE_ANCILLARY_4, FRAC_SFTOF, [])
]


@pytest.mark.parametrize('cube,out,err', LAND_FRACTION)
def test_get_land_fraction(cube, out, err):
    """Test calculation of land fraction."""
    (land_fraction, errors) = weighting._get_land_fraction(cube)
    if land_fraction is None:
        assert land_fraction == out
    else:
        assert np.allclose(land_fraction, out)
    assert len(errors) == len(err)
    for (idx, error) in enumerate(errors):
        assert err[idx] in error


CUBE_3_L = CUBE_3.copy([1.0, 0.0, 0.0])
CUBE_3_O = CUBE_3.copy([9.0, 20.0, 0.0])
CUBE_4_L = CUBE_4.copy([0.0, 2.0, -0.5, 0.6])
CUBE_4_O = CUBE_4.copy([1.0, 0.0, -0.5, 1.4])

WEIGHTING_LANDSEA_FRACTION = [
    (CUBE_3, 'land', ValueError),
    (CUBE_3, 'sea', ValueError),
    (CUBE_ANCILLARY_3, 'land', CUBE_3_L),
    (CUBE_ANCILLARY_3, 'sea', CUBE_3_O),
    (CUBE_4, 'land', ValueError),
    (CUBE_4, 'sea', ValueError),
    (CUBE_ANCILLARY_4, 'land', CUBE_4_L),
    (CUBE_ANCILLARY_4, 'sea', CUBE_4_O),
]


@pytest.mark.parametrize('cube,area_type,out',
                         WEIGHTING_LANDSEA_FRACTION)
def test_weighting_landsea_fraction(cube,
                                    area_type,
                                    out):
    """Test landsea fraction weighting preprocessor."""
    # Exceptions
    if isinstance(out, type):
        with pytest.raises(out):
            weighted_cube = weighting.weighting_landsea_fraction(
                cube, area_type)
        return

    # Regular cases
    weighted_cube = weighting.weighting_landsea_fraction(cube, area_type)
    assert np.array_equal(weighted_cube.data, cube.data)
    assert weighted_cube is cube
