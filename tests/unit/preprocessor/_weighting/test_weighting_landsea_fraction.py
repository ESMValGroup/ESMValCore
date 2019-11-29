"""Unit tests for :mod:`esmvalcore.preprocessor._weighting`."""
from collections import OrderedDict

import iris
import mock
import numpy as np
import pytest
from cf_units import Unit

import esmvalcore.preprocessor._weighting as weighting

CUBE_SFTLF = iris.cube.Cube(
    [10.0, 0.0, 100.0],
    var_name='sftlf',
    standard_name='land_area_fraction',
    units=Unit('%'),
)
CUBE_SFTOF = iris.cube.Cube(
    [100.0, 0.0, 50.0, 70.0],
    var_name='sftof',
    standard_name='sea_area_fraction',
    units=Unit('%'),
)
CUBE_3 = iris.cube.Cube(
    [10.0, 20.0, 0.0],
    var_name='dim3',
)
CUBE_4 = iris.cube.Cube(
    [1.0, 2.0, -1.0, 2.0],
    var_name='dim4',
)
FRAC_SFTLF = np.array([0.1, 0.0, 1.0])
FRAC_SFTOF = np.array([0.0, 1.0, 0.5, 0.3])
EMPTY_FX_FILES = OrderedDict([
    ('sftlf', []),
    ('sftof', []),
])
L_FX_FILES = OrderedDict([
    ('sftlf', 'not/a/real/path'),
    ('sftof', []),
])
O_FX_FILES = OrderedDict([
    ('sftlf', []),
    ('sftof', 'not/a/real/path'),
])
FX_FILES = OrderedDict([
    ('sftlf', 'not/a/real/path'),
    ('sftof', 'i/was/mocked'),
])
WRONG_FX_FILES = OrderedDict([
    ('wrong', 'test'),
    ('sftlf', 'not/a/real/path'),
    ('sftof', 'i/was/mocked'),
])

LAND_FRACTION = [
    (CUBE_3, {}, [], None, ["No fx files given"]),
    (CUBE_3, {'sftlf': []}, [], None, ["'sftlf' not found"]),
    (CUBE_3, {'sftlf': 'a'}, [CUBE_SFTLF], FRAC_SFTLF, []),
    (CUBE_3, {'sftof': 'a'}, [CUBE_SFTOF], None, ["not broadcastable"]),
    (CUBE_3, EMPTY_FX_FILES, [], None,
     ["'sftlf' not found", "'sftof' not found"]),
    (CUBE_3, L_FX_FILES, [CUBE_SFTLF], FRAC_SFTLF, []),
    (CUBE_3, O_FX_FILES, [CUBE_SFTOF], None,
     ["'sftlf' not found", "not broadcastable"]),
    (CUBE_3, FX_FILES, [CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTLF, []),
    (CUBE_3, {'wrong': 'a'}, [CUBE_SFTLF], None,
     ["expected 'sftlf' or 'sftof'"]),
    (CUBE_3, {'wrong': 'a'}, [CUBE_SFTOF], None, ["not broadcastable"]),
    (CUBE_3, WRONG_FX_FILES, [CUBE_SFTLF, CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTLF,
     ["expected 'sftlf' or 'sftof'"]),
    (CUBE_3, WRONG_FX_FILES, [CUBE_SFTOF, CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTLF,
     ["not broadcastable"]),
    (CUBE_4, {}, [], None, ["No fx files given"]),
    (CUBE_4, {'sftlf': []}, [], None, ["'sftlf' not found"]),
    (CUBE_4, {'sftlf': 'a'}, [CUBE_SFTLF], None, ["not broadcastable"]),
    (CUBE_4, {'sftof': 'a'}, [CUBE_SFTOF], FRAC_SFTOF, []),
    (CUBE_4, EMPTY_FX_FILES, [], None,
     ["'sftlf' not found", "'sftof' not found"]),
    (CUBE_4, L_FX_FILES, [CUBE_SFTLF], None,
     ["not broadcastable", "'sftof' not found"]),
    (CUBE_4, O_FX_FILES, [CUBE_SFTOF], FRAC_SFTOF, ["'sftlf' not found"]),
    (CUBE_4, FX_FILES, [CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTOF,
     ["not broadcastable"]),
    (CUBE_4, {'wrong': 'a'}, [CUBE_SFTLF], None, ["not broadcastable"]),
    (CUBE_4, {'wrong': 'a'}, [CUBE_SFTOF], None,
     ["expected 'sftlf' or 'sftof'"]),
    (CUBE_4, WRONG_FX_FILES, [CUBE_SFTLF, CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTOF,
     ["not broadcastable", "not broadcastable"]),
    (CUBE_4, WRONG_FX_FILES, [CUBE_SFTOF, CUBE_SFTLF, CUBE_SFTOF], FRAC_SFTOF,
     ["expected 'sftlf' or 'sftof'", "not broadcastable"]),
]


@pytest.mark.parametrize('cube,fx_files,fx_cubes,out,err', LAND_FRACTION)
@mock.patch.object(weighting, 'iris', autospec=True)
def test_get_land_fraction(mock_iris, cube, fx_files, fx_cubes, out, err):
    """Test calulation of land fraction."""
    mock_iris.load_cube.side_effect = fx_cubes
    (land_fraction, errors) = weighting._get_land_fraction(cube, fx_files)
    if land_fraction is None:
        assert land_fraction == out
    else:
        assert np.allclose(land_fraction, out)
    assert len(errors) == len(err)
    for (idx, error) in enumerate(errors):
        assert err[idx] in error
    mock_iris.reset_mock()


SHAPES_TO_BROADCAST = [
    ((), (1, ), True),
    ((), (10, 10), True),
    ((1, ), (10, ), True),
    ((1, ), (10, 10), True),
    ((2, ), (10, ), False),
    ((10, ), (), True),
    ((10, ), (1, ), True),
    ((10, ), (10, ), True),
    ((10, ), (10, 10), True),
    ((10, ), (7, 1), True),
    ((10, ), (10, 7), False),
    ((10, ), (7, 1, 10), True),
    ((10, ), (7, 1, 1), True),
    ((10, ), (7, 1, 7), False),
    ((10, ), (7, 10, 7), False),
    ((10, 1), (1, 1), True),
    ((10, 1), (1, 100), True),
    ((10, 1), (10, 7), True),
    ((10, 12), (10, 1), True),
    ((10, 12), (), True),
    ((10, 12), (1, ), True),
    ((10, 12), (12, ), True),
    ((10, 12), (1, 1), True),
    ((10, 12), (1, 12), True),
    ((10, 12), (10, 10, 1), True),
    ((10, 12), (10, 12, 1), False),
    ((10, 12), (10, 12, 12), False),
    ((10, 12), (10, 10, 12), True),
]


@pytest.mark.parametrize('shape_1,shape_2,out', SHAPES_TO_BROADCAST)
def test_shape_is_broadcastable(shape_1, shape_2, out):
    """Test check if two shapes are broadcastable."""
    is_broadcastable = weighting._shape_is_broadcastable(shape_1, shape_2)
    assert is_broadcastable == out


CUBE_3_L = CUBE_3.copy([1.0, 0.0, 0.0])
CUBE_3_O = CUBE_3.copy([9.0, 20.0, 0.0])
CUBE_4_L = CUBE_4.copy([0.0, 2.0, -0.5, 0.6])
CUBE_4_O = CUBE_4.copy([1.0, 0.0, -0.5, 1.4])

WEIGHTING_LANDSEA_FRACTION = [
    (CUBE_3, {}, 'land', ValueError),
    (CUBE_3, {}, 'sea', ValueError),
    (CUBE_3, EMPTY_FX_FILES, 'land', ValueError),
    (CUBE_3, EMPTY_FX_FILES, 'sea', ValueError),
    (CUBE_3, L_FX_FILES, 'land', CUBE_3_L),
    (CUBE_3, L_FX_FILES, 'sea', CUBE_3_O),
    (CUBE_3, O_FX_FILES, 'land', ValueError),
    (CUBE_3, O_FX_FILES, 'sea', ValueError),
    (CUBE_3, FX_FILES, 'land', CUBE_3_L),
    (CUBE_3, FX_FILES, 'sea', CUBE_3_O),
    (CUBE_3, FX_FILES, 'wrong', TypeError),
    (CUBE_4, {}, 'land', ValueError),
    (CUBE_4, {}, 'sea', ValueError),
    (CUBE_4, EMPTY_FX_FILES, 'land', ValueError),
    (CUBE_4, EMPTY_FX_FILES, 'sea', ValueError),
    (CUBE_4, L_FX_FILES, 'land', ValueError),
    (CUBE_4, L_FX_FILES, 'sea', ValueError),
    (CUBE_4, O_FX_FILES, 'land', CUBE_4_L),
    (CUBE_4, O_FX_FILES, 'sea', CUBE_4_O),
    (CUBE_4, FX_FILES, 'land', CUBE_4_L),
    (CUBE_4, FX_FILES, 'sea', CUBE_4_O),
    (CUBE_4, FX_FILES, 'wrong', TypeError),
]


@pytest.mark.parametrize('cube,fx_files,area_type,out',
                         WEIGHTING_LANDSEA_FRACTION)
@mock.patch.object(weighting, 'iris', autospec=True)
def test_weighting_landsea_fraction(mock_iris,
                                    cube,
                                    fx_files,
                                    area_type,
                                    out):
    """Test landsea fraction weighting preprocessor."""
    # Exceptions
    if isinstance(out, type):
        with pytest.raises(out):
            weighted_cube = weighting.weighting_landsea_fraction(
                cube, fx_files, area_type)
        return

    # Regular cases
    fx_cubes = []
    if fx_files.get('sftlf'):
        fx_cubes.append(CUBE_SFTLF)
    if fx_files.get('sftof'):
        fx_cubes.append(CUBE_SFTOF)
    mock_iris.load_cube.side_effect = fx_cubes
    weighted_cube = weighting.weighting_landsea_fraction(
        cube, fx_files, area_type)
    assert weighted_cube == cube
    assert weighted_cube is cube
    mock_iris.reset_mock()
