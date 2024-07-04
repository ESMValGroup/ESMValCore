"""Test AIRS-2-0 fixes."""
import dask.array as da
import numpy as np
from iris.cube import Cube, CubeList

from esmvalcore.cmor.fix import fix_metadata


def test_fix_metadata_hur():
    """Test ``fix_metadata`` for hur."""
    cubes = CubeList([
        Cube(
            da.from_array([-0.1, 0.2, 1.2, 1.7]),
            var_name='hur',
            units='1',
            attributes={'valid_range': [0.0, 1.5]},
        ),
    ])

    fixed_cubes = fix_metadata(
        cubes, 'hur', 'obs4MIPs', 'AIRS-2-0', 'Amon', check_level=5,
    )

    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]
    assert fixed_cube.units == '%'
    assert fixed_cube.attributes == {}
    assert fixed_cube.has_lazy_data()
    expected_data = np.ma.masked_invalid([np.nan, 20.0, 120.0, np.nan])
    np.testing.assert_allclose(fixed_cube.data.mask, expected_data.mask)
    np.testing.assert_allclose(fixed_cube.data, expected_data)
