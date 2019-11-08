from iris.cube import Cube, CubeList

from esmvalcore.preprocessor import derive
from esmvalcore.preprocessor._derive import get_required
from esmvalcore.preprocessor._derive.csoil_grid import DerivedVariable


def test_get_required():

    variables = get_required('alb', 'CMIP5')

    reference = [
        {
            'short_name': 'rsdscs',
        },
        {
            'short_name': 'rsuscs',
        },
    ]

    assert variables == reference


def test_get_required_with_fx():

    variables = get_required('nbp_grid', 'CMIP5')

    reference = [
        {'short_name': 'nbp'},
        {'short_name': 'sftlf', 'mip': 'fx', 'optional': True},
    ]

    assert variables == reference


def test_derive_nonstandard_nofx():

    short_name = 'alb'
    long_name = 'albedo at the surface'
    units = 1
    standard_name = ''

    rsdscs = Cube([2.])
    rsdscs.short_name = 'rsdscs'
    rsdscs.var_name = rsdscs.short_name

    rsuscs = Cube([1.])
    rsuscs.short_name = 'rsuscs'
    rsuscs.var_name = rsuscs.short_name

    cubes = CubeList([rsdscs, rsuscs])

    alb = derive(cubes, short_name, long_name, units, standard_name)

    print(alb)
    assert alb.var_name == short_name
    assert alb.long_name == long_name
    assert alb.units == units
    assert alb.data == [0.5]


def test_derive_noop():

    alb = Cube([1.])
    alb.var_name = 'alb'
    alb.long_name = 'albedo at the surface'
    alb.units = 1

    cube = derive([alb], alb.var_name, alb.long_name, alb.units)

    print(cube)
    assert cube is alb


def test_derive_mixed_case_with_fx(tmp_path, monkeypatch):

    short_name = 'cSoil_grid'
    long_name = 'Carbon Mass in Soil Pool relative to grid cell area'
    units = 'kg m-2'

    csoil_cube = Cube([])

    def mock_calculate(self, cubes):
        assert len(cubes) == 1
        assert cubes[0] == csoil_cube
        return Cube([])

    monkeypatch.setattr(DerivedVariable, 'calculate', mock_calculate)

    derive(
        [csoil_cube],
        short_name,
        long_name,
        units,
    )
