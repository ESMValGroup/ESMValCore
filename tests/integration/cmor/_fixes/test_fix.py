"""Integration tests for fixes."""

import os
from pathlib import Path

import pytest
from iris.cube import Cube

from esmvalcore.cmor._fixes.cmip5.bnu_esm import Ch4
from esmvalcore.cmor._fixes.cmip5.canesm2 import FgCo2
from esmvalcore.cmor._fixes.cmip5.cesm1_bgc import Gpp
from esmvalcore.cmor._fixes.cmip6.cesm2 import Omon, Tos
from esmvalcore.cmor._fixes.cordex.cnrm_cerfacs_cnrm_cm5.aladin63 import (
    Tas,
)
from esmvalcore.cmor._fixes.cordex.cordex_fixes import AllVars
from esmvalcore.cmor._fixes.fix import GenericFix
from esmvalcore.cmor.fix import Fix
from esmvalcore.cmor.table import get_var_info
from esmvalcore.config import CFG


def test_get_fix():
    assert Fix.get_fixes('CMIP5', 'CanESM2', 'Amon', 'fgco2') == [
        FgCo2(None), GenericFix(None)
    ]


def test_get_fix_case_insensitive():
    assert Fix.get_fixes('CMIP5', 'CanESM2', 'Amon', 'fgCo2') == [
        FgCo2(None), GenericFix(None)
    ]


def test_get_fix_cordex():
    fix = Fix.get_fixes(
        'CORDEX',
        'ALADIN63',
        'Amon',
        'tas',
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'},
    )
    assert fix == [Tas(None), AllVars(None), GenericFix(None)]


def test_get_grid_fix_cordex():
    fix = Fix.get_fixes(
        'CORDEX',
        'ALADIN53',
        'Amon',
        'tas',
        extra_facets={'driver': 'CNRM-CERFACS-CNRM-CM5'},
    )
    assert fix == [AllVars(None), GenericFix(None)]


def test_get_fixes_with_replace():
    assert Fix.get_fixes('CMIP5', 'BNU-ESM', 'Amon', 'ch4') == [
        Ch4(None), GenericFix(None)
    ]


def test_get_fixes_with_generic():
    assert Fix.get_fixes('CMIP5', 'CESM1-BGC', 'Amon', 'gpp') == [
        Gpp(None), GenericFix(None)
    ]


def test_get_fix_no_project():
    with pytest.raises(KeyError):
        Fix.get_fixes('BAD_PROJECT', 'BNU-ESM', 'Amon', 'ch4')


def test_get_fix_no_model():
    assert Fix.get_fixes('CMIP5', 'BAD_MODEL', 'Amon', 'ch4') == [
        GenericFix(None)
    ]


def test_get_fix_no_var():
    assert Fix.get_fixes('CMIP5', 'BNU-ESM', 'Amon', 'BAD_VAR') == [
        GenericFix(None)
    ]


def test_get_fix_only_mip():
    assert Fix.get_fixes('CMIP6', 'CESM2', 'Omon', 'thetao') == [
        Omon(None), GenericFix(None)
    ]


def test_get_fix_only_mip_case_insensitive():
    assert Fix.get_fixes('CMIP6', 'CESM2', 'omOn', 'thetao') == [
        Omon(None), GenericFix(None)
    ]


def test_get_fix_mip_and_var():
    assert Fix.get_fixes('CMIP6', 'CESM2', 'Omon', 'tos') == [
        Tos(None), Omon(None), GenericFix(None)
    ]


def test_fix_metadata():
    cube = Cube([0])
    reference = Cube([0])
    assert Fix(None).fix_metadata(cube) == reference


def test_fix_data():
    cube = Cube([0])
    reference = Cube([0])
    assert Fix(None).fix_data(cube) == reference


def test_fix_file():
    filepath = 'sample_filepath'
    assert Fix(None).fix_file(filepath, 'preproc') == filepath


def test_get_fixed_filepath_paths(tmp_path):
    output_dir = tmp_path / 'fixed'
    filepath = Path('this', 'is', 'a', 'file.nc')
    assert not output_dir.is_dir()
    fixed_path = Fix(None).get_fixed_filepath(output_dir, filepath)
    assert output_dir.is_dir()
    assert isinstance(fixed_path, Path)
    assert fixed_path == tmp_path / 'fixed' / 'file.nc'


def test_get_fixed_filepath_unique_suffix_paths(tmp_path):
    output_dir = tmp_path / 'fixed' / 'prefix_1_'
    filepath = Path('this', 'is', 'a', 'file.nc')
    assert not output_dir.parent.is_dir()
    fixed_path = Fix(None).get_fixed_filepath(
        output_dir, filepath, add_unique_suffix=True
    )
    assert fixed_path.parent.is_dir()
    assert isinstance(fixed_path, Path)
    assert fixed_path != tmp_path / 'fixed' / 'prefix_1_' / 'file.nc'
    assert fixed_path.parent.name.startswith('prefix_1_')
    assert fixed_path.name == 'file.nc'


def test_get_fixed_filepath_strs(tmp_path):
    output_dir = os.path.join(str(tmp_path), 'fixed')
    filepath = os.path.join('this', 'is', 'a', 'file.nc')
    assert not Path(output_dir).is_dir()
    fixed_path = Fix(None).get_fixed_filepath(output_dir, filepath)
    assert Path(output_dir).is_dir()
    assert isinstance(fixed_path, Path)
    assert fixed_path == tmp_path / 'fixed' / 'file.nc'


def test_get_fixed_filepath_unique_suffix_strs(tmp_path):
    output_dir = os.path.join(str(tmp_path), 'fixed', 'prefix_1_')
    filepath = os.path.join('this', 'is', 'a', 'file.nc')
    assert not Path(output_dir).parent.is_dir()
    fixed_path = Fix(None).get_fixed_filepath(
        output_dir, filepath, add_unique_suffix=True
    )
    assert fixed_path.parent.is_dir()
    assert isinstance(fixed_path, Path)
    assert fixed_path != tmp_path / 'fixed' / 'prefix_1_' / 'file.nc'
    assert fixed_path.parent.name.startswith('prefix_1_')
    assert fixed_path.name == 'file.nc'


def test_session_empty():
    fix = Fix(None)
    assert fix.session is None


def test_session():
    session = CFG.start_session('my session')
    fix = Fix(None, session=session)
    assert fix.session == session


def test_frequency_empty():
    fix = Fix(None)
    assert fix.frequency is None


def test_frequency_from_vardef():
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Fix(vardef)
    assert fix.frequency == 'mon'


def test_frequency_given():
    fix = Fix(None, frequency='1hr')
    assert fix.frequency == '1hr'


def test_frequency_not_from_vardef():
    vardef = get_var_info('CMIP6', 'Amon', 'tas')
    fix = Fix(vardef, frequency='3hr')
    assert fix.frequency == '3hr'
