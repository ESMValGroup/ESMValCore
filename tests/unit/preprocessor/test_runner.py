from pathlib import Path

import iris.cube
import pytest

import esmvalcore.preprocessor


def test_first_argument_name():
    """Check that the input type of all preprocessor functions is valid."""
    valid_itypes = ('file', 'files', 'cube', 'cubes', 'products',
                    'input_products')
    for step in esmvalcore.preprocessor.DEFAULT_ORDER:
        itype = esmvalcore.preprocessor._get_itype(step)
        assert itype in valid_itypes, (
            "Invalid preprocessor function definition {}, first argument "
            "should be one of {} but is {}".format(step, valid_itypes, itype))


def test_multi_model_exist():
    assert esmvalcore.preprocessor.MULTI_MODEL_FUNCTIONS.issubset(
        set(esmvalcore.preprocessor.DEFAULT_ORDER))


@pytest.mark.parametrize('debug', [False, True])
def test_preprocess_debug(mocker, debug):
    in_cube = iris.cube.Cube([1], var_name='tas')
    out_cube = iris.cube.Cube([2], var_name='tas')

    items = [in_cube]
    result = [out_cube]
    step = 'annual_statistics'
    input_files = [Path('/path/to/input.nc')]
    output_file = Path('/path/to/output.nc')

    mock_annual_statistics = mocker.create_autospec(
        esmvalcore.preprocessor.annual_statistics,
        return_value=out_cube,
    )
    mock_save = mocker.create_autospec(esmvalcore.preprocessor.save)
    mocker.patch(
        'esmvalcore.preprocessor.annual_statistics',
        new=mock_annual_statistics
    )
    mocker.patch('esmvalcore.preprocessor.save', new=mock_save)

    esmvalcore.preprocessor.preprocess(
        items,
        step,
        input_files=input_files,
        output_file=output_file,
        debug=debug,
        operator='mean',
    )
    esmvalcore.preprocessor.annual_statistics.assert_called_with(
        in_cube, operator='mean')
    if debug:
        esmvalcore.preprocessor.save.assert_called_with(
            result, '/path/to/output/00_annual_statistics.nc')
    else:
        esmvalcore.preprocessor.save.assert_not_called()
