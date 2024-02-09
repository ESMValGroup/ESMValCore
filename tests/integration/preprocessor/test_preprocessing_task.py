"""Tests for `esmvalcore.preprocessor.PreprocessingTask`."""
import iris
import iris.cube
from prov.model import ProvDocument

import esmvalcore.preprocessor
from esmvalcore.dataset import Dataset
from esmvalcore.preprocessor import PreprocessingTask, PreprocessorFile


def test_load_save_task(tmp_path):
    """Test that a task that just loads and saves a file."""
    # Prepare a test dataset
    cube = iris.cube.Cube(data=[273.], var_name='tas', units='K')
    in_file = tmp_path / 'tas_in.nc'
    iris.save(cube, in_file)
    dataset = Dataset(short_name='tas')
    dataset.files = [in_file]
    dataset.load = lambda: cube.copy()

    # Create task
    task = PreprocessingTask([
        PreprocessorFile(
            filename=tmp_path / 'tas_out.nc',
            settings={},
            datasets=[dataset],
        ),
    ])

    # Create an 'activity' representing a run of the tool
    provenance = ProvDocument()
    provenance.add_namespace('software', uri='https://example.com/software')
    activity = provenance.activity('software:esmvalcore')
    task.initialize_provenance(activity)

    task.run()

    assert len(task.products) == 1
    preproc_file = task.products.pop().filename
    result = iris.load_cube(preproc_file)

    result.attributes.clear()
    assert result == cube


def test_load_save_and_other_task(tmp_path, monkeypatch):
    """Test that a task just copies one file and preprocesses another file."""
    # Prepare test datasets
    in_cube = iris.cube.Cube(data=[0.], var_name='tas', units='degrees_C')
    (tmp_path / 'climate_data').mkdir()
    file1 = tmp_path / 'climate_data' / 'tas_dataset1.nc'
    file2 = tmp_path / 'climate_data' / 'tas_dataset2.nc'

    # Save cubes for reading global attributes into provenance
    iris.save(in_cube, target=file1)
    iris.save(in_cube, target=file2)

    dataset1 = Dataset(short_name='tas', dataset='dataset1')
    dataset1.files = [file1]
    dataset1.load = lambda: in_cube.copy()

    dataset2 = Dataset(short_name='tas', dataset='dataset1')
    dataset2.files = [file2]
    dataset2.load = lambda: in_cube.copy()

    # Create some mock preprocessor functions and patch
    # `esmvalcore.preprocessor` so it uses them.
    def single_preproc_func(cube):
        cube.data = cube.core_data() + 1.
        return cube

    def multi_preproc_func(products):
        for product in products:
            cube = product.cubes[0]
            cube.data = cube.core_data() + 1.
            product.cubes = [cube]
        return products

    monkeypatch.setattr(
        esmvalcore.preprocessor,
        'single_preproc_func',
        single_preproc_func,
        raising=False,
    )
    monkeypatch.setattr(
        esmvalcore.preprocessor,
        'multi_preproc_func',
        multi_preproc_func,
        raising=False,
    )
    monkeypatch.setattr(
        esmvalcore.preprocessor,
        'MULTI_MODEL_FUNCTIONS',
        {'multi_preproc_func'},
    )
    default_order = (esmvalcore.preprocessor.INITIAL_STEPS +
                     ('single_preproc_func', 'multi_preproc_func') +
                     esmvalcore.preprocessor.FINAL_STEPS)
    monkeypatch.setattr(
        esmvalcore.preprocessor,
        'DEFAULT_ORDER',
        default_order,
    )

    # Create task
    task = PreprocessingTask(
        [
            PreprocessorFile(
                filename=tmp_path / 'tas_dataset1.nc',
                settings={},
                datasets=[dataset1],
                attributes={'dataset': 'dataset1'},
            ),
            PreprocessorFile(
                filename=tmp_path / 'tas_dataset2.nc',
                settings={
                    'single_preproc_func': {},
                    'multi_preproc_func': {},
                },
                datasets=[dataset2],
                attributes={'dataset': 'dataset2'},
            ),
        ],
        order=default_order,
    )

    # Create an 'activity' representing a run of the tool
    provenance = ProvDocument()
    provenance.add_namespace('software', uri='https://example.com/software')
    activity = provenance.activity('software:esmvalcore')
    task.initialize_provenance(activity)

    task.run()

    # Check that two files were saved and the preprocessor functions were
    # only applied to the second one.
    assert len(task.products) == 2
    for product in task.products:
        print(product.filename)
        assert product.filename.exists()
        out_cube = iris.load_cube(product.filename)
        print(out_cube.data)
        if product.attributes['dataset'] == 'dataset1':
            assert out_cube.data.tolist() == [0.]
        elif product.attributes['dataset'] == 'dataset2':
            assert out_cube.data.tolist() == [2.]
        else:
            assert False, "unexpected product"
