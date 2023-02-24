"""Tests for `esmvalcore.preprocessor.PreprocessingTask`."""
import iris
import iris.cube
from prov.model import ProvDocument

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
    dataset._load_with_callback = lambda _: cube

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
