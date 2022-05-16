import yaml

from esmvalcore._task import ResumeTask


def test_run(tmp_path):
    """Test `esmvalcore._task.ResumeTask.run`."""
    task_name = 'diagnostic_name/var_name'
    prev_output_dir = tmp_path / 'recipe_test_20210911_102100'
    prev_preproc_dir = prev_output_dir / 'preproc' / task_name
    prev_preproc_dir.mkdir(parents=True)
    prev_metadata = {
        f'/original/recipe_output/preproc/{task_name}/file.nc': {
            'filename': f'/original/recipe_output/preproc/{task_name}/file.nc',
            'attribute1': 'value1',
        }
    }
    prev_metadata_file = prev_preproc_dir / 'metadata.yml'
    with prev_metadata_file.open('w') as file:
        yaml.safe_dump(prev_metadata, file)

    output_dir = tmp_path / 'recipe_test_20211001_092100'
    preproc_dir = output_dir / 'preproc' / task_name

    task = ResumeTask(
        prev_preproc_dir,
        preproc_dir,
        task_name,
    )

    result = task.run()

    metadata_file = preproc_dir / 'metadata.yml'

    assert result == [str(metadata_file)]

    with metadata_file.open('rb') as file:
        metadata = yaml.safe_load(file)
    assert metadata == {
        str(prev_preproc_dir / 'file.nc'): {
            'filename': str(prev_preproc_dir / 'file.nc'),
            'attribute1': 'value1',
        },
    }
