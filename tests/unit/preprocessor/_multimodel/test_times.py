import iris
# import pytest
from cf_units import Unit
import numpy as np

from esmvalcore.preprocessor._multimodel import multi_model_statistics

def timecoord(days=[1, 2], calendar='gregorian'):
    """Return a standard time coordinate with the given days as time points."""
    return iris.coords.DimCoord(days, standard_name='time', units = Unit('days since 1850-01-01', calendar=calendar))

cube1 = iris.cube.Cube([1, 1], dim_coords_and_dims=[(timecoord([1, 2]), 0)])
cube2 = iris.cube.Cube([2, 2, 2], dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
cube3 = iris.cube.Cube([3, 3], dim_coords_and_dims=[(timecoord([1, 3]), 0)])

# overlap between cube 1 and 2
result = multi_model_statistics([cube1, cube2], span='overlap', statistics=['mean'])['mean']
expected = iris.cube.Cube([1.5, 1.5], dim_coords_and_dims=[(timecoord([1, 2]), 0)])
assert np.all(result.data == expected.data)

# overlap between cube 1 and 3
result = multi_model_statistics([cube1, cube3], span='overlap', statistics=['mean'])['mean']
expected = iris.cube.Cube([2], dim_coords_and_dims=[(timecoord([1]), 0)])
assert np.all(result.data == expected.data)

# overlap between cube 2 and 3
result = multi_model_statistics([cube2, cube3], span='overlap', statistics=['mean'])['mean']
expected = iris.cube.Cube([2.5, 2.5], dim_coords_and_dims=[(timecoord([1, 3]), 0)])
assert np.all(result.data == expected.data)

# overlap between cube 1 and 2 and 3
result = multi_model_statistics([cube1, cube2, cube3], span='overlap', statistics=['mean'])['mean']
expected = iris.cube.Cube([2], dim_coords_and_dims=[(timecoord([1]), 0)])
assert np.all(result.data == expected.data)

###################################################################################

# full between cube 1 and 2
result = multi_model_statistics([cube1, cube2], span='full', statistics=['mean'])['mean']
expected = iris.cube.Cube([1.5, 1.5, 2], dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
assert np.all(result.data == expected.data)

# full between cube 1 and 3
result = multi_model_statistics([cube1, cube3], span='full', statistics=['mean'])['mean']
expected = iris.cube.Cube([2, 1, 3], dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
assert np.all(result.data == expected.data)

# full between cube 2 and 3
result = multi_model_statistics([cube2, cube3], span='full', statistics=['mean'])['mean']
expected = iris.cube.Cube([2.5, 2, 2.5], dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
assert np.all(result.data == expected.data)

# full between cube 1 and 2 and 3
result = multi_model_statistics([cube1, cube2, cube3], span='full', statistics=['mean'])['mean']
expected = iris.cube.Cube([2, 1.5, 2.5], dim_coords_and_dims=[(timecoord([1, 2, 3]), 0)])
assert np.all(result.data == expected.data)