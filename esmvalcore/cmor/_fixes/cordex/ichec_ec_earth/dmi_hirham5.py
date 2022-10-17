
from esmvalcore.cmor.fix import Fix

#rcp85 data does not have grid_lat and grid_lon defined
# Traceback (most recent call last):
#   File "<string>", line 1, in <module>
#   File "/home/b/b381943/ESMValCore/esmvalcore/preprocessor/_io.py", line 220, in _get_concatenation_error
#     raise ValueError(f'Can not concatenate cubes: {msg}')
# ValueError: Can not concatenate cubes: failed to concatenate into a single cube.
#   Dimension coordinates differ: grid_latitude, grid_longitude, time != time