# height 2m a differents altures
from esmvalcore.cmor.fix import Fix
from ..ncc_noresm1_m.gerics_remo2015 import Pr as BaseFix


Pr = BaseFix

# for historical + rcp45 in Pr Amon, longitudes are shifted

# cubes[0].coord('longitude').points - cubes[1].coord('longitude').points
# masked_array(
#   data=[[-360., -360., -360., ...,    0.,    0.,    0.],
#         [-360., -360., -360., ...,    0.,    0.,    0.],
#         [-360., -360., -360., ...,    0.,    0.,    0.],
#         ...,
#         [-360., -360., -360., ...,    0.,    0.,    0.],
#         [-360., -360., -360., ...,    0.,    0.,    0.],
#         [-360., -360., -360., ...,    0.,    0.,    0.]],