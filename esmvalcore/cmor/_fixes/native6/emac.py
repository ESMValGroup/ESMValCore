import logging
from ..fix import Fix


logger = logging.getLogger(__name__)


class AllVars(Fix):

  var_mappings = { # EMAC : CMOR
    "awhea_ave" : "awhea"
  }


  """ 
  def fix_file(self, var1, var2):
    # It seems it's only useful to correct filenames
    logger.info(f"\nFix file\n=========\n{var1}\n{var2}")
    return var1, var2
  """


  def fix_metadata(self, cubes):
#    logger.info(f"\nFix metadata\n==================\n{cubes}\n")
    for cube in cubes:
      logger.info(f"{cube.var_name}")
      if cube.var_name in self.var_mappings.keys():
        logger.info(f"{cube.var_name} = {self.var_mappings[cube.var_name]}")
        cube.var_name = self.var_mappings[cube.var_name]
    return cubes


