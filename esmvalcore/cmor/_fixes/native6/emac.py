import logging
from ..fix import Fix


logger = logging.getLogger(__name__)


class AllVars(Fix):

  var_mappings = { #CMOR : EMAC
    "awhea" : "awhea_ave"
  }


  """ 
  def fix_file(self, var1, var2):
    # It seems it's only useful to correct filenames
    logger.info(f"\nFix file\n=========\n{var1}\n{var2}")
    return var1, var2
  """


  def fix_metadata(self, cubes):
    logger.info(f"\nFix metadata\n==================\n{cubes}\n")
    return cubes


