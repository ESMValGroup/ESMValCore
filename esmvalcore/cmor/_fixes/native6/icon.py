import logging
from ..fix import Fix


logger = logging.getLogger(__name__)


class AllVars(Fix):



  def fix_metadata(self, cubes):
#    logger.info(f"\nFix metadata\n==================\n{cubes}\n")
    for cube in cubes:
      logger.info(f"{cube.var_name}")
      # TODO fix coordinates
    return cubes


