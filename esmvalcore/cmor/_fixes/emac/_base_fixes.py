"""Fix base classes for EMAC on-the-fly CMORizer."""

import logging

from iris import NameConstraint
from iris.exceptions import ConstraintMismatchError

from ..native_datasets import NativeDatasetFix

logger = logging.getLogger(__name__)


class EmacFix(NativeDatasetFix):
    """Base class for all EMAC fixes."""

    def get_cube(self, cubes, var_name=None):
        """Extract single cube."""
        # If no var_name given, use the CMOR short_name
        if var_name is None:
            var_name = self.extra_facets.get('raw_name',
                                             self.vardef.short_name)

        # Convert to list if only a single var_name is given
        if isinstance(var_name, str):
            var_names = [var_name]
        else:
            var_names = var_name

        # Try to extract the variable (prioritize variables as given by the
        # list)
        for v_name in var_names:
            try:
                return cubes.extract_cube(NameConstraint(var_name=v_name))
            except ConstraintMismatchError:
                pass

        # If no cube could be extracted, raise an error
        raise ValueError(
            f"No variable of {var_names} necessary for the extraction/"
            f"derivation the CMOR variable '{self.vardef.short_name}' is "
            f"available in the input file. Hint: in case you tried to extract "
            f"a 3D variable defined on pressure levels, it might be necessary "
            f"to define the EMAC variable name in the recipe (e.g., "
            f"'raw_name: tm1_p39_cav') if the default number of pressure "
            f"levels is not available in the input file."
        )

    def add_additional_cubes(self, cubes):
        """Add additional user-defined cubes to list of cubes (in-place).

        An example use case is adding a vertical coordinate (e.g., `hyam`) to the
        dataset if the vertical coordinate data is not provided directly by the
        output file.

        Currently, the following cubes can be added:
        - 'hyam' from facet `vct_table`
        - 'hybm' from facet `vct_table`
        - 'hyai' from facet `vct_table`
        - 'hybi' from facet `vct_table`

        Note
        ----
        Files can be specified as absolute or relative (to
        ``auxiliary_data_dir`` as defined in the :ref:`user configuration
        file`) paths.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Input cubes which will be modified in place.

        Returns
        -------
        iris.cube.CubeList
            Modified cubes. The cubes are modified in place; they are just
            returned out of convenience for easy access.

        Raises
        ------
        InputFilesNotFound
            A specified file does not exist.

        """
        facets_to_consider = [
            'vct_table',
        ]
        for facet in facets_to_consider:
            if facet not in self.extra_facets:
                continue
            path_to_add = self._get_path_from_facet(facet)
            logger.debug("Adding cubes from %s", path_to_add)
            new_cubes = self._load_cubes(path_to_add)
            cubes.extend(new_cubes)

        return cubes

class NegateData(EmacFix):
    """Base fix to negate data."""

    def fix_data(self, cube):
        """Fix data."""
        cube.data = -cube.core_data()
        return cube
