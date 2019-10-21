"""Fixes for CNRM-ESM2-1 model."""
from ..fix import Fix
from ..shared import add_scalar_height_coord
import iris


class Abs550aer(Fix):
    """Fixes for abs550aer"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension lambda550nm
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        lambda550nm = iris.coords.AuxCoord('550.0',
                                         standard_name='radiation_wavelength',
                                         long_name='Radiation Wavelength 550 nanometers',
                                         var_name='wavelength',
                                         units='nm',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(lambda550nm)
        return cubes

class Od550aer(Fix):
    """Fixes for od550aer"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension lambda550nm
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        lambda550nm = iris.coords.AuxCoord('550.0',
                                         standard_name='radiation_wavelength',
                                         long_name='Radiation Wavelength 550 nanometers',
                                         var_name='wavelength',
                                         units='nm',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(lambda550nm)
        return cubes
