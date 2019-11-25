"""Fixes for all datasets """
from ..shared import add_scalar_height_coord
from ..fix import Fix
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
        lambda550nm = iris.coords.AuxCoord(550.0,
                                         standard_name='radiation_wavelength',
                                         long_name='Radiation Wavelength 550 nanometers',
                                         var_name='wavelength',
                                         units='nm',
                                         bounds=None)
        for cube in cubes:
            try:
                cube.coord(standard_name="radiation_wavelength")
            except iris.exceptions.CoordinateNotFoundError:
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
        lambda550nm = iris.coords.AuxCoord(550.0,
                                         standard_name='radiation_wavelength',
                                         long_name='Radiation Wavelength 550 nanometers',
                                         var_name='wavelength',
                                         units='nm',
                                         bounds=None)
        for cube in cubes:
            try:
                cube.coord(standard_name="radiation_wavelength")
            except iris.exceptions.CoordinateNotFoundError:
                cube.add_aux_coord(lambda550nm)
        return cubes

class Od550lt1aer(Fix):
    """Fixes for od550lt1aer"""
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
        lambda550nm = iris.coords.AuxCoord(550.0,
                                         standard_name='radiation_wavelength',
                                         long_name='Radiation Wavelength 550 nanometers',
                                         var_name='wavelength',
                                         units='nm',
                                         bounds=None)
        for cube in cubes:
            try:
                cube.coord(standard_name="radiation_wavelength")
            except iris.exceptions.CoordinateNotFoundError:
                cube.add_aux_coord(lambda550nm)
        return cubes

class Mrsos(Fix):
    """Fixes for Mrsos"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension sdepth1
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        Returns
        -------
        iris.cube.CubeList
        """
        sdepth1 = iris.coords.AuxCoord(0.05,
                                         standard_name='depth',
                                         long_name='depth',
                                         var_name='depth',
                                         units='m',
                                         bounds=[0.0, 0.1])
        for cube in cubes:
            try:
                cube.coord("depth")
            except iris.exceptions.CoordinateNotFoundError:
                cube.add_aux_coord(sdepth1)
        return cubes
class Siconc(Fix):
    """Fixes for siconc"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension sdepth1
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        Returns
        -------
        iris.cube.CubeList
        """
        typesi = iris.coords.AuxCoord("sea_ice",
                                         standard_name='area_type',
                                         long_name='Sea Ice area type',
                                         var_name='type',
                                         units='',
                                         bounds=None)
        for cube in cubes:
            try:
                cube.coord(standard_name="area_type")
            except iris.exceptions.CoordinateNotFoundError:
                cube.add_aux_coord(typesi)
        return cubes
class Tas(Fix):
    """Fixes for tas"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension height2m
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        height2m = iris.coords.AuxCoord(2.0,
                                        standard_name='height',
                                        long_name='height',
                                        var_name='height',
                                        units='m',
                                        bounds=None)
        for cube in cubes:
            try:
                cube.coord("height")
            except iris.exceptions.CoordinateNotFoundError:
                cube.add_aux_coord(height2m)
        return cubes

