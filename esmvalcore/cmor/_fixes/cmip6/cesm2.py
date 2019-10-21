"""Fixes for CESM2 model."""
from ..fix import Fix
from ..shared import add_scalar_height_coord
import iris


class Tas(Fix):
    """Fixes for tas."""
    def fix_metadata(self, cubes):
        """Add height (2m) coordinate.

        Parameters
        ----------
        cube : iris.cube.CubeList

        Returns
        -------
        iris.cube.Cube

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube)
        return cubes


class BaresoilFrac(Fix):
    """Fixes for baresoilFrac"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension typebare
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        typebare = iris.coords.AuxCoord('bare_ground',
                                        standard_name='area_type',
                                        long_name='surface type',
                                        var_name='type',
                                        units='1',
                                        bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typebare)
        return cubes


class CropFrac(Fix):
    """Fixes for CropFrac"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension typecrop
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        typecrop = iris.coords.AuxCoord('crops',
                                        standard_name='area_type',
                                        long_name='Crop area type',
                                        var_name='type',
                                        units='1',
                                        bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typecrop)
        return cubes


class GrassFrac(Fix):
    """Fixes for GrassFrac"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension typecrop
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        typenatgr = iris.coords.AuxCoord('natural_grasses',
                                         standard_name='area_type',
                                         long_name='Natural grass area type',
                                         var_name='type',
                                         units='1',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typenatgr)
        return cubes

class ShrubFrac(Fix):
    """Fixes for GrassFrac"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension typeshrub
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        typeshrub = iris.coords.AuxCoord('shrubs',
                                         standard_name='area_type',
                                         long_name='Shrub area type',
                                         var_name='type',
                                         units='1',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typeshrub)
        return cubes

class TreeFrac(Fix):
    """Fixes for GrassFrac"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension typetree
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        typetree = iris.coords.AuxCoord('trees',
                                         standard_name='area_type',
                                         long_name='Tree area type',
                                         var_name='type',
                                         units='1',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(typetree)
        return cubes

class Dpco2(Fix):
    """Fixes for dpco2"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension depth0m
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        depth0m = iris.coords.AuxCoord('0.',
                                         standard_name='depth',
                                         long_name='depth',
                                         var_name='depth',
                                         units='m',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(depth0m)
        return cubes

class Fgco2(Fix):
    """Fixes for fgco2"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension depth0m
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        depth0m = iris.coords.AuxCoord('0.',
                                         standard_name='depth',
                                         long_name='depth',
                                         var_name='depth',
                                         units='m',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(depth0m)
        return cubes

class Spco2(Fix):
    """Fixes for spco2"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension depth0m
        
        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix
        
        Returns
        -------
        iris.cube.CubeList
        """
        depth0m = iris.coords.AuxCoord('0.',
                                         standard_name='depth',
                                         long_name='depth',
                                         var_name='depth',
                                         units='m',
                                         bounds=None)
        for cube in cubes:
            cube.add_aux_coord(depth0m)
        return cubes

class Abs550aer(Fix):
    """Fixes for abs550aer"""
    def fix_metadata(self, cubes):
        """
        Fix missing scalar dimension lamdba550nm
        
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
