"""Contains the base class for dataset fixes."""
import importlib
import os
import inspect

from ..table import CMOR_TABLES


class Fix:
    """Base class for dataset fixes."""
    def __init__(self, vardef, extra_facets=None):
        """Initialize fix object.

        Parameters
        ----------
        vardef: str
            CMOR table entry
        extra_facets: dict, optional
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.
        """
        self.vardef = vardef
        if extra_facets is None:
            extra_facets = {}
        self.extra_facets = extra_facets

    def fix_file(self, filepath, output_dir):
        """Apply fixes to the files prior to creating the cube.

        Should be used only to fix errors that prevent loading or can
        not be fixed in the cube (i.e. those related with missing_value
        and _FillValue)

        Parameters
        ----------
        filepath: str
            file to fix
        output_dir: str
            path to the folder to store the fixed files, if required

        Returns
        -------
        str
            Path to the corrected file. It can be different from the original
            filepath if a fix has been applied, but if not it should be the
            original filepath
        """
        return filepath

    def fix_metadata(self, cubes):
        """Apply fixes to the metadata of the cube.

        Changes applied here must not require data loading.

        These fixes should be applied before checking the metadata.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Cubes to fix

        Returns
        -------
        iris.cube.CubeList
            Fixed cubes. They can be different instances.
        """
        return cubes

    def get_cube_from_list(self, cubes, short_name=None):
        """Get a cube from the list with a given short name.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            List of cubes to search
        short_name : str
            Cube's variable short name. If None, short name is the class name

        Raises
        ------
        Exception
            If no cube is found

        Returns
        -------
        iris.Cube
            Variable's cube
        """
        if short_name is None:
            short_name = self.vardef.short_name
        for cube in cubes:
            if cube.var_name == short_name:
                return cube
        raise Exception('Cube for variable "{}" not found'.format(short_name))

    def fix_data(self, cube):
        """Apply fixes to the data of the cube.

        These fixes should be applied before checking the data.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix

        Returns
        -------
        iris.cube.Cube
            Fixed cube. It can be a difference instance.
        """
        return cube

    def __eq__(self, other):
        return isinstance(self, other.__class__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def get_fixes(project, dataset, mip, short_name, extra_facets=None):
        """Get the fixes that must be applied for a given dataset.

        It will look for them at the module
        esmvalcore.cmor._fixes.PROJECT in the file DATASET, and get
        the classes named allvars (which should be use for fixes that are
        present in all the variables of a dataset, i.e. bad name for the time
        coordinate) and VARIABLE (which should be use for fixes for the
        specific variable).

        Project, dataset and variable names will have '-' replaced by '_'
        before checking because it is not possible to use the character '-' in
        python names.

        Parameters
        ----------
        project: str
        dataset: str
        mip: str
        short_name: str
        extra_facets: dict, optional
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.

        Returns
        -------
        list(Fix)
            Fixes to apply for the given data
        """
        cmor_table = CMOR_TABLES[project]
        vardef = cmor_table.get_variable(mip, short_name)

        project = project.replace('-', '_').lower()
        dataset = dataset.replace('-', '_').lower()
        short_name = short_name.replace('-', '_').lower()

        if extra_facets is None:
            extra_facets = {}

        fixes = []
        try:
            fixes_module = importlib.import_module(
                'esmvalcore.cmor._fixes.{0}.{1}'.format(project, dataset))

            classes = inspect.getmembers(fixes_module, inspect.isclass)
            classes = dict((name.lower(), value) for name, value in classes)
            for fix_name in (short_name, mip.lower(), 'allvars'):
                try:
                    fixes.append(classes[fix_name](vardef, extra_facets))
                except KeyError:
                    pass
        except ImportError:
            pass
        return fixes

    @staticmethod
    def get_fixed_filepath(output_dir, filepath):
        """Get the filepath for the fixed file.

        Parameters
        ----------
        var_path: str
            Original path

        Returns
        -------
        str
            Path to the fixed file
        """
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        return os.path.join(output_dir, os.path.basename(filepath))
