"""Contains the base class for dataset fixes."""
from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..table import CMOR_TABLES

if TYPE_CHECKING:
    from ...config import Session
    from ..table import VariableInfo


class Fix:
    """Base class for dataset fixes."""

    def __init__(
        self,
        vardef: VariableInfo,
        extra_facets: Optional[dict] = None,
        session: Optional[Session] = None,
    ):
        """Initialize fix object.

        Parameters
        ----------
        vardef: VariableInfo
            CMOR table entry.
        extra_facets: dict, optional
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.
        session: Session, optional
            Current session which includes configuration and directory
            information.

        """
        self.vardef = vardef
        if extra_facets is None:
            extra_facets = {}
        self.extra_facets = extra_facets
        self.session = session

    def fix_file(
        self,
        filepath: Path,
        output_dir: Path,
        add_unique_suffix: bool = False,
    ) -> Path:
        """Apply fixes to the files prior to creating the cube.

        Should be used only to fix errors that prevent loading or cannot be
        fixed in the cube (e.g., those related to `missing_value` or
        `_FillValue`).

        Parameters
        ----------
        filepath: Path
            File to fix.
        output_dir: Path
            Output directory for fixed files.
        add_unique_suffix: bool, optional (default: False)
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        Path
            Path to the corrected file. It can be different from the original
            filepath if a fix has been applied, but if not it should be the
            original filepath.

        """
        return filepath

    def fix_metadata(self, cubes):
        """Apply fixes to the metadata of the cube.

        Changes applied here must not require data loading.

        These fixes should be applied before checking the metadata.

        Parameters
        ----------
        cubes: iris.cube.CubeList
            Cubes to fix.

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
            List of cubes to search.
        short_name : str or None
            Cube's variable short name. If `None`, `short name` is the class
            name.

        Raises
        ------
        Exception
            If no cube is found.

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
        raise Exception(f'Cube for variable "{short_name}" not found')

    def fix_data(self, cube):
        """Apply fixes to the data of the cube.

        These fixes should be applied before checking the data.

        Parameters
        ----------
        cube: iris.cube.Cube
            Cube to fix.

        Returns
        -------
        iris.cube.Cube
            Fixed cube. It can be a difference instance.

        """
        return cube

    def __eq__(self, other):
        """Fix equality."""
        return isinstance(self, other.__class__)

    def __ne__(self, other):
        """Fix inequality."""
        return not self.__eq__(other)

    @staticmethod
    def get_fixes(
        project: str,
        dataset: str,
        mip: str,
        short_name: str,
        extra_facets: Optional[dict] = None,
        session: Optional[Session] = None,
    ) -> list:
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
            Project of the dataset.
        dataset: str
            Name of the dataset.
        mip: str
            Variable's MIP.
        short_name: str
            Variable's short name.
        extra_facets: dict, optional
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.
        session: Session, optional
            Current session which includes configuration and directory
            information.

        Returns
        -------
        list[Fix]
            Fixes to apply for the given data.

        """
        cmor_table = CMOR_TABLES[project]
        vardef = cmor_table.get_variable(mip, short_name)

        project = project.replace('-', '_').lower()
        dataset = dataset.replace('-', '_').lower()
        short_name = short_name.replace('-', '_').lower()

        if extra_facets is None:
            extra_facets = {}

        fixes = []

        fixes_modules = []
        if project == 'cordex':
            driver = extra_facets['driver'].replace('-', '_').lower()
            extra_facets['dataset'] = dataset
            try:
                fixes_modules.append(importlib.import_module(
                    f'esmvalcore.cmor._fixes.{project}.{driver}.{dataset}'
                ))
            except ImportError:
                pass
            fixes_modules.append(importlib.import_module(
                'esmvalcore.cmor._fixes.cordex.cordex_fixes'))
        else:
            try:
                fixes_modules.append(importlib.import_module(
                    f'esmvalcore.cmor._fixes.{project}.{dataset}'))
            except ImportError:
                pass

        for fixes_module in fixes_modules:
            classes = dict(
                (name.lower(), value) for (name, value) in
                inspect.getmembers(fixes_module, inspect.isclass)
            )
            for fix_name in (short_name, mip.lower(), 'allvars'):
                if fix_name in classes:
                    fixes.append(classes[fix_name](
                        vardef, extra_facets=extra_facets, session=session
                    ))

        return fixes

    @staticmethod
    def get_fixed_filepath(
        output_dir: str | Path,
        filepath: str | Path,
        add_unique_suffix: bool = False,
    ) -> Path:
        """Get the filepath for the fixed file.

        Parameters
        ----------
        output_dir: Path
            Output directory for fixed files. Will be created if it does not
            exist yet.
        filepath: str or Path
            Original path.
        add_unique_suffix: bool, optional (default: False)
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        Path
            Path to the fixed file.

        """
        output_dir = Path(output_dir)
        if add_unique_suffix:
            parent_dir = output_dir.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_dir.name
            output_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=parent_dir))
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / Path(filepath).name
