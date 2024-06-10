"""Contains the base class for dataset fixes."""
from __future__ import annotations

import importlib
import inspect
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import dask
import numpy as np
from cf_units import Unit
from iris.coords import Coord, CoordExtent
from iris.cube import Cube, CubeList
from iris.exceptions import UnitConversionError
from iris.util import reverse

from esmvalcore.cmor._utils import (
    _get_alternative_generic_lev_coord,
    _get_generic_lev_coord_names,
    _get_new_generic_level_coord,
    _get_simplified_calendar,
    _get_single_cube,
)
from esmvalcore.cmor.fixes import get_time_bounds
from esmvalcore.cmor.table import get_var_info
from esmvalcore.iris_helpers import has_unstructured_grid

if TYPE_CHECKING:
    from esmvalcore.cmor.table import CoordinateInfo, VariableInfo
    from esmvalcore.config import Session

logger = logging.getLogger(__name__)
generic_fix_logger = logging.getLogger(f'{__name__}.genericfix')


class Fix:
    """Base class for dataset fixes."""

    def __init__(
        self,
        vardef: VariableInfo,
        extra_facets: Optional[dict] = None,
        session: Optional[Session] = None,
        frequency: Optional[str] = None,
    ) -> None:
        """Initialize fix object.

        Parameters
        ----------
        vardef:
            CMOR table entry of the variable.
        extra_facets:
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.
        session:
            Current session which includes configuration and directory
            information.
        frequency:
            Expected frequency of the variable. If not given, use the one from
            the CMOR table entry of the variable.

        """
        self.vardef = vardef
        if extra_facets is None:
            extra_facets = {}
        self.extra_facets = extra_facets
        self.session = session
        if frequency is None and self.vardef is not None:
            frequency = self.vardef.frequency
        self.frequency = frequency

    def fix_file(
        self,
        filepath: Path,
        output_dir: Path,
        add_unique_suffix: bool = False,
    ) -> str | Path:
        """Apply fixes to the files prior to creating the cube.

        Should be used only to fix errors that prevent loading or cannot be
        fixed in the cube (e.g., those related to `missing_value` or
        `_FillValue`).

        Parameters
        ----------
        filepath:
            File to fix.
        output_dir:
            Output directory for fixed files.
        add_unique_suffix:
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        str or pathlib.Path
            Path to the corrected file. It can be different from the original
            filepath if a fix has been applied, but if not it should be the
            original filepath.

        """
        return filepath

    def fix_metadata(self, cubes: Sequence[Cube]) -> Sequence[Cube]:
        """Apply fixes to the metadata of the cube.

        Changes applied here must not require data loading.

        These fixes should be applied before checking the metadata.

        Parameters
        ----------
        cubes:
            Cubes to fix.

        Returns
        -------
        Iterable[iris.cube.Cube]
            Fixed cubes. They can be different instances.

        """
        return cubes

    def get_cube_from_list(
        self,
        cubes: CubeList,
        short_name: Optional[str] = None,
    ) -> Cube:
        """Get a cube from the list with a given short name.

        Parameters
        ----------
        cubes:
            List of cubes to search.
        short_name:
            Cube's variable short name. If `None`, `short name` is the class
            name.

        Raises
        ------
        Exception
            No cube is found.

        Returns
        -------
        iris.cube.Cube
            Variable's cube.

        """
        if short_name is None:
            short_name = self.vardef.short_name
        for cube in cubes:
            if cube.var_name == short_name:
                return cube
        raise Exception(f'Cube for variable "{short_name}" not found')

    def fix_data(self, cube: Cube) -> Cube:
        """Apply fixes to the data of the cube.

        These fixes should be applied before checking the data.

        Parameters
        ----------
        cube:
            Cube to fix.

        Returns
        -------
        iris.cube.Cube
            Fixed cube. It can be a difference instance.

        """
        return cube

    def __eq__(self, other: Any) -> bool:
        """Fix equality."""
        return isinstance(self, other.__class__)

    def __ne__(self, other: Any) -> bool:
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
        frequency: Optional[str] = None,
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

        In addition, generic fixes for all datasets are added.

        Parameters
        ----------
        project:
            Project of the dataset.
        dataset:
            Name of the dataset.
        mip:
            Variable's MIP.
        short_name:
            Variable's short name.
        extra_facets:
            Extra facets are mainly used for data outside of the big projects
            like CMIP, CORDEX, obs4MIPs. For details, see :ref:`extra_facets`.
        session:
            Current session which includes configuration and directory
            information.
        frequency:
            Expected frequency of the variable. If not given, use the one from
            the CMOR table entry of the variable.

        Returns
        -------
        list[Fix]
            Fixes to apply for the given data.

        """
        vardef = get_var_info(project, mip, short_name)

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
                    fixes.append(
                        classes[fix_name](
                            vardef,
                            extra_facets=extra_facets,
                            session=session,
                            frequency=frequency,
                        )
                    )

        # Always perform generic fixes for all datasets
        fixes.append(
            GenericFix(
                vardef,  # type: ignore
                extra_facets=extra_facets,
                session=session,
                frequency=frequency,
            )
        )

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
        output_dir:
            Output directory for fixed files. Will be created if it does not
            exist yet.
        filepath:
            Original path.
        add_unique_suffix:
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


class GenericFix(Fix):
    """Class providing generic fixes for all datasets."""

    def fix_metadata(self, cubes: Sequence[Cube]) -> CubeList:
        """Fix cube metadata.

        Parameters
        ----------
        cubes:
            Cubes to be fixed.

        Returns
        -------
        CubeList
            Fixed cubes.

        """
        # Make sure the this fix also works when no extra_facets are given
        if 'project' in self.extra_facets and 'dataset' in self.extra_facets:
            dataset_str = (
                f"{self.extra_facets['project']}:"
                f"{self.extra_facets['dataset']}"
            )
        else:
            dataset_str = None

        # The following fixes are designed to operate on the actual cube that
        # corresponds to the variable. Thus, it needs to be assured (possibly
        # by prior dataset-specific fixes) that the cubes here contain only one
        # relevant cube.
        cube = _get_single_cube(
            cubes, self.vardef.short_name, dataset_str=dataset_str
        )

        cube = self._fix_standard_name(cube)
        cube = self._fix_long_name(cube)
        cube = self._fix_psu_units(cube)
        cube = self._fix_units(cube)

        cube = self._fix_regular_coord_names(cube)
        cube = self._fix_alternative_generic_level_coords(cube)
        cube = self._fix_coords(cube)
        cube = self._fix_time_coord(cube)

        return CubeList([cube])

    def fix_data(self, cube: Cube) -> Cube:
        """Fix cube data.

        Parameters
        ----------
        cube:
            Cube to be fixed.

        Returns
        -------
        Cube
            Fixed cube.

        """
        return cube

    @staticmethod
    def _msg_suffix(cube: Cube) -> str:
        """Get prefix for log messages."""
        if 'source_file' in cube.attributes:
            return f"\n(for file {cube.attributes['source_file']})"
        return f"\n(for variable {cube.var_name})"

    def _debug_msg(self, cube: Cube, msg: str, *args) -> None:
        """Print debug message."""
        msg += self._msg_suffix(cube)
        generic_fix_logger.debug(msg, *args)

    def _warning_msg(self, cube: Cube, msg: str, *args) -> None:
        """Print debug message."""
        msg += self._msg_suffix(cube)
        generic_fix_logger.warning(msg, *args)

    @staticmethod
    def _set_range_in_0_360(array: np.ndarray) -> np.ndarray:
        """Convert longitude coordinate to [0, 360]."""
        return (array + 360.0) % 360.0

    def _reverse_coord(self, cube: Cube, coord: Coord) -> tuple[Cube, Coord]:
        """Reverse cube along a given coordinate."""
        if coord.ndim == 1:
            cube = reverse(cube, cube.coord_dims(coord))
            coord = cube.coord(var_name=coord.var_name)
            if coord.has_bounds():
                bounds = coord.core_bounds()
                right_bounds = bounds[:-2, 1]
                left_bounds = bounds[1:-1, 0]
                if np.all(right_bounds != left_bounds):
                    coord.bounds = np.fliplr(bounds)
            self._debug_msg(
                cube,
                "Coordinate %s values have been reversed",
                coord.var_name,
            )
        return (cube, coord)

    def _get_effective_units(self) -> str:
        """Get effective units."""
        if self.vardef.units.lower() == 'psu':
            return '1'
        return self.vardef.units

    def _fix_units(self, cube: Cube) -> Cube:
        """Fix cube units."""
        if self.vardef.units:
            units = self._get_effective_units()

            # We use str(cube.units) in the following to catch `degrees` !=
            # `degrees_north`
            if str(cube.units) != units:
                old_units = cube.units
                try:
                    cube.convert_units(units)
                except (ValueError, UnitConversionError):
                    self._warning_msg(
                        cube,
                        "Failed to convert cube units from '%s' to '%s'",
                        old_units,
                        units,
                    )
                else:
                    self._warning_msg(
                        cube,
                        "Converted cube units from '%s' to '%s'",
                        old_units,
                        units,
                    )
        return cube

    def _fix_standard_name(self, cube: Cube) -> Cube:
        """Fix standard_name."""
        # Do not change empty standard names
        if not self.vardef.standard_name:
            return cube

        if cube.standard_name != self.vardef.standard_name:
            self._warning_msg(
                cube,
                "Standard name changed from '%s' to '%s'",
                cube.standard_name,
                self.vardef.standard_name,
            )
            cube.standard_name = self.vardef.standard_name

        return cube

    def _fix_long_name(self, cube: Cube) -> Cube:
        """Fix long_name."""
        # Do not change empty long names
        if not self.vardef.long_name:
            return cube

        if cube.long_name != self.vardef.long_name:
            self._warning_msg(
                cube,
                "Long name changed from '%s' to '%s'",
                cube.long_name,
                self.vardef.long_name,
            )
            cube.long_name = self.vardef.long_name

        return cube

    def _fix_psu_units(self, cube: Cube) -> Cube:
        """Fix psu units."""
        if cube.attributes.get('invalid_units', '').lower() == 'psu':
            cube.units = '1'
            cube.attributes.pop('invalid_units')
            self._debug_msg(cube, "Units converted from 'psu' to '1'")
        return cube

    def _fix_regular_coord_names(self, cube: Cube) -> Cube:
        """Fix regular (non-generic-level) coordinate names."""
        for cmor_coord in self.vardef.coordinates.values():
            if cmor_coord.generic_level:
                continue  # Ignore generic level coordinate in this function
            if cube.coords(var_name=cmor_coord.out_name):
                continue  # Coordinate found -> fine here
            if cube.coords(cmor_coord.standard_name):
                cube_coord = cube.coord(cmor_coord.standard_name)
                self._fix_cmip6_multidim_lat_lon_coord(
                    cube, cmor_coord, cube_coord
                )
        return cube

    def _fix_alternative_generic_level_coords(self, cube: Cube) -> Cube:
        """Fix alternative generic level coordinates."""
        # Avoid overriding existing variable information
        cmor_var_coordinates = self.vardef.coordinates.copy()
        for (coord_name, cmor_coord) in cmor_var_coordinates.items():
            if not cmor_coord.generic_level:
                continue  # Ignore non-generic-level coordinates
            if not cmor_coord.generic_lev_coords:
                continue  # Cannot fix anything without coordinate info

            # Extract names of the actual generic level coordinates present in
            # the cube (e.g., `hybrid_height`, `standard_hybrid_sigma`)
            (standard_name, out_name, name) = _get_generic_lev_coord_names(
                cube, cmor_coord
            )

            # Make sure to update variable information with actual generic
            # level coordinate if one has been found; this is necessary for
            # subsequent fixes
            if standard_name:
                new_generic_level_coord = _get_new_generic_level_coord(
                    self.vardef, cmor_coord, coord_name, name
                )
                self.vardef.coordinates[coord_name] = new_generic_level_coord
                self._debug_msg(
                    cube,
                    "Generic level coordinate %s will be checked against %s "
                    "coordinate information",
                    coord_name,
                    name,
                )

            # If a generic level coordinate has been found, we don't need to
            # look for alternatives
            if standard_name or out_name:
                continue

            # Search for alternative coordinates (i.e., regular level
            # coordinates); if none found, do nothing
            try:
                (alternative_coord,
                 cube_coord) = _get_alternative_generic_lev_coord(
                    cube, coord_name, self.vardef.table_type
                )
            except ValueError:  # no alternatives found
                continue

            # Fix alternative coord
            (cube, cube_coord) = self._fix_coord(
                cube, alternative_coord, cube_coord
            )

        return cube

    def _fix_cmip6_multidim_lat_lon_coord(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix CMIP6 multidimensional latitude and longitude coordinates."""
        is_cmip6_multidim_lat_lon = all([
            'CMIP6' in self.vardef.table_type,
            cube_coord.ndim > 1,
            cube_coord.standard_name in ('latitude', 'longitude'),
        ])
        if is_cmip6_multidim_lat_lon:
            self._debug_msg(
                cube,
                "Multidimensional %s coordinate is not set in CMOR standard, "
                "ESMValTool will change the original value of '%s' to '%s' to "
                "match the one-dimensional case",
                cube_coord.standard_name,
                cube_coord.var_name,
                cmor_coord.out_name,
            )
            cube_coord.var_name = cmor_coord.out_name

    def _fix_coord_units(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix coordinate units."""
        if not cmor_coord.units:
            return

        # We use str(cube_coord.units) in the following to catch `degrees` !=
        # `degrees_north`
        if str(cube_coord.units) != cmor_coord.units:
            old_units = cube_coord.units
            try:
                cube_coord.convert_units(cmor_coord.units)
            except (ValueError, UnitConversionError):
                self._warning_msg(
                    cube,
                    "Failed to convert units of coordinate %s from '%s' to "
                    "'%s'",
                    cmor_coord.out_name,
                    old_units,
                    cmor_coord.units,
                )
            else:
                self._warning_msg(
                    cube,
                    "Coordinate %s units '%s' converted to '%s'",
                    cmor_coord.out_name,
                    old_units,
                    cmor_coord.units,
                )

    def _fix_requested_coord_values(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix requested coordinate values."""
        if not cmor_coord.requested:
            return

        # Cannot fix non-1D points
        if cube_coord.core_points().ndim != 1:
            return

        # Get requested CMOR values
        try:
            cmor_points = np.array(cmor_coord.requested, dtype=float)
        except ValueError:
            return

        # Align coordinate points with CMOR values if possible
        if cube_coord.core_points().shape != cmor_points.shape:
            return
        atol = 1e-7 * np.mean(cmor_points)
        align_coords = np.allclose(
            cube_coord.core_points(),
            cmor_points,
            rtol=1e-7,
            atol=atol,
        )
        if align_coords:
            cube_coord.points = cmor_points
            self._debug_msg(
                cube,
                "Aligned %s points with CMOR points",
                cmor_coord.out_name,
            )

    def _fix_longitude_0_360(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix longitude coordinate to be in [0, 360]."""
        if not cube_coord.standard_name == 'longitude':
            return (cube, cube_coord)

        points = cube_coord.core_points()
        min_, max_ = dask.compute(points.min(), points.max())

        # Do not apply fixes when values are inside of valid range [0, 360]
        if min_ >= 0.0 and max_ <= 360.0:
            return (cube, cube_coord)

        # Cannot fix longitudes outside [-360, 720]
        if min_ < -360.0:
            return (cube, cube_coord)
        if max_ > 720.0:
            return (cube, cube_coord)

        # cube.intersection only works for cells with 0 or 2 bounds
        # Note: nbounds==0 means there are no bounds given, nbounds==2
        # implies a regular grid with bounds in the grid direction,
        # nbounds>2 implies an irregular grid with bounds given as vertices
        # of the cell polygon.
        if cube_coord.ndim == 1 and cube_coord.nbounds in (0, 2):
            lon_extent = CoordExtent(cube_coord, 0.0, 360., True, False)
            cube = cube.intersection(lon_extent)
        else:
            new_lons = cube_coord.core_points().copy()
            new_lons = self._set_range_in_0_360(new_lons)

            if cube_coord.core_bounds() is None:
                new_bounds = None
            else:
                new_bounds = cube_coord.core_bounds().copy()
                new_bounds = self._set_range_in_0_360(new_bounds)

            new_coord = cube_coord.copy(new_lons, new_bounds)
            dims = cube.coord_dims(cube_coord)
            cube.remove_coord(cube_coord)
            cube.add_aux_coord(new_coord, dims)
        new_coord = cube.coord(var_name=cmor_coord.out_name)
        self._debug_msg(cube, "Shifted longitude to [0, 360]")

        return (cube, new_coord)

    def _fix_coord_bounds(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> None:
        """Fix coordinate bounds."""
        if cmor_coord.must_have_bounds != 'yes' or cube_coord.has_bounds():
            return

        # Skip guessing bounds for unstructured grids
        if has_unstructured_grid(cube) and cube_coord.standard_name in (
                'latitude', 'longitude'):
            self._debug_msg(
                cube,
                "Will not guess bounds for coordinate %s of unstructured grid",
                cube_coord.var_name,
            )
            return

        try:
            cube_coord.guess_bounds()
            self._warning_msg(
                cube,
                "Added guessed bounds to coordinate %s",
                cube_coord.var_name,
            )
        except ValueError as exc:
            self._warning_msg(
                cube,
                "Cannot guess bounds for coordinate %s: %s",
                cube.var_name,
                str(exc),
            )

    def _fix_coord_direction(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix coordinate direction (increasing vs. decreasing)."""
        # Skip fix for a variety of reasons
        if cube_coord.ndim > 1:
            return (cube, cube_coord)
        if cube_coord.dtype.kind == 'U':
            return (cube, cube_coord)
        if has_unstructured_grid(cube) and cube_coord.standard_name in (
                'latitude', 'longitude'
        ):
            return (cube, cube_coord)
        if len(cube_coord.core_points()) == 1:
            return (cube, cube_coord)
        if not cmor_coord.stored_direction:
            return (cube, cube_coord)

        # Fix coordinates with wrong direction
        if cmor_coord.stored_direction == 'increasing':
            if cube_coord.core_points()[0] > cube_coord.core_points()[1]:
                (cube, cube_coord) = self._reverse_coord(cube, cube_coord)
        elif cmor_coord.stored_direction == 'decreasing':
            if cube_coord.core_points()[0] < cube_coord.core_points()[1]:
                (cube, cube_coord) = self._reverse_coord(cube, cube_coord)

        return (cube, cube_coord)

    def _fix_time_units(self, cube: Cube, cube_coord: Coord) -> None:
        """Fix time units in cube and attributes."""
        # Fix cube units
        old_units = cube_coord.units
        cube_coord.convert_units(
            Unit(
                'days since 1850-1-1 00:00:00',
                calendar=cube_coord.units.calendar,
            )
        )
        simplified_cal = _get_simplified_calendar(cube_coord.units.calendar)
        cube_coord.units = Unit(
            cube_coord.units.origin, calendar=simplified_cal
        )

        # Fix units of time-related cube attributes
        attrs = cube.attributes
        parent_time = 'parent_time_units'
        if parent_time in attrs:
            if attrs[parent_time] in 'no parent':
                pass
            else:
                try:
                    parent_units = Unit(attrs[parent_time], simplified_cal)
                except ValueError:
                    pass
                else:
                    attrs[parent_time] = 'days since 1850-1-1 00:00:00'

                    branch_parent = 'branch_time_in_parent'
                    if branch_parent in attrs:
                        attrs[branch_parent] = parent_units.convert(
                            attrs[branch_parent], cube_coord.units)

                    branch_child = 'branch_time_in_child'
                    if branch_child in attrs:
                        attrs[branch_child] = old_units.convert(
                            attrs[branch_child], cube_coord.units)

    def _fix_time_bounds(self, cube: Cube, cube_coord: Coord) -> None:
        """Fix time bounds."""
        times = {'time', 'time1', 'time2', 'time3'}
        key = times.intersection(self.vardef.coordinates)
        cmor = self.vardef.coordinates[' '.join(key)]
        if cmor.must_have_bounds == 'yes' and not cube_coord.has_bounds():
            cube_coord.bounds = get_time_bounds(cube_coord, self.frequency)
            self._warning_msg(
                cube,
                "Added guessed bounds to coordinate %s",
                cube_coord.var_name,
            )

    def _fix_time_coord(self, cube: Cube) -> Cube:
        """Fix time coordinate."""
        # Make sure to get dimensional time coordinate if possible
        if cube.coords('time', dim_coords=True):
            cube_coord = cube.coord('time', dim_coords=True)
        elif cube.coords('time'):
            cube_coord = cube.coord('time')
        else:
            return cube

        # Cannot fix wrong time that are not references
        if not cube_coord.units.is_time_reference():
            return cube

        # Fix time units
        self._fix_time_units(cube, cube_coord)

        # Remove time_origin from coordinate attributes
        cube_coord.attributes.pop('time_origin', None)

        # Fix time bounds
        self._fix_time_bounds(cube, cube_coord)

        return cube

    def _fix_coord(
        self,
        cube: Cube,
        cmor_coord: CoordinateInfo,
        cube_coord: Coord,
    ) -> tuple[Cube, Coord]:
        """Fix non-time coordinate."""
        self._fix_coord_units(cube, cmor_coord, cube_coord)
        (cube, cube_coord) = self._fix_longitude_0_360(
            cube, cmor_coord, cube_coord
        )
        self._fix_coord_bounds(cube, cmor_coord, cube_coord)
        (cube, cube_coord) = self._fix_coord_direction(
            cube, cmor_coord, cube_coord
        )
        self._fix_requested_coord_values(cube, cmor_coord, cube_coord)
        return (cube, cube_coord)

    def _fix_coords(self, cube: Cube) -> Cube:
        """Fix non-time coordinates."""
        for cmor_coord in self.vardef.coordinates.values():

            # Cannot fix generic level coords with no unique CMOR information
            if cmor_coord.generic_level and not cmor_coord.out_name:
                continue

            # Try to get coordinate from cube; if it does not exists, skip
            if not cube.coords(var_name=cmor_coord.out_name):
                continue
            cube_coord = cube.coord(var_name=cmor_coord.out_name)

            # Fixes for time coord are done separately
            if cube_coord.var_name == 'time':
                continue

            # Fixes
            (cube, cube_coord) = self._fix_coord(cube, cmor_coord, cube_coord)

        return cube
