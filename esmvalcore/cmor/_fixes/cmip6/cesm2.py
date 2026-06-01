"""Fixes for CESM2 model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.coords
import ncdata
import ncdata.netcdf4
import numpy as np

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.shared import (
    add_scalar_depth_coord,
    add_scalar_height_coord,
    add_scalar_typeland_coord,
    add_scalar_typesea_coord,
    fix_ocean_depth_coord,
)
from esmvalcore.iris_helpers import dataset_to_iris

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from iris.cube import Cube


class Cl(Fix):
    """Fixes for ``cl``."""

    @staticmethod
    def _fix_formula_terms(dataset: ncdata.NcData) -> None:
        """Fix ``formula_terms`` attribute."""
        lev = dataset.variables["lev"]
        lev.set_attrval("formula_terms", "p0: p0 a: a b: b ps: ps")
        lev.set_attrval(
            "standard_name",
            "atmosphere_hybrid_sigma_pressure_coordinate",
        )
        lev.set_attrval("units", "1")
        dataset.variables["lev_bnds"].attributes.pop("units")

    def fix_file(
        self,
        file: Path,
        output_dir: Path,  # noqa: ARG002
        add_unique_suffix: bool = False,  # noqa: ARG002
    ) -> Path | Sequence[Cube]:
        """Fix hybrid pressure coordinate.

        Adds missing ``formula_terms`` attribute to file.

        Note
        ----
        Fixing this with :mod:`iris` in ``fix_metadata`` or ``fix_data`` is
        **not** possible, since the bounds of the vertical coordinates ``a``
        and ``b`` are not present in the loaded :class:`iris.cube.CubeList`,
        even when :func:`iris.load_raw` is used.

        Parameters
        ----------
        file : str
            Path to the original file.
        output_dir: Path
            Output directory for fixed files.
        add_unique_suffix: bool, optional (default: False)
            Adds a unique suffix to `output_dir` for thread safety.

        Returns
        -------
        str
            Path to the fixed file.

        """
        dataset = ncdata.netcdf4.from_nc4(
            file,
            # Use iris-style chunks to avoid mismatching chunks between data
            # and derived coordinates, as the latter are automatically rechunked
            # by iris.
            dim_chunks={
                "time": "auto",
                "lev": None,
                "lat": None,
                "lon": None,
                "nbnd": None,
            },
        )
        self._fix_formula_terms(dataset)

        # Correct order of bounds data
        a_bnds = dataset.variables["a_bnds"]
        a_bnds.data = a_bnds.data[::-1, :]
        b_bnds = dataset.variables["b_bnds"]
        b_bnds.data = b_bnds.data[::-1, :]

        # Correct lev and lev_bnds data
        lev = dataset.variables["lev"]
        lev.data = dataset.variables["a"].data + dataset.variables["b"].data
        lev_bnds = dataset.variables["lev_bnds"]
        lev_bnds.data = (
            dataset.variables["a_bnds"].data + dataset.variables["b_bnds"].data
        )
        # Remove 'title' attribute that duplicates long name
        for var_name in dataset.variables:
            dataset.variables[var_name].attributes.pop("title", None)
        return [self.get_cube_from_list(dataset_to_iris(dataset, file))]


Cli = Cl


Clw = Cl


class Fgco2(Fix):
    """Fixes for fgco2."""

    def fix_metadata(self, cubes):
        """Add depth (0m) coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_depth_coord(cube)
        return cubes


class Prw(Fix):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            for coord_name in ["latitude", "longitude"]:
                coord = cube.coord(coord_name)
                if not coord.has_bounds():
                    coord.guess_bounds()
                coord.bounds = np.round(
                    coord.core_bounds().astype(np.float64),
                    4,
                )

        return cubes


class Tas(Prw):
    """Fixes for tas."""

    def fix_metadata(self, cubes):
        """
        Add height (2m) coordinate and time coordinate.

        Fix also done for prw.
        Fix latitude_bounds and longitude_bounds data type and round to 4 d.p.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList, iris.cube.CubeList

        """
        super().fix_metadata(cubes)
        # Specific code for tas
        cube = self.get_cube_from_list(cubes)
        add_scalar_height_coord(cube)
        new_list = iris.cube.CubeList()
        for cube in cubes:
            try:
                old_time = cube.coord("time")
            except iris.exceptions.CoordinateNotFoundError:
                new_list.append(cube)
            else:
                if old_time.is_monotonic():
                    new_list.append(cube)
                else:
                    time_units = old_time.units
                    time_data = old_time.points

                    # erase erroneously copy-pasted points
                    time_diff = np.diff(time_data)
                    idx_neg = np.where(time_diff <= 0.0)[0]
                    while len(idx_neg) > 0:
                        time_data = np.delete(time_data, idx_neg[0] + 1)
                        time_diff = np.diff(time_data)
                        idx_neg = np.where(time_diff <= 0.0)[0]

                    # create the new time coord
                    new_time = iris.coords.DimCoord(
                        time_data,
                        standard_name="time",
                        var_name="time",
                        units=time_units,
                    )

                    # create a new cube with the right shape
                    dims = (
                        time_data.shape[0],
                        cube.coord("latitude").shape[0],
                        cube.coord("longitude").shape[0],
                    )
                    data = cube.data
                    new_data = np.ma.append(
                        data[: dims[0] - 1, :, :],
                        data[-1, :, :],
                    )
                    new_data = new_data.reshape(dims)

                    tmp_cube = iris.cube.Cube(
                        new_data,
                        standard_name=cube.standard_name,
                        long_name=cube.long_name,
                        var_name=cube.var_name,
                        units=cube.units,
                        attributes=cube.attributes,
                        cell_methods=cube.cell_methods,
                        dim_coords_and_dims=[
                            (new_time, 0),
                            (cube.coord("latitude"), 1),
                            (cube.coord("longitude"), 2),
                        ],
                    )

                    new_list.append(tmp_cube)

        return new_list


class Sftlf(Fix):
    """Fixes for sftlf."""

    def fix_metadata(self, cubes):
        """Add typeland coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typeland_coord(cube)
        return cubes


class Sftof(Fix):
    """Fixes for sftof."""

    def fix_metadata(self, cubes):
        """Add typesea coordinate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        add_scalar_typesea_coord(cube)
        return cubes


Siconc = SiconcFixScalarCoord


class Tos(Fix):
    """Fixes for tos."""

    def fix_metadata(self, cubes):
        """
        Round times to 1 d.p. for monthly means.

        Required to get hist-GHG and ssp245-GHG Omon tos to concatenate.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)

        for cube in cubes:
            if cube.attributes["mipTable"] == "Omon":
                cube.coord("time").points = np.round(
                    cube.coord("time").points,
                    1,
                )
        return cubes


class Omon(Fix):
    """Fixes for ocean variables."""

    def fix_metadata(self, cubes):
        """Fix ocean depth coordinate.

        Parameters
        ----------
        cubes: iris CubeList
            List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            if cube.coords(axis="Z"):
                z_coord = cube.coord(axis="Z")

                # Only points need to be fixed, not bounds
                if z_coord.units == "cm":
                    z_coord.points = z_coord.core_points() / 100.0
                    z_coord.units = "m"

                # Fix depth metadata
                if z_coord.standard_name is None:
                    fix_ocean_depth_coord(cube)
        return cubes


class Pr(Fix):
    """Fixes for pr."""

    def fix_metadata(self, cubes):
        """Fix time coordinates.

        Parameters
        ----------
        cubes : iris.cube.CubeList
                Cubes to fix

        Returns
        -------
        iris.cube.CubeList
        """
        new_list = iris.cube.CubeList()
        for cube in cubes:
            try:
                old_time = cube.coord("time")
            except iris.exceptions.CoordinateNotFoundError:
                new_list.append(cube)
            else:
                if old_time.is_monotonic():
                    new_list.append(cube)
                else:
                    time_units = old_time.units
                    time_data = old_time.points

                    # erase erroneously copy-pasted points
                    time_diff = np.diff(time_data)
                    idx_neg = np.where(time_diff <= 0.0)[0]
                    while len(idx_neg) > 0:
                        time_data = np.delete(time_data, idx_neg[0] + 1)
                        time_diff = np.diff(time_data)
                        idx_neg = np.where(time_diff <= 0.0)[0]

                    # create the new time coord
                    new_time = iris.coords.DimCoord(
                        time_data,
                        standard_name="time",
                        var_name="time",
                        units=time_units,
                    )

                    # create a new cube with the right shape
                    dims = (
                        time_data.shape[0],
                        cube.coord("latitude").shape[0],
                        cube.coord("longitude").shape[0],
                    )
                    data = cube.data
                    new_data = np.ma.append(
                        data[: dims[0] - 1, :, :],
                        data[-1, :, :],
                    )
                    new_data = new_data.reshape(dims)

                    tmp_cube = iris.cube.Cube(
                        new_data,
                        standard_name=cube.standard_name,
                        long_name=cube.long_name,
                        var_name=cube.var_name,
                        units=cube.units,
                        attributes=cube.attributes,
                        cell_methods=cube.cell_methods,
                        dim_coords_and_dims=[
                            (new_time, 0),
                            (cube.coord("latitude"), 1),
                            (cube.coord("longitude"), 2),
                        ],
                    )

                    new_list.append(tmp_cube)
        return new_list


class Tasmin(Pr):
    """Fixes for tasmin."""

    def fix_metadata(self, cubes):
        """Fix time and height 2m coordinates.

        Fix for time coming from Pr.

        Parameters
        ----------
        cubes : iris.cube.CubeList
                Cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube, height=2.0)
        return cubes


class Tasmax(Pr):
    """Fixes for tasmax."""

    def fix_metadata(self, cubes):
        """Fix time and height 2m coordinates.

        Fix for time coming from Pr.

        Parameters
        ----------
        cubes : iris.cube.CubeList
                Cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        for cube in cubes:
            add_scalar_height_coord(cube, height=2.0)
        return cubes


class Msftmz(Fix):
    """Fixes for discrete DimCoord."""

    @staticmethod
    def transform_region_coord(
        coord: iris.coords.DimCoord,
    ) -> iris.coords.AuxCoord:
        """Transform a DimCoord to AuxCoord.

        indexes as points to names as points.

        Parameters
        ----------
        coord: iris.coords.DimCoord
               DimCoord to be transformed

        Returns
        -------
        iris.coords.AuxCoord

        """
        # parses string like: 'atlantic_arctic_ocean=0, indian_pacific_ocean=1, global_ocean=2'
        region_string = coord.attributes["requested"]
        lookup = {
            int(split_point[1]): split_point[0]
            for split_point in [
                point.strip().split("=") for point in region_string.split(",")
            ]
        }
        new_points = [""] * len(lookup)
        for old_label in coord.points:
            new_points[old_label] = lookup[old_label]
        return iris.coords.AuxCoord(
            new_points,
            standard_name="region",
            var_name="basin",
            long_name="ocean basin",
            units="no unit",
        )

    def fix_metadata(self, cubes: iris.cube.CubeList) -> iris.cube.CubeList:
        """Transform a DimCoord to AuxCoord.

        indexes as points to names as points.

        Parameters
        ----------
        cubes: iris.cube.CubeList
               List of cubes to fix

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        coord = cube.coord("region")
        new_coord = self.transform_region_coord(coord=coord)
        dims = cube.coord_dims("region")
        for cube in cubes:
            cube.remove_coord("region")
            cube.add_aux_coord(new_coord, dims)
        return cubes
