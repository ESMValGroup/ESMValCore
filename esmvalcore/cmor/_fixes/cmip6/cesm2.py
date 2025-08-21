"""Fixes for CESM2 model."""

from pathlib import Path
from shutil import copyfile

import iris
import iris.coords
import numpy as np
from netCDF4 import Dataset

from esmvalcore.cmor._fixes.common import SiconcFixScalarCoord
from esmvalcore.cmor._fixes.fix import Fix
from esmvalcore.cmor._fixes.shared import (
    add_scalar_depth_coord,
    add_scalar_height_coord,
    add_scalar_typeland_coord,
    add_scalar_typesea_coord,
    fix_ocean_depth_coord,
)


class Cl(Fix):
    """Fixes for ``cl``."""

    def _fix_formula_terms(
        self,
        file: str | Path,
        output_dir: str | Path,
        add_unique_suffix: bool = False,
    ) -> Path:
        """Fix ``formula_terms`` attribute."""
        new_path = self.get_fixed_filepath(
            output_dir,
            file,
            add_unique_suffix=add_unique_suffix,
        )
        copyfile(file, new_path)
        with Dataset(new_path, mode="a") as dataset:
            dataset.variables["lev"].formula_terms = "p0: p0 a: a b: b ps: ps"
            dataset.variables[
                "lev"
            ].standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"
        return new_path

    def fix_file(
        self,
        file: str | Path,
        output_dir: str | Path,
        add_unique_suffix: bool = False,
    ) -> Path:
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
        new_path = self._fix_formula_terms(
            file,
            output_dir,
            add_unique_suffix=add_unique_suffix,
        )
        with Dataset(new_path, mode="a") as dataset:
            dataset.variables["a_bnds"][:] = dataset.variables["a_bnds"][
                ::-1,
                :,
            ]
            dataset.variables["b_bnds"][:] = dataset.variables["b_bnds"][
                ::-1,
                :,
            ]
        return new_path

    def fix_metadata(self, cubes):
        """Fix ``atmosphere_hybrid_sigma_pressure_coordinate``.

        See discussion in #882 for more details on that.

        Parameters
        ----------
        cubes : iris.cube.CubeList
            Input cubes.

        Returns
        -------
        iris.cube.CubeList

        """
        cube = self.get_cube_from_list(cubes)
        lev_coord = cube.coord(var_name="lev")
        a_coord = cube.coord(var_name="a")
        b_coord = cube.coord(var_name="b")
        lev_coord.points = a_coord.core_points() + b_coord.core_points()
        lev_coord.bounds = a_coord.core_bounds() + b_coord.core_bounds()
        lev_coord.units = "1"
        return cubes


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
