"""Fix base classes for ACCESS-ESM on-the-fly CMORizer."""

import iris
import logging
import warnings
from pathlib import Path

import iris
import numpy as np
from cf_units import Unit
from iris.cube import CubeList
from esmvalcore.cmor._fixes.native_datasets import NativeDatasetFix
from pathlib import Path

logger = logging.getLogger(__name__)


class AccessFix(NativeDatasetFix):
    """Fixes functions."""

    def fix_coord_system(self, cube):
        """Delete coord_system to make CubeList able to merge."""
        for dim in cube.dim_coords:
            if dim.coord_system is not None:
                cube.coord(dim.standard_name).coord_system = None

    def get_cubes_from_multivar(self, cubes):
        """Get cube before calculate from multiple variables."""
        name_list = self.extra_facets.get("raw_name", self.vardef.short_name)
        name_list = self.extra_facets.get("raw_name", self.vardef.short_name)

        data_list = []
        for name in name_list:
            data_list.append(self.get_cube(cubes, name))
        return CubeList(data_list)
    
    def fix_ocean_dim_coords(self, cube):
        """Fix dim coords of ocean variables"""
        cube.dim_coords[-2].points = np.array([int(i) for i in range(300)])
        cube.dim_coords[-2].standard_name = None
        cube.dim_coords[-2].var_name = 'j'
        cube.dim_coords[-2].long_name = 'cell index along second dimension'
        cube.dim_coords[-2].attributes = None

        cube.dim_coords[-1].points = np.array([int(i) for i in range(360)])
        cube.dim_coords[-1].standard_name = None
        cube.dim_coords[-1].var_name = 'i'
        cube.dim_coords[-1].long_name = 'cell index along first dimension'
        cube.dim_coords[-1].attributes = None
        cube.dim_coords[-1].units = Unit(1)

    def fix_ocean_aux_coords(self, cube, gridpath):
        """Fix aux coords of ocean variables."""
        lat_bounds, lon_bounds = self.load_ocean_grid_data(
            "ocean_grid_path", gridpath
        )
        temp_points = []
        for i in cube.aux_coords[-1].points:
            temp_points.append([j + 360 for j in i if j < 0] 
                                +[j for j in i if j >= 0])
        cube.aux_coords[-1].points = np.array(temp_points)
        cube.aux_coords[-1].standard_name = 'longitude'
        cube.aux_coords[-1].long_name = 'longitude'
        cube.aux_coords[-1].var_name = 'longitude'
        cube.aux_coords[-1].attributes = None
        cube.aux_coords[-1].units = "degrees"
        cube.aux_coords[-1].bounds = lon_bounds

        temp_points=[]
        for i in cube.aux_coords[-2].points:
            temp_points.append([j.astype(np.float64) for j in i])
        cube.aux_coords[-2].points = np.array(temp_points)
        cube.aux_coords[-2].standard_name = 'latitude'
        cube.aux_coords[-2].long_name = 'latitude'
        cube.aux_coords[-2].var_name = 'latitude'
        cube.aux_coords[-2].attributes = None
        cube.aux_coords[-2].units = "degrees"
        cube.aux_coords[-2].bounds = lat_bounds

    def _get_path_from_facet(self, facet):
        """Try to get path from facet."""
        path = Path(self.extra_facets[facet])
        if not path.is_file():
            raise FileNotFoundError(
                f"'{path}' given by facet '{facet}' does not exist"
            )
        return path

    def load_ocean_grid_data(self, facet, gridpath):
        """Load supplementary grid data for ACCESS ocean variable."""
        if gridpath is not None:
            path_to_grid_data = Path(gridpath)
        else:
            path_to_grid_data = self._get_path_from_facet(facet)
        cubes = self._load_cubes(path_to_grid_data)

        y_vert_t = [cube for cube in cubes if cube.var_name == "y_vert_T"][0]
        lat_bounds = np.transpose(y_vert_t.data, (1, 2, 0))
        x_vert_t = [cube for cube in cubes if cube.var_name == "x_vert_T"][0]
        lon_bounds = np.transpose(x_vert_t.data, (1, 2, 0))

        return lat_bounds, lon_bounds

    @staticmethod
    def _load_cubes(path: Path | str) -> CubeList:
        """Load cubes and ignore certain warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Ignoring netCDF variable .* invalid units .*",
                category=UserWarning,
                module="iris",
            )  # iris < 3.8
            warnings.filterwarnings(
                "ignore",
                message="Ignoring invalid units .* on netCDF variable .*",
                category=UserWarning,
                module="iris",
            )  # iris >= 3.8
            warnings.filterwarnings(
                "ignore",
                message="Gracefully filling .* dimension coordinate.*",
                category=UserWarning,
                module="iris",
            )
            warnings.filterwarnings(
                "ignore",
                message="Failed to create .*dimension coordinate.*",
                category=UserWarning,
                module="iris",
            )
            cubes = iris.load(path)
        return cubes
