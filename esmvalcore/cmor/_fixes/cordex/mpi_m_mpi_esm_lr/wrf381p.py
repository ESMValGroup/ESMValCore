"""Fixes for rcm WRF381P driven by MPI-M-MPI-ESM-LR."""

from esmvalcore.cmor.fix import Fix
from esmvalcore.preprocessor._regrid import _cordex_stock_cube


class AllVars(Fix):
    """Fixes for all vars."""

    def fix_metadata(self, cubes):
        cube = self.get_cube_from_list(cubes).copy()
        if cube.coords("projection_x_coordinate"):
            # WRF381P datasets with the MPI-M-MPI-ESM-LR driver have bad
            # projection coordinates containing all zeros, but the latitude
            # and longitude points match those of the standard grid. Therefore
            # we replace the bad coordinates with the standard rotated pole
            # coordinates.
            # https://github.com/ESMValGroup/ESMValCore/issues/3145
            standard_grid = _cordex_stock_cube(self.extra_facets["domain"])
            x_dim = cube.coord_dims("projection_x_coordinate")
            y_dim = cube.coord_dims("projection_y_coordinate")

            # Remove the bad coordinates.
            cube.remove_coord("projection_x_coordinate")
            cube.remove_coord("projection_y_coordinate")
            cube.remove_coord("longitude")
            cube.remove_coord("latitude")

            # Add the standard rotated pole coordinates.
            cube.add_dim_coord(
                standard_grid.coord("grid_longitude"),
                x_dim,
            )
            cube.add_dim_coord(
                standard_grid.coord("grid_latitude"),
                y_dim,
            )
            cube.add_aux_coord(standard_grid.coord("longitude"), y_dim + x_dim)
            cube.add_aux_coord(standard_grid.coord("latitude"), y_dim + x_dim)

        return [cube]
