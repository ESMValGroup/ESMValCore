"""Horizontal and vertical regridding module."""

import os
import re
from copy import deepcopy

import numpy as np
import stratify
import iris
from iris.analysis import AreaWeighted, Linear, Nearest, UnstructuredNearest
from iris.util import broadcast_to_shape
import cdo
from collections import OrderedDict
import time

from ..cmor.fix import fix_file, fix_metadata
from ..cmor.table import CMOR_TABLES
from ._io import concatenate_callback, load
from ._regrid_esmpy import ESMF_REGRID_METHODS
from ._regrid_esmpy import regrid as esmpy_regrid
from ._shared import guess_bounds


import logging
logger = logging.getLogger(__name__)


# Regular expression to parse a "MxN" cell-specification.
_CELL_SPEC = re.compile(
    r'''\A
        \s*(?P<dlon>\d+(\.\d+)?)\s*
        x
        \s*(?P<dlat>\d+(\.\d+)?)\s*
        \Z
     ''', re.IGNORECASE | re.VERBOSE)

# Default fill-value.
_MDI = 1e+20

# Stock cube - global grid extents (degrees).
_LAT_MIN = -90.0
_LAT_MAX = 90.0
_LAT_RANGE = _LAT_MAX - _LAT_MIN
_LON_MIN = 0.0
_LON_MAX = 360.0
_LON_RANGE = _LON_MAX - _LON_MIN

# A cached stock of standard horizontal target grids.
_CACHE = dict()

# Supported point interpolation schemes.
POINT_INTERPOLATION_SCHEMES = {
    'linear': Linear(extrapolation_mode='mask'),
    'nearest': Nearest(extrapolation_mode='mask'),
}

# Supported horizontal regridding schemes.
HORIZONTAL_SCHEMES = {
    'linear': Linear(extrapolation_mode='mask'),
    'linear_extrapolate': Linear(extrapolation_mode='extrapolate'),
    'nearest': Nearest(extrapolation_mode='mask'),
    'nearest_extrapolate': Nearest(extrapolation_mode='extrapolate'),
    'area_weighted': AreaWeighted(),
    'unstructured_nearest': UnstructuredNearest(),
    'cdo_remapcon': 'remapcon',
}

# Supported vertical interpolation schemes.
VERTICAL_SCHEMES = ('linear', 'nearest',
                    'linear_horizontal_extrapolate_vertical',
                    'nearest_horizontal_extrapolate_vertical')


# # thats the way to go
# cube = iris.load_cube("/pf/b/b380860/tmp/regrid/mod_sea.nc")
# cube.coord('grid_latitude').guess_bounds()
# cube.coord('grid_longitude').guess_bounds()
# cube.remove_coord('latitude')
# cube.coord('grid_latitude').rename('latitude')
# cube.remove_coord('longitude')
# cube.coord('grid_longitude').rename('longitude')
# iris.save(cube, "/pf/b/b380860/tmp/regrid/mod_sea_2.nc", fill_value=1e20)


# iris.save(target_grid, "/pf/b/b380860/tmp/regrid/obs_sea_2.nc")

# cube = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/tas_EUR-44_CNRM-CERFACS-CNRM-CM5_historical-rcp85_r1i1p1_CLMcom-CCLM5-0-6_v1_mon*.nc_1960-2099/07_seasonal_statistics.nc")
# target_grid = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/OBS_BerkeleyEarth_reanaly_2020_Amon_tas_1960-2019/07_seasonal_statistics.nc")
# # scheme = AreaWeighted(mdtol=0.9)

# cube.coord('grid_latitude').guess_bounds()
# cube.coord('grid_longitude').guess_bounds()
# cube.remove_coord('latitude')
# cube.coord('grid_latitude').rename('latitude')
# cube.remove_coord('longitude')
# cube.coord('grid_longitude').rename('longitude')

# import cdo as CDO

# coord = target_grid.coord('time')
# axis = target_grid.coord_dims(coord)[0]
# mask = np.all(target_grid.data.mask, axis=axis)
# mask = np.invert(mask)
# mask = mask.astype(np.int)
# mask = np.where(mask==mask.max(), 1, mask)
# target_grid = target_grid.collapsed('time', iris.analysis.MEAN)
# target_grid.data = mask

# cubefile = "/pf/b/b380860/tmp/regrid/mod.nc"
# rgcubefile = "/pf/b/b380860/tmp/regrid/rgmod.nc"
# gridfile = "/pf/b/b380860/tmp/regrid/grid.nc"
# iris.save(cube, cubefile, fill_value=1e20)
# iris.save(target_grid, gridfile)

# weightfile = "/pf/b/b380860/tmp/regrid/weights.nc"

# cdo = CDO.Cdo()

# cdo.gencon(gridfile, input=cubefile, output=weightfile)
# cdo.remap(gridfile,weightfile, input=cubefile, output=rgcubefile)




# import cdo as CDO

# cube = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/tas_EUR-44_CNRM-CERFACS-CNRM-CM5_historical-rcp85_r1i1p1_CLMcom-CCLM5-0-6_v1_mon*.nc_1960-2099/07_seasonal_statistics.nc")
# # target_grid = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/OBS_BerkeleyEarth_reanaly_2020_Amon_tas_1960-2019/07_seasonal_statistics.nc")
# # scheme = AreaWeighted(mdtol=0.9)

# cube.coord('grid_latitude').guess_bounds()
# cube.coord('grid_longitude').guess_bounds()
# cube.remove_coord('latitude')
# cube.coord('grid_latitude').rename('latitude')
# cube.remove_coord('longitude')
# cube.coord('grid_longitude').rename('longitude')
# iris.save(cube, cubefile, fill_value=1e20)

# cubefile = "/pf/b/b380860/tmp/regrid/mod.nc"
# rgcubefile = "/pf/b/b380860/tmp/regrid/rgmod_2.nc"
# gridfile = "/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/OBS_BerkeleyEarth_reanaly_2020_Amon_tas_1960-2019/07_seasonal_statistics.nc"

# cdo = CDO.Cdo()

# cdo.remapcon(gridfile, input=cubefile, output=rgcubefile)
# # load cube again
# # either use cube mask or mask of the observations




# cdo -P 4 gencon,${maskdestnc} -seltimestep,1 ${datanc} weights.nc




# # IPCC_SCHEMES = {
# #     'atlas_cdo_remapcon':
# # }

# # maskdestnc
# # cdo -P 4 gencon,${maskdestnc} -seltimestep,1 ${datanc} weights.nc
# # # cdo div ${datanc} -setctomiss,0 maskland.nc land.nc
# # # cdo div ${datanc} -setctomiss,0 masksea.nc sea.nc
# # cdo remap,${maskdestnc},weights.nc land.nc landr.nc
# # cdo remap,${maskdestnc},weights.nc sea.nc sear.nc
# # cdo ifthenelse -setmisstoc,0 ${maskdestnc} landr.nc sear.nc merged.nc

# # # Fill the gaps with unconstrained remapping (doremap preferred option)
# # cdo setmisstoc,1 -setrtoc,-9999999,9999999,0 merged.nc gaps.nc
# # cdo remap,${maskdestnc},weights.nc ${datanc} unconstrained.nc
# # cdo ifthenelse gaps.nc unconstrained.nc merged.nc ${outfile}

# # # procedure
# # - get the obs landmaks (maskdestnc), i.e. all points that are masked for every timestep
# # - generate the weights
# # - remap the cube



# #####################
# # The atlas procedure
# #####################
# # Make input landmask binary
# #cd /oceano/gmeteo/WORK/PROYECTOS/2018_IPCC/data/CMIP6/mask_and_refGrid
# cdo -setrtoc,-1,0.999,0 -setrtoc,0.999,2,1 ${masknc} maskland.nc
# cdo mulc,-1 -setrtoc,0.001,2,0 -setrtoc,-1,0.001,-1 ${masknc} masksea.nc
# # Sharp change at 0.5 land fraction
# # cdo -setrtoc,-1,0.5,0 -setrtoc,0.5,2,1 ${masknc} maskland.nc
# # cdo mulc,0.5 -setmisstoc,1 -setrtoc,-0.5,0.5,2 maskland.nc masksea.nc
# # Global destination grid (from cdo)
# #cdo -f nc4 topo orogdest.nc
# #cdo setrtoc,-20000,20000,1 -setrtomiss,-20000,0 orogdest.nc ${maskdestnc}
# cdo -P 4 gencon,${maskdestnc} -seltimestep,1 ${datanc} weights.nc

# #if test "${LSMASK}" -eq 1; then
#   cdo div ${datanc} -setctomiss,0 maskland.nc land.nc
#   cdo div ${datanc} -setctomiss,0 masksea.nc sea.nc
#   cdo remap,${maskdestnc},weights.nc land.nc landr.nc
#   cdo remap,${maskdestnc},weights.nc sea.nc sear.nc
#   cdo ifthenelse -setmisstoc,0 ${maskdestnc} landr.nc sear.nc merged.nc
#   # Fill the gaps with unconstrained remapping (doremap preferred option)
#   cdo setmisstoc,1 -setrtoc,-9999999,9999999,0 merged.nc gaps.nc
#   cdo remap,${maskdestnc},weights.nc ${datanc} unconstrained.nc
#   cdo ifthenelse gaps.nc unconstrained.nc merged.nc ${outfile}
#   # Fill the gaps by nearest neigbours
# #  cdo setmisstonn landr.nc landrfilled.nc
# #  cdo setmisstonn sear.nc searfilled.nc
# #  cdo ifthenelse -setmisstoc,0 ${maskdestnc} landrfilled.nc searfilled.nc ${outfile}
# #else
# #  cdo remap,${maskdestnc},weights.nc ${datanc} ${outfile}
# #fi



def parse_cell_spec(spec):
    """
    Parse an MxN cell specification string.

    Parameters
    ----------
    spec: str

    Returns
    -------
    tuple
        tuple of (float, float) of parsed (lon, lat)

    Raises
    ------
    ValueError
        if the MxN cell specification is malformed.
    ValueError
        invalid longitude and latitude delta in cell specification.
    """
    cell_match = _CELL_SPEC.match(spec)
    logger.info(f"{spec}")
    if cell_match is None:
        emsg = 'Invalid MxN cell specification for grid, got {!r}.'
        raise ValueError(emsg.format(spec))

    cell_group = cell_match.groupdict()
    dlon = float(cell_group['dlon'])
    dlat = float(cell_group['dlat'])

    if (np.trunc(_LON_RANGE / dlon) * dlon) != _LON_RANGE:
        emsg = ('Invalid longitude delta in MxN cell specification '
                'for grid, got {!r}.')
        raise ValueError(emsg.format(dlon))

    if (np.trunc(_LAT_RANGE / dlat) * dlat) != _LAT_RANGE:
        emsg = ('Invalid latitude delta in MxN cell specification '
                'for grid, got {!r}.')
        raise ValueError(emsg.format(dlat))

    return dlon, dlat


def _stock_cube(spec, lat_offset=True, lon_offset=True):
    """
    Create a stock cube.

    Create a global cube with M degree-east by N degree-north regular grid
    cells.

    The longitude range is from 0 to 360 degrees. The latitude range is from
    -90 to 90 degrees. Each cell grid point is calculated as the mid-point of
    the associated MxN cell.

    Parameters
    ----------
    spec : str
        Specifies the 'MxN' degree cell-specification for the global grid.
    lat_offset : bool
        Offset the grid centers of the latitude coordinate w.r.t. the
        pole by half a grid step. This argument is ignored if `target_grid`
        is a cube or file.
    lon_offset : bool
        Offset the grid centers of the longitude coordinate w.r.t. Greenwich
        meridian by half a grid step.
        This argument is ignored if `target_grid` is a cube or file.

    Returns
    -------
        A :class:`~iris.cube.Cube`.

    """
    dlon, dlat = parse_cell_spec(spec)
    mid_dlon, mid_dlat = dlon / 2, dlat / 2

    # Construct the latitude coordinate, with bounds.
    if lat_offset:
        latdata = np.linspace(_LAT_MIN + mid_dlat, _LAT_MAX - mid_dlat,
                              int(_LAT_RANGE / dlat))
    else:
        latdata = np.linspace(_LAT_MIN, _LAT_MAX, int(_LAT_RANGE / dlat) + 1)

    # Construct the longitude coordinat, with bounds.
    if lon_offset:
        londata = np.linspace(_LON_MIN + mid_dlon, _LON_MAX - mid_dlon,
                              int(_LON_RANGE / dlon))
    else:
        londata = np.linspace(_LON_MIN, _LON_MAX - dlon,
                              int(_LON_RANGE / dlon))

    lats = iris.coords.DimCoord(
        latdata,
        standard_name='latitude',
        units='degrees_north',
        var_name='lat')
    lats.guess_bounds()

    lons = iris.coords.DimCoord(
        londata,
        standard_name='longitude',
        units='degrees_east',
        var_name='lon')
    lons.guess_bounds()

    # Construct the resultant stock cube, with dummy data.
    shape = (latdata.size, londata.size)
    dummy = np.empty(shape, dtype=np.dtype('int8'))
    coords_spec = [(lats, 0), (lons, 1)]
    cube = iris.cube.Cube(dummy, dim_coords_and_dims=coords_spec)

    return cube


def _attempt_irregular_regridding(cube, scheme):
    """Check if irregular regridding with ESMF should be used."""
    if scheme in ESMF_REGRID_METHODS:
        try:
            lat_dim = cube.coord('latitude').ndim
            lon_dim = cube.coord('longitude').ndim
            if lat_dim == lon_dim == 2:
                return True
        except iris.exceptions.CoordinateNotFoundError:
            pass
    return False


def extract_point(cube, latitude, longitude, scheme):
    """Extract a point, with interpolation

    Extracts a single latitude/longitude point from a cube, according
    to the interpolation scheme `scheme`.

    Multiple points can also be extracted, by supplying an array of
    latitude and/or longitude coordinates. The resulting point cube
    will match the respective latitude and longitude coordinate to
    those of the input coordinates. If the input coordinate is a
    scalar, the dimension will be missing in the output cube (that is,
    it will be a scalar).


    Parameters
    ----------
    cube : cube
        The source cube to extract a point from.

    latitude, longitude : float, or array of float
        The latitude and longitude of the point.

    scheme : str
        The interpolation scheme. 'linear' or 'nearest'. No default.


    Returns
    -------
    Returns a cube with the extracted point(s), and with adjusted
    latitude and longitude coordinates (see above).


    Examples
    --------
    With a cube that has the coordinates

    - latitude: [1, 2, 3, 4]
    - longitude: [1, 2, 3, 4]
    - data values: [[[1, 2, 3, 4], [5, 6, ...], [...], [...],
                      ... ]]]

    >>> point = extract_point(cube, 2.5, 2.5, 'linear')  # doctest: +SKIP
    >>> point.data  # doctest: +SKIP
    array([ 8.5, 24.5, 40.5, 56.5])

    Extraction of multiple points at once, with a nearest matching scheme.
    The values for 0.1 will result in masked values, since this lies outside
    the cube grid.

    >>> point = extract_point(cube, [1.4, 2.1], [0.1, 1.1],
    ...                       'nearest')  # doctest: +SKIP
    >>> point.data.shape  # doctest: +SKIP
    (4, 2, 2)
    >>> # x, y, z indices of masked values
    >>> np.where(~point.data.mask)     # doctest: +SKIP
    (array([0, 0, 1, 1, 2, 2, 3, 3]), array([0, 1, 0, 1, 0, 1, 0, 1]),
    array([1, 1, 1, 1, 1, 1, 1, 1]))
    >>> point.data[~point.data.mask].data  # doctest: +SKIP
    array([ 1,  5, 17, 21, 33, 37, 49, 53])

    """

    msg = f"Unknown interpolation scheme, got {scheme!r}."
    scheme = POINT_INTERPOLATION_SCHEMES.get(scheme.lower())
    if not scheme:
        raise ValueError(msg)

    point = [('latitude', latitude), ('longitude', longitude)]
    cube = cube.interpolate(point, scheme=scheme)
    return cube


def regrid(cube, target_grid, scheme, lat_offset=True, lon_offset=True):
    """
    Perform horizontal regridding.

    Parameters
    ----------
    cube : cube
        The source cube to be regridded.
    target_grid : cube or str
        The cube that specifies the target or reference grid for the regridding
        operation. Alternatively, a string cell specification may be provided,
        of the form 'MxN', which specifies the extent of the cell, longitude by
        latitude (degrees) for a global, regular target grid.
    scheme : str
        The regridding scheme to perform, choose from
        'linear',
        'linear_extrapolate',
        'nearest',
        'area_weighted',
        'unstructured_nearest'.
    lat_offset : bool
        Offset the grid centers of the latitude coordinate w.r.t. the
        pole by half a grid step. This argument is ignored if `target_grid`
        is a cube or file.
    lon_offset : bool
        Offset the grid centers of the longitude coordinate w.r.t. Greenwich
        meridian by half a grid step.
        This argument is ignored if `target_grid` is a cube or file.

    Returns
    -------
    cube

    See Also
    --------
    extract_levels : Perform vertical regridding.

    """
    # logger.info(f"{type(scheme)}")
    if type(scheme) == OrderedDict:
        tmp_dir = scheme['tmp_dir']
        scheme = scheme['method']

    if HORIZONTAL_SCHEMES.get(scheme.lower()) is None:
        emsg = 'Unknown regridding scheme, got {!r}.'
        raise ValueError(emsg.format(scheme))

    if isinstance(target_grid, str):
        if os.path.isfile(target_grid):
            target_grid = iris.load_cube(target_grid)
        else:
            # Generate a target grid from the provided cell-specification,
            # and cache the resulting stock cube for later use.
            target_grid = _CACHE.setdefault(
                target_grid,
                _stock_cube(target_grid, lat_offset, lon_offset),
            )
            # Align the target grid coordinate system to the source
            # coordinate system. This does not make a lot of sense in the light
            # of rotated grid systems
            # src_cs = cube.coord_system()
            global_coord_sys = iris.coord_systems.GeogCS(
                                            semi_major_axis=6378137.0,
                                            semi_minor_axis=6356752.31424)
            xcoord = target_grid.coord(axis='x', dim_coords=True)
            ycoord = target_grid.coord(axis='y', dim_coords=True)
            xcoord.coord_system = global_coord_sys
            ycoord.coord_system = global_coord_sys
            target_grid.data = target_grid.data.astype('float32')

    if target_grid.coord_system() == None:
        target_grid.coord('latitude').coord_system = iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
                                                                         semi_minor_axis=6356752.31424)
        target_grid.coord('longitude').coord_system = iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
                                                                         semi_minor_axis=6356752.31424)

    # if cube.coord_system() == None:
    #     cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
    #                                                                      semi_minor_axis=6356752.31424)
    #     cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(semi_major_axis=6378137.0,
    #                                                                      semi_minor_axis=6356752.31424)

    if not isinstance(target_grid, iris.cube.Cube):
        raise ValueError('Expecting a cube, got {}.'.format(target_grid))

    # Unstructured regridding requires x2 2d spatial coordinates,
    # so ensure to purge any 1d native spatial dimension coordinates
    # for the regridder.
    if scheme == 'unstructured_nearest':
        for axis in ['x', 'y']:
            coords = cube.coords(axis=axis, dim_coords=True)
            if coords:
                [coord] = coords
                cube.remove_coord(coord)

    # import IPython
    # from traitlets.config import get_config
    # c = get_config()
    # c.InteractiveShellEmbed.colors = "Linux"

    # IPython.embed(config=c)

    # Perform the horizontal regridding.
    if 'cdo' in scheme:
        cube = regrid_cdo(cube, target_grid, scheme, tmp_dir)
    elif _attempt_irregular_regridding(cube, scheme):
        cube = esmpy_regrid(cube, target_grid, scheme)
    else:
        cube = cube.regrid(target_grid, HORIZONTAL_SCHEMES[scheme])

    return cube


def regrid_cdo(cube, target_grid, scheme, tmp_dir):
    """
    Regrid cube using a CDO regrid scheme

    cube : cube
        The source cube to be regridded.
    target_grid : cube or str
        The cube that specifies the target or reference grid for the regridding
        operation.
    scheme : str
        The regridding scheme to perform, choose from
        'cdo_remapcon',
    tmp_dir : str
        path to directory where cube, target_grid are saved to run cdo

    Returns
    -------
    cube


    Sideline info: Redoing the IPCC Atlas regrid function:
    Note that the Atlas is doing it seperately for land sea data
    (Ch 10 only deals with data over land so incomming cube data is already
    masked)

    # #####################
    # # The atlas procedure
    # #####################
    # # Make input landmask binary
    # #cd /oceano/gmeteo/WORK/PROYECTOS/2018_IPCC/data/CMIP6/mask_and_refGrid
    # cdo -setrtoc,-1,0.999,0 -setrtoc,0.999,2,1 ${masknc} maskland.nc
    # cdo mulc,-1 -setrtoc,0.001,2,0 -setrtoc,-1,0.001,-1 ${masknc} masksea.nc
    # # Sharp change at 0.5 land fraction
    # # cdo -setrtoc,-1,0.5,0 -setrtoc,0.5,2,1 ${masknc} maskland.nc
    # # cdo mulc,0.5 -setmisstoc,1 -setrtoc,-0.5,0.5,2 maskland.nc masksea.nc
    # # Global destination grid (from cdo)
    # #cdo -f nc4 topo orogdest.nc
    # #cdo setrtoc,-20000,20000,1 -setrtomiss,-20000,0 orogdest.nc ${maskdestnc}
    # cdo -P 4 gencon,${maskdestnc} -seltimestep,1 ${datanc} weights.nc

    # #if test "${LSMASK}" -eq 1; then
    #   cdo div ${datanc} -setctomiss,0 maskland.nc land.nc
    #   cdo div ${datanc} -setctomiss,0 masksea.nc sea.nc
    #   cdo remap,${maskdestnc},weights.nc land.nc landr.nc
    #   cdo remap,${maskdestnc},weights.nc sea.nc sear.nc
    #   cdo ifthenelse -setmisstoc,0 ${maskdestnc} landr.nc sear.nc merged.nc
    #   # Fill the gaps with unconstrained remapping (doremap preferred option)
    #   cdo setmisstoc,1 -setrtoc,-9999999,9999999,0 merged.nc gaps.nc
    #   cdo remap,${maskdestnc},weights.nc ${datanc} unconstrained.nc
    #   cdo ifthenelse gaps.nc unconstrained.nc merged.nc ${outfile}
    #   # Fill the gaps by nearest neigbours
    # #  cdo setmisstonn landr.nc landrfilled.nc
    # #  cdo setmisstonn sear.nc searfilled.nc
    # #  cdo ifthenelse -setmisstoc,0 ${maskdestnc} landrfilled.nc searfilled.nc ${outfile}
    # #else
    # #  cdo remap,${maskdestnc},weights.nc ${datanc} ${outfile}
    # #fi
    """
    cdo_scheme = scheme.split('cdo_')[1]

    unique_id = str(time.time()) + str(np.random.rand())
    unique_id = unique_id.replace('.', '-')

    cube_fname = os.path.join(tmp_dir, f'cube{unique_id}.nc')
    grid_fname = os.path.join(tmp_dir, f'grid{unique_id}.nc')
    rcube_fname = os.path.join(tmp_dir, f'rcube{unique_id}.nc')

    import IPython
    from traitlets.config import get_config
    c = get_config()
    c.InteractiveShellEmbed.colors = "Linux"

#    IPython.embed(config=c)
    # get the x, y coordinates named longitude, latitude
    coord_names = [coord.standard_name for coord in cube.coords()]
    x_name = cube.coord(axis='x', dim_coords=True).standard_name
    if x_name != 'longitude':
        cube = guess_bounds(cube, [x_name])
        if 'longitude' in coord_names:
            cube.remove_coord('longitude')
        cube.coord(x_name).rename('longitude')

    y_name = cube.coord(axis='y', dim_coords=True).standard_name
    if y_name != 'latitude':
        cube = guess_bounds(cube, [y_name])
        if 'latitude' in coord_names:
            cube.remove_coord('latitude')
        cube.coord(y_name).rename('latitude')

    # check if the target has a time axis and remove it
    coord_names = [coord.standard_name for coord in target_grid.coords()]
    if 'time' in coord_names:
        target_grid = target_grid[0]

    # for some reason cdo crashes when there are bounds
    if isinstance(cube.coord('latitude').coord_system,
                  iris.coord_systems.LambertConformal):
        cube.coord('latitude').bounds = None
        cube.coord('longitude').bounds = None

    # # check if target grid is rotatec/lcc and maybe save lat lon
    # tar_coord_names = [coord.standard_name for coord in target_grid.coords()]
    # tar_x_name = target_grid.coord(axis='x', dim_coords=True).standard_name
    # tar_y_name = target_grid.coord(axis='y', dim_coords=True).standard_name
    # if x_name != 'latitude' and y_name != 'latitude':
    #     target_aux = [target_grid.coord('latitude'),
    #                   target_grid.coord('longitude')]
    # IPython.embed(config=c)



    iris.save(cube, cube_fname, fill_value=1e20)
            #   netcdf_format="NETCDF3_CLASSIC")
    iris.save(target_grid, grid_fname, fill_value=1e20)

    if cdo_scheme == 'remapcon':
        cdo.Cdo().remapcon(grid_fname, input=cube_fname, output=rcube_fname)
    else:
        emsg = f'Unknown regridding scheme, got {cdo_scheme}.'
        raise ValueError(emsg.format(scheme))

    rcube = iris.load_cube(rcube_fname)

    # # clean up, this won't work ...
    # for fname in [cube_fname, grid_fname, rcube_fname]:
    #     os.remove(fname)

    return rcube

    # target_grid = iris.load_cube(target_grid)

    # iris.save(cube, "/pf/b/b380860/tmp/regrid/mod_sea_2.nc", fill_value=1e20)


    # iris.save(target_grid, "/pf/b/b380860/tmp/regrid/obs_sea_2.nc")

    # cube = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/tas_EUR-44_CNRM-CERFACS-CNRM-CM5_historical-rcp85_r1i1p1_CLMcom-CCLM5-0-6_v1_mon*.nc_1960-2099/07_seasonal_statistics.nc")
    # target_grid = iris.load_cube("/work/bk1088/b380860/results/recipe_CORDEX_EUR44_DKRZ_20200603_103519/preproc/fig_10_31/tas_cmip5_maps_trend/OBS_BerkeleyEarth_reanaly_2020_Amon_tas_1960-2019/07_seasonal_statistics.nc")
    # # scheme = AreaWeighted(mdtol=0.9)

    # cube.coord('grid_latitude').guess_bounds()
    # cube.coord('grid_longitude').guess_bounds()
    # cube.remove_coord('latitude')
    # cube.coord('grid_latitude').rename('latitude')
    # cube.remove_coord('longitude')
    # cube.coord('grid_longitude').rename('longitude')

    # import cdo as CDO

    # coord = target_grid.coord('time')
    # axis = target_grid.coord_dims(coord)[0]
    # mask = np.all(target_grid.data.mask, axis=axis)
    # mask = np.invert(mask)
    # mask = mask.astype(np.int)
    # mask = np.where(mask==mask.max(), 1, mask)
    # target_grid = target_grid.collapsed('time', iris.analysis.MEAN)
    # target_grid.data = mask

    # cubefile = "/pf/b/b380860/tmp/regrid/mod.nc"
    # rgcubefile = "/pf/b/b380860/tmp/regrid/rgmod.nc"
    # gridfile = "/pf/b/b380860/tmp/regrid/grid.nc"
    # iris.save(cube, cubefile, fill_value=1e20)
    # iris.save(target_grid, gridfile)

    # weightfile = "/pf/b/b380860/tmp/regrid/weights.nc"

    # cdo = CDO.Cdo()

    # cdo.gencon(gridfile, input=cubefile, output=weightfile)
    # cdo.remap(gridfile,weightfile, input=cubefile, output=rgcubefile)


def _create_cube(src_cube, data, src_levels, levels, ):
    """
    Generate a new cube with the interpolated data.

    The resultant cube is seeded with `src_cube` metadata and coordinates,
    excluding any source coordinates that span the associated vertical
    dimension. The `levels` of interpolation are used along with the
    associated source cube vertical coordinate metadata to add a new
    vertical coordinate to the resultant cube.

    Parameters
    ----------
    src_cube : cube
        The source cube that was vertically interpolated.
    data : array
        The payload resulting from interpolating the source cube
        over the specified levels.
    levels : array
        The vertical levels of interpolation.

    Returns
    -------
    cube

    .. note::

        If there is only one level of interpolation, the resultant cube
        will be collapsed over the associated vertical dimension, and a
        scalar vertical coordinate will be added.

    """
    # Get the source cube vertical coordinate and associated dimension.
    z_coord = src_cube.coord(axis='z', dim_coords=True)
    z_dim, = src_cube.coord_dims(z_coord)

    if data.shape[z_dim] != levels.size:
        emsg = ('Mismatch between data and levels for data dimension {!r}, '
                'got data shape {!r} with levels shape {!r}.')
        raise ValueError(emsg.format(z_dim, data.shape, levels.shape))

    # Construct the resultant cube with the interpolated data
    # and the source cube metadata.
    kwargs = deepcopy(src_cube.metadata)._asdict()
    result = iris.cube.Cube(data, **kwargs)

    # Add the appropriate coordinates to the cube, excluding
    # any coordinates that span the z-dimension of interpolation.
    for coord in src_cube.dim_coords:
        [dim] = src_cube.coord_dims(coord)
        if dim != z_dim:
            result.add_dim_coord(coord.copy(), dim)

    for coord in src_cube.aux_coords:
        dims = src_cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)

    for coord in src_cube.derived_coords:
        dims = src_cube.coord_dims(coord)
        if z_dim not in dims:
            result.add_aux_coord(coord.copy(), dims)

    # Construct the new vertical coordinate for the interpolated
    # z-dimension, using the associated source coordinate metadata.
    kwargs = deepcopy(src_levels._as_defn())._asdict()

    try:
        coord = iris.coords.DimCoord(levels, **kwargs)
        result.add_dim_coord(coord, z_dim)
    except ValueError:
        coord = iris.coords.AuxCoord(levels, **kwargs)
        result.add_aux_coord(coord, z_dim)

    # Collapse the z-dimension for the scalar case.
    if levels.size == 1:
        slicer = [slice(None)] * result.ndim
        slicer[z_dim] = 0
        result = result[tuple(slicer)]

    return result


def _vertical_interpolate(cube, src_levels, levels, interpolation,
                          extrapolation):
    """Perform vertical interpolation."""
    # Determine the source levels and axis for vertical interpolation.
    z_axis, = cube.coord_dims(cube.coord(axis='z', dim_coords=True))

    # Broadcast the 1d source cube vertical coordinate to fully
    # describe the spatial extent that will be interpolated.
    src_levels_broadcast = broadcast_to_shape(
        src_levels.points, cube.shape, cube.coord_dims(src_levels))

    # force mask onto data as nan's
    if np.ma.is_masked(cube.data):
        cube.data[cube.data.mask] = np.nan

    # Now perform the actual vertical interpolation.
    new_data = stratify.interpolate(
        levels,
        src_levels_broadcast,
        cube.data,
        axis=z_axis,
        interpolation=interpolation,
        extrapolation=extrapolation)

    # Calculate the mask based on the any NaN values in the interpolated data.
    mask = np.isnan(new_data)

    if np.any(mask):
        # Ensure that the data is masked appropriately.
        new_data = np.ma.array(new_data, mask=mask, fill_value=_MDI)

    # Construct the resulting cube with the interpolated data.
    return _create_cube(cube, new_data, src_levels, levels.astype(float))


def extract_levels(cube, levels, scheme, coordinate=None):
    """
    Perform vertical interpolation.

    Parameters
    ----------
    cube : cube
        The source cube to be vertically interpolated.
    levels : array
        One or more target levels for the vertical interpolation. Assumed
        to be in the same S.I. units of the source cube vertical dimension
        coordinate.
    scheme : str
        The vertical interpolation scheme to use. Choose from
        'linear',
        'nearest',
        'nearest_horizontal_extrapolate_vertical',
        'linear_horizontal_extrapolate_vertical'.
    coordinate :  optional str
        The coordinate to interpolate

    Returns
    -------
    cube

    See Also
    --------
    regrid : Perform horizontal regridding.

    """
    if scheme not in VERTICAL_SCHEMES:
        emsg = 'Unknown vertical interpolation scheme, got {!r}. '
        emsg += 'Possible schemes: {!r}'
        raise ValueError(emsg.format(scheme, VERTICAL_SCHEMES))

    # This allows us to put level 0. to load the ocean surface.
    extrap_scheme = 'nan'
    if scheme == 'nearest_horizontal_extrapolate_vertical':
        scheme = 'nearest'
        extrap_scheme = 'nearest'

    if scheme == 'linear_horizontal_extrapolate_vertical':
        scheme = 'linear'
        extrap_scheme = 'nearest'

    # Ensure we have a non-scalar array of levels.
    levels = np.array(levels, ndmin=1)

    # Get the source cube vertical coordinate, if available.
    if coordinate:
        src_levels = cube.coord(coordinate)
    else:
        src_levels = cube.coord(axis='z', dim_coords=True)

    if (src_levels.shape == levels.shape
            and np.allclose(src_levels.points, levels)):
        # Only perform vertical extraction/interploation if the source
        # and target levels are not "similar" enough.
        result = cube
    elif len(src_levels.shape) == 1 and \
            set(levels).issubset(set(src_levels.points)):
        # If all target levels exist in the source cube, simply extract them.
        name = src_levels.name()
        coord_values = {name: lambda cell: cell.point in set(levels)}
        constraint = iris.Constraint(coord_values=coord_values)
        result = cube.extract(constraint)
        # Ensure the constraint did not fail.
        if not result:
            emsg = 'Failed to extract levels {!r} from cube {!r}.'
            raise ValueError(emsg.format(list(levels), name))
    else:
        # As a last resort, perform vertical interpolation.
        result = _vertical_interpolate(
            cube, src_levels, levels, scheme, extrap_scheme)

    return result


def get_cmor_levels(cmor_table, coordinate):
    """Get level definition from a CMOR coordinate.

    Parameters
    ----------
    cmor_table: str
        CMOR table name
    coordinate: str
        CMOR coordinate name

    Returns
    -------
    list[int]

    Raises
    ------
    ValueError:
        If the CMOR table is not defined, the coordinate does not specify any
        levels or the string is badly formatted.

    """
    if cmor_table not in CMOR_TABLES:
        raise ValueError(
            "Level definition cmor_table '{}' not available".format(
                cmor_table))

    if coordinate not in CMOR_TABLES[cmor_table].coords:
        raise ValueError('Coordinate {} not available for {}'.format(
            coordinate, cmor_table))

    cmor = CMOR_TABLES[cmor_table].coords[coordinate]

    if cmor.requested:
        return [float(level) for level in cmor.requested]
    if cmor.value:
        return [float(cmor.value)]

    raise ValueError(
        'Coordinate {} in {} does not have requested values'.format(
            coordinate, cmor_table))


def get_reference_levels(filename,
                         project,
                         dataset,
                         short_name,
                         mip,
                         frequency,
                         fix_dir):
    """Get level definition from a reference dataset.

    Parameters
    ----------
    filename: str
        Path to the reference file

    Returns
    -------
    list[float]

    Raises
    ------
    ValueError:
        If the dataset is not defined, the coordinate does not specify any
        levels or the string is badly formatted.

    """
    filename = fix_file(
        file=filename,
        short_name=short_name,
        project=project,
        dataset=dataset,
        mip=mip,
        output_dir=fix_dir,
    )
    cubes = load(filename, callback=concatenate_callback)
    cubes = fix_metadata(
        cubes=cubes,
        short_name=short_name,
        project=project,
        dataset=dataset,
        mip=mip,
        frequency=frequency,
    )
    cube = cubes[0]
    try:
        coord = cube.coord(axis='Z')
    except iris.exceptions.CoordinateNotFoundError:
        raise ValueError('z-coord not available in {}'.format(filename))
    return coord.points.tolist()
