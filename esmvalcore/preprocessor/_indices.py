"""Module to compute various fixed indices."""
import logging
import os
from datetime import datetime

import iris
import numpy as np
import yaml

from ._area import extract_region
from ._regrid import extract_levels
from ._time import annual_statistics, extract_season

seasons = ['DJF', 'MAM', 'JJA', 'SON']
MOC_VARIABLE = 'moc_mar_hc10'
AIRPRESS_VARIABLE = 'UM_0_fc8_vn405'

# set up logging
logger = logging.getLogger(__name__)

def _extract_u850(cube):
    """
    Get jet speed and jet latitude.

    Extract eastern wind u at 850 hPa 15-75N lat.
    Extract mean 0-60W lon. Mean on LON. Extract season.
    Return each season's cube in a dictionary.
    """
    # extract 300-360W lon; 15-75N lat region; CMOR coords standards
    cube = extract_region(cube, 0., 60., 15., 75.)

    # extract 850 hPa
    cube = extract_levels(cube, 85000., 'linear')

    # collapse-mean on lon
    cube = cube.collapsed(['longitude'], iris.analysis.MEAN)

    # extract seasons
    seasonal_dict = {
        season: extract_season(cube, season) for season in seasons
    }

    return seasonal_dict


def _get_jets(seasonal_dict):
    """Take seasonal dictionary and get jets dicts (speeds and lats)."""
    # remove non-seasons and sub-dimensional data
    seasonal_dict = {k: v for k, v in seasonal_dict.items() if v is not None}
    seasonal_dict = {
        k: v for k, v in seasonal_dict.items() if len(v.data.shape) >= 2
    }

    # get jet speeds
    jet_speeds = {
        season: np.amax(seasonal_dict[season].data,
                        axis=1) for season in seasonal_dict.keys()
    }

    # get the jet latitudes
    jet_lats = {}
    for season, _ in seasonal_dict.items():
        cube = seasonal_dict[season]
        jet_lat = np.empty((cube.data.shape[0], ))
        max_ind = np.argmax(cube.data, axis=1)
        jet_lat[max_ind[0]] = cube.coord('latitude').points[max_ind[1]]
        jet_lat_cube = iris.cube.Cube(
            jet_lat,
            dim_coords_and_dims=[],
            long_name="jet-latitudes")
        jet_speed_cube = iris.cube.Cube(
            jet_speeds[season],
            dim_coords_and_dims=[],
            long_name="jet-speeds")
        jet_lats[season] = jet_lat_cube
        jet_speeds[season] = jet_speed_cube

    return jet_speeds, jet_lats


def _djf_greenland_iceland(data_file, var_constraint, season):
    """Get the DJF mean for Greenland-Iceland."""
    cube = iris.load(data_file, constraints=var_constraint)[0]
    greenland_map = extract_region(cube, 25., 35., 30., 40.)
    iceland_map = extract_region(cube, 15., 25., 60., 70.)
    greenland = greenland_map.collapsed(['longitude', 'latitude'],
                                        iris.analysis.MEAN)
    iceland = iceland_map.collapsed(['longitude', 'latitude'],
                                    iris.analysis.MEAN)

    # get cubes of interest
    greenland_djf = extract_season(greenland, season)
    iceland_djf = extract_season(iceland, season)
    diff = greenland - iceland
    season_geo_diff = extract_season(diff, season)

    return season_geo_diff


def _moc_vn(moc_file, vn_file):
    """Compute yearly means for moc and vn."""
    # moc
    moc_constraint = iris.Constraint(
        cube_func=(lambda c: c.var_name == MOC_VARIABLE))
    moc_cube = iris.load(moc_file, constraints=moc_constraint)[0]
    annual_moc = annual_statistics(moc_cube)

    # vn405
    vn_constraint = iris.Constraint(
        cube_func=(lambda c: c.var_name == AIRPRESS_VARIABLE))
    vn_cube = iris.load(vn_file, constraints=vn_constraint)[0]
    annual_vn = annual_statistics(vn_cube)

    return annual_moc, annual_vn


def acsis_indices(cube, moc_file, vn_file):
    """
    Function to compute and write to disk ACSIS indices.
    """
    logger.info("Computing and saving ACSIS indices...")
    if os.path.isfile(moc_file):
        logger.info("Using file for {}: {}".format(MOC_VARIABLE, moc_file))
    else:
        raise OSError("File {} for {} does not exist.".format(moc_file,
                                                              MOC_VARIABLE))
    if os.path.isfile(vn_file):
        logger.info("Using file for {}: {}".format(AIRPRESS_VARIABLE,
                                                   vn_file))
    else:
        raise OSError("File {} for {} does not exist.".format(
                          vn_file,
                          AIRPRESS_VARIABLE))

    # jets compute
    season_dict = _extract_u850(cube)
    jet_speeds, jet_lats = _get_jets(season_dict)

    # jets save
    for season in seasons:
        iris.save(jet_speeds[season], 'u850_{}_jet-speeds.nc'.format(season))
        iris.save(jet_lats[season], 'u850_{}_jet-latitudes.nc'.format(season))

    # moc-vn

