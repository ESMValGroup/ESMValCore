"""Module to compute various fixed indices."""
import logging

import iris
import numpy as np

from ._area import extract_region
from ._regrid import extract_levels
from ._time import annual_statistics, extract_season

# global vars: seasons
CLIM_SEASONS = ['DJF', 'MAM', 'JJA', 'SON']

# set up logging
logger = logging.getLogger(__name__)


def _load_cube(data_file, variable):
    """Load cube with constraint."""
    var_constraint = iris.Constraint(
        cube_func=(lambda c: c.var_name == variable))
    cube = iris.load(data_file, constraints=var_constraint)
    if not cube:
        raise ValueError("No variable {} found in file {}".format(
            variable, data_file))
    else:
        cube = cube[0]

    return cube


def _add_attribute(cube, data, long_name):
    """Add the indices as aux coords to cube."""
    cube.attributes[long_name] = data

    return cube


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
        season: extract_season(cube, season)
        for season in CLIM_SEASONS
    }

    return seasonal_dict


def _get_jets(season_dict):
    """Take seasonal dictionary and get jets dicts (speeds and lats)."""
    # remove non-seasons and sub-dimensional data
    season_dict = {k: v for k, v in season_dict.items() if v is not None}
    season_dict = {
        k: v
        for k, v in season_dict.items() if len(v.data.shape) >= 2
    }

    # get jet speeds
    jet_speeds = {
        season: np.amax(season_dict[season].data, axis=1)
        for season in season_dict.keys()
    }

    # get the jet latitudes
    jet_lats = {
        s:
        season_dict[s].coord('latitude').points[np.argmax(season_dict[s].data,
                                                          axis=1)]
        for s, _ in season_dict.items()
    }

    return jet_speeds, jet_lats


def _djf_greenland_iceland(vn_file, vn_variable, season):
    """Get the DJF mean for Greenland-Iceland."""
    cube = _load_cube(vn_file, vn_variable)
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

    return greenland_djf, iceland_djf, season_geo_diff


def _moc_vn(moc_file, vn_file, moc_variable, vn_variable):
    """Compute yearly means for moc and vn."""
    # moc
    moc_cube = _load_cube(moc_file, moc_variable)
    annual_moc = annual_statistics(moc_cube)

    # vn405
    vn_cube = _load_cube(vn_file, vn_variable)
    annual_vn = annual_statistics(vn_cube)

    # greenland-iceland
    greenland_djf, iceland_djf, season_geo_diff = \
        _djf_greenland_iceland(vn_file, vn_variable, "DJF")

    return (annual_moc, annual_vn, greenland_djf, iceland_djf, season_geo_diff)


def acsis_indices(cube, moc_file, vn_file, moc_variable, vn_variable):
    """
    Compute and store ACSIS indices.

    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    moc_file: str
        file containing moc data
    vn_file: str
        file containing vn data
    moc_variable: str
        identify the moc variable
    vn_variable: str
        identify the vn variable

    Returns
    -------
    iris.cube.Cube
        original cube with ACSIS indicators as attributes.
    """
    logger.info("Computing and saving ACSIS indices...")

    # jets compute
    logger.debug("Extracting 850hPa and seasons from {}".format(cube))
    season_dict = _extract_u850(cube)
    logger.debug("Computing jets from {}".format(str(season_dict)))
    jet_speeds, jet_lats = _get_jets(season_dict)

    # jets aux coord
    if not cube.coords('clim_season'):
        iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    for season in CLIM_SEASONS:
        if season in jet_speeds:
            jets_name = 'u850_{}_jet-speeds'.format(season)
            _add_attribute(cube, jet_speeds[season], jets_name)
            lats_name = 'u850_{}_jet-latitudes'.format(season)
            _add_attribute(cube, jet_lats[season], lats_name)

    # moc-vn-greenland-iceland and add aux coord
    (annual_moc, annual_vn, greenland_djf, iceland_djf,
     season_geo_diff) = _moc_vn(moc_file, vn_file, moc_variable, vn_variable)
    _add_attribute(cube, np.max(annual_moc), 'max_annual_moc')
    _add_attribute(cube, np.max(annual_vn), 'max_annual_vn')
    _add_attribute(cube, np.max(greenland_djf), 'max_greenland_djf')
    _add_attribute(cube, np.max(iceland_djf), 'max_iceland_djf')
    _add_attribute(cube, np.max(season_geo_diff), 'max_iceland_greenland')

    return cube
