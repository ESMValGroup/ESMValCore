"""Operations related to cycles (annual cycle, diurnal cycle, etc.)."""
import logging

import iris
import iris.coord_categorisation

logger = logging.getLogger(__name__)


def amplitude(cube, coords):
    """Calculate amplitude of cycles by aggregating over coordinates.

    Note
    ----
    The amplitude is calculated as `peak-to-peak` amplitude (difference
    between maximum and minimum value of the signal). Other amplitude types
    are currently not supported.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input data.
    coords : str or list of str
        Coordinates over which is aggregated. For example, use ``'year'`` to
        extract the annual cycle amplitude for each year in the data or
        ``['day_of_year', 'year']`` to extract the diurnal cycle amplitude for
        each individual day in the data. If the coordinates are not found in
        ``cube``, try to add it via :mod:`iris.coord_categorisation` (at the
        moment, this only works for the temporal coordinates ``day_of_month``,
        ``day_of_year``, ``hour``, ``month``, ``month_fullname``,
        ``month_number``, ``season``, ``season_number``, ``season_year``,
        ``weekday``, ``weekday_fullname``, ``weekday_number`` or ``year``.

    Returns
    -------
    iris.cube.Cube
        Amplitudes.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        A coordinate is not found in ``cube`` and cannot be added via
        :mod:`iris.coord_categorisation`.

    """
    if isinstance(coords, str):
        coords = [coords]

    # Add coordinate if necessary
    for coord_name in coords:
        if cube.coords(coord_name):
            continue
        logger.debug("Trying to add coordinate '%s' to cube via iris."
                     "coord_categorisation", coord_name)
        if hasattr(iris.coord_categorisation, f'add_{coord_name}'):
            getattr(iris.coord_categorisation, f'add_{coord_name}')(cube,
                                                                    'time')
            logger.debug("Added temporal coordinate '%s'", coord_name)
        else:
            raise iris.exceptions.CoordinateNotFoundError(
                f"Coordinate '{coord_name}' is not a coordinate of cube "
                f"{cube.summary(shorten=True)} and cannot be added via "
                f"iris.coord_categorisation")

    # Calculate amplitude
    max_cube = cube.aggregated_by(coords, iris.analysis.MAX)
    min_cube = cube.aggregated_by(coords, iris.analysis.MIN)
    amplitude_cube = max_cube - min_cube
    amplitude_cube.metadata = cube.metadata

    return amplitude_cube
