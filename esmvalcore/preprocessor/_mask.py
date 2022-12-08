"""Mask module.

Module that performs a number of masking operations that include:
masking with ancillary variables, masking with Natural Earth shapefiles
(land or ocean), masking on thresholds, missing values masking.
"""

import logging
import os
import warnings

import cartopy.io.shapereader as shpreader
import dask.array as da
import iris
import numpy as np
import shapely.vectorized as shp_vect
from iris.analysis import Aggregator
from iris.util import rolling_window

from esmvalcore.exceptions import ESMValCoreDeprecationWarning

logger = logging.getLogger(__name__)


def _get_fx_mask(fx_data, fx_option, mask_type):
    """Build a percentage-thresholded mask from an fx file."""
    inmask = np.zeros_like(fx_data, bool)
    if mask_type == 'sftlf':
        if fx_option == 'land':
            # Mask land out
            inmask[fx_data > 50.] = True
        elif fx_option == 'sea':
            # Mask sea out
            inmask[fx_data <= 50.] = True
    elif mask_type == 'sftof':
        if fx_option == 'land':
            # Mask land out
            inmask[fx_data < 50.] = True
        elif fx_option == 'sea':
            # Mask sea out
            inmask[fx_data >= 50.] = True
    elif mask_type == 'sftgif':
        if fx_option == 'ice':
            # Mask ice out
            inmask[fx_data > 50.] = True
        elif fx_option == 'landsea':
            # Mask landsea out
            inmask[fx_data <= 50.] = True

    return inmask


def _apply_fx_mask(fx_mask, var_data):
    """Apply the fx data extracted mask on the actual processed data."""
    # Apply mask across
    if np.ma.is_masked(var_data):
        fx_mask |= var_data.mask

    # Build the new masked data
    var_data = np.ma.array(var_data, mask=fx_mask, fill_value=1e+20)

    return var_data


def mask_landsea(cube, mask_out, always_use_ne_mask=False):
    """Mask out either land mass or sea (oceans, seas and lakes).

    It uses dedicated ancillary variables (sftlf or sftof) or,
    in their absence, it applies a
    Natural Earth mask (land or ocean contours).
    Note that the Natural Earth masks have different resolutions:
    10m for land, and 50m for seas.
    These are more than enough for ESMValTool purposes.

    Parameters
    ----------
    cube: iris.cube.Cube
        data cube to be masked.

    mask_out: str
        either "land" to mask out land mass or "sea" to mask out seas.

    always_use_ne_mask: bool, optional (default: False)
        always apply Natural Earths mask, regardless if fx files are available
        or not.

        .. warning::
            This option has been deprecated in ESMValCore version 2.5. To
            always use Natural Earth masks, either explicitly remove all
            ``ancillary_variables`` from the input cube (when this function is
            used directly) or specify ``fx_variables: null`` as option for this
            preprocessor in the recipe (when this function is used as part of
            ESMValTool).

    Returns
    -------
    iris.cube.Cube
        Returns the masked iris cube.

    Raises
    ------
    ValueError
        Error raised if masking on irregular grids is attempted.
        Irregular grids are not currently supported for masking
        with Natural Earth shapefile masks.
    """
    # Dict to store the Natural Earth masks
    cwd = os.path.dirname(__file__)
    # ne_10m_land is fast; ne_10m_ocean is very slow
    shapefiles = {
        'land': os.path.join(cwd, 'ne_masks/ne_10m_land.shp'),
        'sea': os.path.join(cwd, 'ne_masks/ne_50m_ocean.shp')
    }

    if not always_use_ne_mask:
        # preserve importance order: try stflf first then sftof
        fx_cube = None
        try:
            fx_cube = cube.ancillary_variable('land_area_fraction')
        except iris.exceptions.AncillaryVariableNotFoundError:
            try:
                fx_cube = cube.ancillary_variable('sea_area_fraction')
            except iris.exceptions.AncillaryVariableNotFoundError:
                logger.debug('Ancillary variables land/sea area fraction '
                             'not found in cube. Check fx_file availability.')

        if fx_cube:
            fx_cube_data = da.broadcast_to(fx_cube.core_data(), cube.shape)
            landsea_mask = _get_fx_mask(fx_cube_data, mask_out,
                                        fx_cube.var_name)
            cube.data = _apply_fx_mask(landsea_mask, cube.data)
            logger.debug("Applying land-sea mask: %s", fx_cube.var_name)
        else:
            if cube.coord('longitude').points.ndim < 2:
                cube = _mask_with_shp(cube, shapefiles[mask_out], [
                    0,
                ])
                logger.debug(
                    "Applying land-sea mask from Natural Earth"
                    " shapefile: \n%s", shapefiles[mask_out])
            else:
                msg = ("Use of shapefiles with irregular grids not "
                       "yet implemented, land-sea mask not applied.")
                raise ValueError(msg)
    else:
        deprecation_msg = (
            "The option ``always_use_ne_masks``' has been deprecated in "
            "ESMValCore version 2.5. To always use Natural Earth masks, "
            "either explicitly remove all ``ancillary_variables`` from the "
            "input cube (when this function is used directly) or specify "
            "``fx_variables: null`` as option for this preprocessor in the "
            "recipe (when this function is used as part of ESMValTool).")
        warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning)
        if cube.coord('longitude').points.ndim < 2:
            cube = _mask_with_shp(cube, shapefiles[mask_out], [
                0,
            ])
            logger.debug(
                "Applying land-sea mask from Natural Earth"
                " shapefile: \n%s", shapefiles[mask_out])
        else:
            msg = ("Use of shapefiles with irregular grids not "
                   "yet implemented, land-sea mask not applied.")
            raise ValueError(msg)

    return cube


def mask_landseaice(cube, mask_out):
    """Mask out either landsea (combined) or ice.

    Function that masks out either landsea (land and seas) or ice (Antarctica
    and Greenland and some wee glaciers).

    It uses dedicated ancillary variables (sftgif).

    Parameters
    ----------
    cube: iris.cube.Cube
        data cube to be masked.

    mask_out: str
        either "landsea" to mask out landsea or "ice" to mask out ice.

    Returns
    -------
    iris.cube.Cube
        Returns the masked iris cube with either land or ice masked out.

    Raises
    ------
    ValueError
        Error raised if landsea-ice mask not found as an ancillary variable.
    """
    # sftgif is the only one so far but users can set others
    fx_cube = None
    try:
        fx_cube = cube.ancillary_variable('land_ice_area_fraction')
    except iris.exceptions.AncillaryVariableNotFoundError:
        logger.debug('Ancillary variable land ice area fraction '
                     'not found in cube. Check fx_file availability.')
    if fx_cube:
        fx_cube_data = da.broadcast_to(fx_cube.core_data(), cube.shape)
        landice_mask = _get_fx_mask(fx_cube_data, mask_out, fx_cube.var_name)
        cube.data = _apply_fx_mask(landice_mask, cube.data)
        logger.debug("Applying landsea-ice mask: sftgif")
    else:
        msg = "Landsea-ice mask could not be found. Stopping. "
        raise ValueError(msg)

    return cube


def mask_glaciated(cube, mask_out):
    """Mask out glaciated areas.

    It applies a Natural Earth mask. Note that for computational reasons
    only the 10 largest polygons are used for masking.

    Parameters
    ----------
    cube: iris.cube.Cube
        data cube to be masked.

    mask_out: str
        "glaciated" to mask out glaciated areas

    Returns
    -------
    iris.cube.Cube
        Returns the masked iris cube.

    Raises
    ------
    ValueError
        Error raised if masking on irregular grids is attempted or if
        mask_out has a wrong value.
    """
    # Dict to store the Natural Earth masks
    cwd = os.path.dirname(__file__)
    # read glaciated shapefile
    shapefiles = {
        'glaciated': os.path.join(cwd, 'ne_masks/ne_10m_glaciated_areas.shp'),
    }
    if mask_out == 'glaciated':
        cube = _mask_with_shp(cube, shapefiles[mask_out], [
            1859,
            1860,
            1861,
            1857,
            1858,
            1716,
            1587,
            1662,
            1578,
            1606,
        ])
        logger.debug(
            "Applying glaciated areas mask from Natural Earth"
            " shapefile: \n%s", shapefiles[mask_out])
    else:
        msg = (f"Invalid argument mask_out: {mask_out}")
        raise ValueError(msg)

    return cube


def _get_geometries_from_shp(shapefilename):
    """Get the mask geometries out from a shapefile."""
    reader = shpreader.Reader(shapefilename)
    # Index 0 grabs the lowest resolution mask (no zoom)
    geometries = [contour for contour in reader.geometries()]
    if not geometries:
        msg = "Could not find any geometry in {}".format(shapefilename)
        raise ValueError(msg)

    # TODO might need this for a later, more enhanced, version
    # geometries = sorted(geometries, key=lambda x: x.area, reverse=True)

    return geometries


def _mask_with_shp(cube, shapefilename, region_indices=None):
    """Apply a Natural Earth land/sea mask.

    Apply a pre-made land or sea mask that is extracted form a Natural
    Earth shapefile (proprietary file format). The masking process is
    performed by checking if any given (x, y) point from the data cube
    lies within the desired geometries (eg land, sea) stored in the
    shapefile (this is done via shapefle vectorization and is fast).
    region_indices is a list of indices that the user will want to index
    the regions on (select a region by its index as it is listed in the
    shapefile).
    """
    # Create the region
    regions = _get_geometries_from_shp(shapefilename)
    if region_indices:
        regions = [regions[idx] for idx in region_indices]

    # Create a mask for the data
    mask = np.zeros(cube.shape, dtype=bool)

    # Create a set of x,y points from the cube
    # 1D regular grids
    if cube.coord('longitude').points.ndim < 2:
        x_p, y_p = np.meshgrid(
            cube.coord(axis='X').points,
            cube.coord(axis='Y').points)
    # 2D irregular grids; spit an error for now
    else:
        msg = ("No fx-files found (sftlf or sftof)!"
               "2D grids are suboptimally masked with "
               "Natural Earth masks. Exiting.")
        raise ValueError(msg)

    # Wrap around longitude coordinate to match data
    x_p_180 = np.where(x_p >= 180., x_p - 360., x_p)
    # the NE mask has no points at x = -180 and y = +/-90
    # so we will fool it and apply the mask at (-179, -89, 89) instead
    x_p_180 = np.where(x_p_180 == -180., x_p_180 + 1., x_p_180)
    y_p_0 = np.where(y_p == -90., y_p + 1., y_p)
    y_p_90 = np.where(y_p_0 == 90., y_p_0 - 1., y_p_0)

    for region in regions:
        # Build mask with vectorization
        if cube.ndim == 2:
            mask = shp_vect.contains(region, x_p_180, y_p_90)
        elif cube.ndim == 3:
            mask[:] = shp_vect.contains(region, x_p_180, y_p_90)
        elif cube.ndim == 4:
            mask[:, :] = shp_vect.contains(region, x_p_180, y_p_90)

        # Then apply the mask
        if isinstance(cube.data, np.ma.MaskedArray):
            cube.data.mask |= mask
        else:
            cube.data = np.ma.masked_array(cube.data, mask)

    return cube


def count_spells(data, threshold, axis, spell_length):
    """Count data occurrences.

    Define a function to perform the custom statistical operation.
    Note: in order to meet the requirements of iris.analysis.Aggregator,
    it must do the calculation over an arbitrary (given) data axis.

    Function to calculate the number of points in a sequence where the value
    has exceeded a threshold value for at least a certain number of timepoints.

    Generalised to operate on multiple time sequences arranged on a specific
    axis of a multidimensional array.

    Parameters
    ----------
    data: ndarray
        raw data to be compared with value threshold.

    threshold: float
        threshold point for 'significant' datapoints.

    axis: int
        number of the array dimension mapping the time sequences.
        (Can also be negative, e.g. '-1' means last dimension)

    spell_length: int
        number of consecutive times at which value > threshold to "count".

    Returns
    -------
    int
        Number of counts.
    """
    if axis < 0:
        # just cope with negative axis numbers
        axis += data.ndim
    # Threshold the data to find the 'significant' points.
    if not threshold:
        data_hits = data
    else:
        data_hits = data > float(threshold)
    # Make an array with data values "windowed" along the time axis.
    ###############################################################
    # WARNING: default step is = window size i.e. no overlapping
    # if you want overlapping windows set the step to be m*spell_length
    # where m is a float
    ###############################################################
    hit_windows = rolling_window(data_hits,
                                 window=spell_length,
                                 step=spell_length,
                                 axis=axis)
    # Find the windows "full of True-s" (along the added 'window axis').
    full_windows = np.all(hit_windows, axis=axis + 1)
    # Count points fulfilling the condition (along the time axis).
    spell_point_counts = np.sum(full_windows, axis=axis, dtype=int)
    return spell_point_counts


def mask_above_threshold(cube, threshold):
    """Mask above a specific threshold value.

    Takes a value 'threshold' and masks off anything that is above
    it in the cube data. Values equal to the threshold are not masked.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded.

    threshold: float
        threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        thresholded cube.
    """
    cube.data = np.ma.masked_where(cube.data > threshold, cube.data)
    return cube


def mask_below_threshold(cube, threshold):
    """Mask below a specific threshold value.

    Takes a value 'threshold' and masks off anything that is below
    it in the cube data. Values equal to the threshold are not masked.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    threshold: float
        threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        thresholded cube.
    """
    cube.data = np.ma.masked_where(cube.data < threshold, cube.data)
    return cube


def mask_inside_range(cube, minimum, maximum):
    """Mask inside a specific threshold range.

    Takes a MINIMUM and a MAXIMUM value for the range, and masks off anything
    that's between the two in the cube data.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        thresholded cube.
    """
    cube.data = np.ma.masked_inside(cube.data, minimum, maximum)
    return cube


def mask_outside_range(cube, minimum, maximum):
    """Mask outside a specific threshold range.

    Takes a MINIMUM and a MAXIMUM value for the range, and masks off anything
    that's outside the two in the cube data.

    Parameters
    ----------
    cube: iris.cube.Cube
        iris cube to be thresholded
    minimum: float
        lower threshold to be applied on input cube data.
    maximum: float
        upper threshold to be applied on input cube data.

    Returns
    -------
    iris.cube.Cube
        thresholded cube.
    """
    cube.data = np.ma.masked_outside(cube.data, minimum, maximum)
    return cube


def _get_shape(cubes):
    """Check and get shape of cubes."""
    shapes = {cube.shape for cube in cubes}
    if len(shapes) > 1:
        raise ValueError(
            f"Expected cubes with identical shapes, got shapes {shapes}")
    return list(shapes)[0]


def _multimodel_mask_cubes(cubes, shape):
    """Apply common mask to all cubes in-place."""
    # Create mask
    mask = da.full(shape, False, dtype=bool)
    for cube in cubes:
        new_mask = da.ma.getmaskarray(cube.core_data())
        mask |= new_mask

    # Apply common mask
    for cube in cubes:
        cube.data = da.ma.masked_array(cube.core_data(), mask=mask)

    return cubes


def _multimodel_mask_products(products, shape):
    """Apply common mask to all cubes of products in-place."""
    # Create mask and get products used for mask
    mask = da.full(shape, False, dtype=bool)
    used_products = set()
    for product in products:
        for cube in product.cubes:
            new_mask = da.ma.getmaskarray(cube.core_data())
            mask |= new_mask
            if da.any(new_mask):
                used_products.add(product)

    # Apply common mask and update provenance information
    for product in products:
        for cube in product.cubes:
            cube.data = da.ma.masked_array(cube.core_data(), mask=mask)
        for other_product in used_products:
            if other_product.filename != product.filename:
                product.wasderivedfrom(other_product)

    return products


def mask_multimodel(products):
    """Apply common mask to all datasets (using logical OR).

    Parameters
    ----------
    products : iris.cube.CubeList or list of PreprocessorFile
        Data products/cubes to be masked.

    Returns
    -------
    iris.cube.CubeList or list of PreprocessorFile
        Masked data products/cubes.

    Raises
    ------
    ValueError
        Datasets have different shapes.
    TypeError
        Invalid input data.
    """
    if not products:
        return products

    # Check input types
    if all(isinstance(p, iris.cube.Cube) for p in products):
        cubes = products
        shape = _get_shape(cubes)
        return _multimodel_mask_cubes(cubes, shape)
    if all(type(p).__name__ == 'PreprocessorFile' for p in products):
        # Avoid circular input: https://stackoverflow.com/q/16964467
        cubes = iris.cube.CubeList()
        for product in products:
            cubes.extend(product.cubes)
        if not cubes:
            return products
        shape = _get_shape(cubes)
        return _multimodel_mask_products(products, shape)
    product_types = {type(p) for p in products}
    raise TypeError(
        f"Input type for mask_multimodel not understood. Expected "
        f"iris.cube.Cube or esmvalcore.preprocessor.PreprocessorFile, "
        f"got {product_types}")


def mask_fillvalues(products,
                    threshold_fraction,
                    min_value=None,
                    time_window=1):
    """Compute and apply a multi-dataset fillvalues mask.

    Construct the mask that fills a certain time window with missing values
    if the number of values in that specific window is less than a given
    fractional threshold.
    This function is the extension of _get_fillvalues_mask and performs the
    combination of missing values masks from each model (of multimodels)
    into a single fillvalues mask to be applied to each model.

    Parameters
    ----------
    products: iris.cube.Cube
        data products to be masked.

    threshold_fraction: float
        fractional threshold to be used as argument for Aggregator.
        Must be between 0 and 1.

    min_value: float
        minimum value threshold; default None
        If default, no thresholding applied so the full mask will be selected.

    time_window: float
        time window to compute missing data counts; default set to 1.

    Returns
    -------
    iris.cube.Cube
        Masked iris cubes.

    Raises
    ------
    NotImplementedError
        Implementation missing for data with higher dimensionality than 4.
    """
    combined_mask = None

    logger.debug("Creating fillvalues mask")
    used = set()
    for product in products:
        for cube in product.cubes:
            cube.data = np.ma.fix_invalid(cube.data, copy=False)
            mask = _get_fillvalues_mask(cube, threshold_fraction, min_value,
                                        time_window)
            if combined_mask is None:
                combined_mask = np.zeros_like(mask)
            # Select only valid (not all masked) pressure levels
            n_dims = len(mask.shape)
            if n_dims == 2:
                valid = ~np.all(mask)
                if valid:
                    combined_mask |= mask
                    used.add(product)
            elif n_dims == 3:
                valid = ~np.all(mask, axis=(1, 2))
                combined_mask[valid] |= mask[valid]
                if np.any(valid):
                    used.add(product)
            else:
                raise NotImplementedError(
                    "Unable to handle {} dimensional data".format(n_dims))

    if np.any(combined_mask):
        logger.debug("Applying fillvalues mask")
        used = {p.copy_provenance() for p in used}
        for product in products:
            for cube in product.cubes:
                cube.data.mask |= combined_mask
            for other in used:
                if other.filename != product.filename:
                    product.wasderivedfrom(other)

    return products


def _get_fillvalues_mask(cube, threshold_fraction, min_value, time_window):
    """Compute the per-model missing values mask.

    Construct the mask that fills a certain time window with missing
    values if the number of values in that specific window is less than
    a given fractional threshold; it uses a custom iris Aggregator
    function that aggregates the cube data by a given time window and
    counts the number of valid (unmasked) data points within that
    window; a simple value thresholding is also applied if needed.
    """
    # basic checks
    if threshold_fraction < 0 or threshold_fraction > 1.0:
        raise ValueError(
            "Fraction of missing values {} should be between 0 and 1.0".format(
                threshold_fraction))
    nr_time_points = len(cube.coord('time').points)
    if time_window > nr_time_points:
        msg = "Time window (in time units) larger than total time span. Stop."
        raise ValueError(msg)

    max_counts_per_time_window = nr_time_points / time_window
    # round to lower integer
    counts_threshold = int(max_counts_per_time_window * threshold_fraction)

    # Make an aggregator
    spell_count = Aggregator('spell_count',
                             count_spells,
                             units_func=lambda units: 1)

    # Calculate the statistic.
    counts_windowed_cube = cube.collapsed('time',
                                          spell_count,
                                          threshold=min_value,
                                          spell_length=time_window)

    # Create mask
    mask = counts_windowed_cube.data < counts_threshold
    if np.ma.isMaskedArray(mask):
        mask = mask.data | mask.mask

    return mask
