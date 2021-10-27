"""Auxiliary functions for :mod:`iris`."""
import iris
import numpy as np


def date2num(date, unit, dtype=np.float64):
    """Custom date2num function to guarantee return dtype.

    Arguments
    ---------
    date : :class:`datetime.datetime` or :class:`cftime.datetime`
    unit : :class:`cf_units.Unit`
    dtype : a numpy dtype

    Returns
    -------
    :class:`numpy.ndarray` of type `dtype`
        the return value of `unit.date2num` with the requested dtype
    """
    return unit.date2num(date).astype(dtype)


def var_name_constraint(var_name):
    """:mod:`iris.Constraint` using `var_name` of a :mod:`iris.cube.Cube`."""
    return iris.Constraint(cube_func=lambda c: c.var_name == var_name)
