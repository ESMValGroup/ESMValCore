"""Auxiliary functions for :mod:`iris`."""
import warnings

import numpy as np
from iris import NameConstraint

from esmvalcore.exceptions import ESMValCoreDeprecationWarning


def date2num(date, unit, dtype=np.float64):
    """Convert datetime object into numeric value with requested dtype.

    This is a custom version of :meth:`cf_units.Unit.date2num` that
    guarantees the correct dtype for the return value.

    Arguments
    ---------
    date : :class:`datetime.datetime` or :class:`cftime.datetime`
    unit : :class:`cf_units.Unit`
    dtype : a numpy dtype

    Returns
    -------
    :class:`numpy.ndarray` of type `dtype`
        The return value of ``unit.date2num`` with the requested dtype.
    """
    num = unit.date2num(date)
    try:
        return num.astype(dtype)
    except AttributeError:
        return dtype(num)


def var_name_constraint(var_name):
    """:class:`iris.Constraint` using ``var_name``.

    Warning
    -------
    .. deprecated:: 2.6.0
        This function has been deprecated in ESMValCore version 2.6.0 and is
        scheduled for removal in version 2.8.0. Please use the function
        :class:`iris.NameConstraint` with the argument ``var_name`` instead:
        this is an exact replacement.

    Parameters
    ----------
    var_name: str
        ``var_name`` used for the constraint.

    Returns
    -------
    iris.Constraint
        Constraint.

    """
    deprecation_msg = (
        "The function ``var_name_constraint`` has been deprecated in "
        "ESMValCore version 2.6.0 and is scheduled for removal in version "
        "2.8.0. Please use the function ``iris.NameConstraint`` with the "
        "argument ``var_name`` instead: this is an exact replacement."
    )
    warnings.warn(deprecation_msg, ESMValCoreDeprecationWarning)
    return NameConstraint(var_name=var_name)
