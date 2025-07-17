"""Test derivation of `vegfrac`."""

from esmvalcore.preprocessor._derive import vegfrac


def test_vegfrac_required_cmip5():
    derived_var = vegfrac.DerivedVariable()
    output = derived_var.required("CMIP5")
    assert output == [
        {"short_name": "baresoilFrac"},
        {"short_name": "residualFrac"},
        {"short_name": "sftlf", "mip": "fx", "ensemble": "r0i0p0"},
    ]


def test_vegfrac_required_cmip6():
    derived_var = vegfrac.DerivedVariable()
    output = derived_var.required("CMIP6")
    assert output == [
        {"short_name": "baresoilFrac"},
        {"short_name": "residualFrac"},
        {"short_name": "sftlf", "mip": "fx"},
    ]
