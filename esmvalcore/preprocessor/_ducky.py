import iris


def _duck_at_pond():
    """Look for a duck at the pond."""
    c = iris.load_cube("moo.nc")
    return c
