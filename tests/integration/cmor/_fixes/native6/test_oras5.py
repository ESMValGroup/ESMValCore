        unstructured_grid_cubes,
        "sos",
        "oras5",
        "oras5",
        "Omon",
    )

    assert len(fixed_cubes) == 1
    fixed_cube = fixed_cubes[0]

    assert fixed_cube.shape == (2, 4)

    assert fixed_cube.coords("time", dim_coords=True)
    assert fixed_cube.coord_dims("time") == (0,)

    assert fixed_cube.coords("latitude", dim_coords=False)
    assert fixed_cube.coord_dims("latitude") == (1,)
    lat = fixed_cube.coord("latitude")
    np.testing.assert_allclose(lat.points, [1, 1, -1, -1])
    assert lat.bounds is None

    assert fixed_cube.coords("longitude", dim_coords=False)
    assert fixed_cube.coord_dims("longitude") == (1,)
    lon = fixed_cube.coord("longitude")
    np.testing.assert_allclose(lon.points, [179, 180, 180, 179])
    assert lon.bounds is None
