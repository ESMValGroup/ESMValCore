"""Create test data for tests of CMOR fixes."""
import os

import numpy as np
from netCDF4 import Dataset


def create_hyb_pres_file_without_ap(dataset, short_name):
    """Create dataset without vertical auxiliary coordinate ``ap``."""
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=2)
    dataset.createDimension('lat', size=3)
    dataset.createDimension('lon', size=4)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = [0.0]
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 6543-2-1'
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].units = '1'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lat'][:] = [-30.0, 0.0, 30.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0, 90.0, 120.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0]]
    dataset.variables['ps'][:] = np.arange(1 * 3 * 4).reshape(1, 3, 4)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'
    dataset.variables['ps'].additional_attribute = 'xyz'

    # Variable
    dataset.createVariable(short_name, np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables[short_name][:] = np.full((1, 2, 3, 4), 0.0,
                                               dtype=np.float32)
    dataset.variables[short_name].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables[short_name].units = '%'


def create_hyb_pres_file_with_a(dataset, short_name):
    """Create netcdf file with issues in hybrid pressure coordinate."""
    create_hyb_pres_file_without_ap(dataset, short_name)
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('p0', np.float64, dimensions=())
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['p0'][:] = 1.0
    dataset.variables['p0'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'p0: p0 a: a b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')


def create_hyb_pres_file_with_ap(dataset, short_name):
    """Create netcdf file with issues in hybrid pressure coordinate."""
    create_hyb_pres_file_without_ap(dataset, short_name)
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['ap'][:] = [1.0, 2.0]
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['ap'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'ap: ap b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'ap: ap_bnds b: b_bnds ps: ps')


def save_cl_file_with_a(save_path):
    """Create netcdf file for ``cl`` with ``a`` coordinate."""
    nc_path = os.path.join(save_path, 'common_cl_a.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hyb_pres_file_with_a(dataset, 'cl')
    dataset.close()
    print(f"Saved {nc_path}")


def save_cl_file_with_ap(save_path):
    """Create netcdf file for ``cl`` with ``ap`` coordinate."""
    nc_path = os.path.join(save_path, 'common_cl_ap.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hyb_pres_file_with_ap(dataset, 'cl')
    dataset.close()
    print(f"Saved {nc_path}")


def create_hybrid_height_file(dataset, short_name):
    """Create dataset with hybrid height coordinate."""
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=2)
    dataset.createDimension('lat', size=1)
    dataset.createDimension('lon', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = [0.0]
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 6543-2-1'
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_height_coordinate')
    dataset.variables['lev'].units = 'm'
    dataset.variables['lev'].formula_terms = 'a: lev b: b orog: orog'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_height_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_terms = (
        'a: lev_bnds b: b_bnds orog: orog')
    dataset.variables['lat'][:] = [0.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of height coordinate
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('orog', np.float64, dimensions=('lat', 'lon'))
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0]]
    dataset.variables['orog'][:] = [[0.0, 1.0]]
    dataset.variables['orog'].standard_name = 'surface_altitude'
    dataset.variables['orog'].units = 'm'

    # Variable
    dataset.createVariable(short_name, np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables[short_name][:] = np.full((1, 2, 1, 2), 0.0,
                                               dtype=np.float32)
    dataset.variables[short_name].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables[short_name].units = '%'


def save_cl_file_with_height(save_path):
    """Create netcdf file for ``cl`` with hybrid height coordinate."""
    nc_path = os.path.join(save_path, 'common_cl_hybrid_height.nc')
    dataset = Dataset(nc_path, mode='w')
    create_hybrid_height_file(dataset, 'cl')
    dataset.close()
    print(f"Saved {nc_path}")


def save_cnrm_cm6_1_cl_file(save_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(save_path, 'cnrm_cm6_1_cl.nc')
    dataset = Dataset(nc_path, mode='w')
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=3)
    dataset.createDimension('lat', size=2)
    dataset.createDimension('lon', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = [0.0]
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 6543-2-1'
    dataset.variables['lev'][:] = [1.0, 2.0, 4.0]
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].units = '1'
    dataset.variables['lev'].formula_term = (
        'ap: ap b: b ps: ps')  # Error in attribute intended
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0], [3.0, 5.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_term = (
        'ap: ap b: b ps: ps')  # Error in attribute intended
    dataset.variables['lat'][:] = [-30.0, 0.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    # Wrong shape of bounds is intended
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('bnds', 'lev'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('bnds', 'lev'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['ap'][:] = [1.0, 2.0, 5.0]
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5, 1.5], [3.0, 3.0, 6.0]]
    dataset.variables['b'][:] = [0.0, 1.0, 3.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5, 0.5], [2.0, 2.0, 5.0]]
    dataset.variables['ps'][:] = np.arange(1 * 2 * 2).reshape(1, 2, 2)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'

    # Cl variable
    dataset.createVariable('cl', np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables['cl'][:] = np.full((1, 3, 2, 2), 0.0, dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'

    dataset.close()
    print(f"Saved {nc_path}")


def save_cesm2_cl_file(save_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(save_path, 'cesm2_cl.nc')
    with Dataset(nc_path, mode='w') as dataset:
        dataset.createDimension('time', size=1)
        dataset.createDimension('lev', size=2)
        dataset.createDimension('lat', size=3)
        dataset.createDimension('lon', size=4)
        dataset.createDimension('bnds', size=2)

        # Dimensional variables
        dataset.createVariable('time', np.float64, dimensions=('time',))
        dataset.createVariable('lev', np.float64, dimensions=('lev',))
        dataset.createVariable('lev_bnds', np.float64, dimensions=('lev',
                                                                   'bnds'))
        dataset.createVariable('lat', np.float64, dimensions=('lat',))
        dataset.createVariable('lon', np.float64, dimensions=('lon',))
        dataset.variables['time'][:] = [0.0]
        dataset.variables['time'].standard_name = 'time'
        dataset.variables['time'].units = 'days since 6543-2-1'
        dataset.variables['lev'][:] = [1.0, 2.0]
        dataset.variables['lev'].bounds = 'lev_bnds'
        dataset.variables['lev'].units = 'hPa'
        dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
        dataset.variables['lev_bnds'].standard_name = (
            'atmosphere_hybrid_sigma_pressure_coordinate')
        dataset.variables['lev_bnds'].units = '1'
        dataset.variables['lev_bnds'].formula_terms = (
            'p0: p0 a: a_bnds b: b_bnds ps: ps')
        dataset.variables['lat'][:] = [-30.0, 0.0, 30.0]
        dataset.variables['lat'].standard_name = 'latitude'
        dataset.variables['lat'].units = 'degrees_north'
        dataset.variables['lon'][:] = [30.0, 60.0, 90.0, 120.0]
        dataset.variables['lon'].standard_name = 'longitude'
        dataset.variables['lon'].units = 'degrees_east'

        # Coordinates for derivation of pressure coordinate
        dataset.createVariable('a', np.float64, dimensions=('lev',))
        dataset.createVariable('a_bnds', np.float64, dimensions=('lev',
                                                                 'bnds'))
        dataset.createVariable('b', np.float64, dimensions=('lev',))
        dataset.createVariable('b_bnds', np.float64, dimensions=('lev',
                                                                 'bnds'))
        dataset.createVariable('p0', np.float64, dimensions=())
        dataset.createVariable('ps', np.float64,
                               dimensions=('time', 'lat', 'lon'))
        dataset.variables['a'][:] = [1.0, 2.0]
        dataset.variables['a'].bounds = 'a_bnds'
        dataset.variables['a_bnds'][:] = [[1.5, 3.0], [0.0, 1.5]]  # intended
        dataset.variables['b'][:] = [0.0, 1.0]
        dataset.variables['b'].bounds = 'b_bnds'
        dataset.variables['b_bnds'][:] = [[0.5, 2.0], [-1.0, 0.5]]  # intended
        dataset.variables['p0'][:] = 1.0
        dataset.variables['p0'].units = 'Pa'
        dataset.variables['ps'][:] = np.arange(1 * 3 * 4).reshape(1, 3, 4)
        dataset.variables['ps'].standard_name = 'surface_air_pressure'
        dataset.variables['ps'].units = 'Pa'

        # Cl variable
        dataset.createVariable('cl', np.float32,
                               dimensions=('time', 'lev', 'lat', 'lon'))
        dataset.variables['cl'][:] = np.full((1, 2, 3, 4),
                                             0.0, dtype=np.float32)
        dataset.variables['cl'].standard_name = (
            'cloud_area_fraction_in_atmosphere_layer')
        dataset.variables['cl'].units = '%'

    print(f"Saved {nc_path}")


def save_cesm2_waccm_cl_file(save_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(save_path, 'cesm2_waccm_cl.nc')
    dataset = Dataset(nc_path, mode='w')
    dataset.createDimension('lev', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].units = '1'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a'].bounds = 'a_bnds'
    dataset.variables['a_bnds'][:] = [[1.5, 0.0], [3.0, 1.5]]
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b'].bounds = 'b_bnds'
    dataset.variables['b_bnds'][:] = [[0.5, -1.0], [2.0, 0.5]]

    # Cl variable
    dataset.createVariable('cl', np.float32, dimensions=('lev',))
    dataset.variables['cl'][:] = np.full((2,), [0.0, 1.0], dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'

    dataset.close()
    print(f"Saved {nc_path}")


def save_gfdl_cm4_cl_file(save_path):
    """Create netcdf file with similar issues as ``cl``."""
    nc_path = os.path.join(save_path, 'gfdl_cm4_cl.nc')
    dataset = Dataset(nc_path, mode='w')
    dataset.createDimension('time', size=1)
    dataset.createDimension('lev', size=3)
    dataset.createDimension('lat', size=2)
    dataset.createDimension('lon', size=2)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = [0.0]
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 6543-2-1'
    dataset.variables['lev'][:] = [1.0, 2.0, 4.0]
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].units = '1'
    dataset.variables['lev'].formula_term = (
        'ap: ap b: b ps: ps')  # Error in attribute intended
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0], [3.0, 5.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lev_bnds'].formula_term = (
        'ap: ap_bnds b: b_bnds ps: ps')  # Error in attribute intended
    dataset.variables['lat'][:] = [-30.0, 0.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('ap', np.float64, dimensions=('lev',))
    dataset.createVariable('ap_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['ap'][:] = [1.0, 2.0, 5.0]
    dataset.variables['ap'].units = 'Pa'
    dataset.variables['ap_bnds'][:] = [[0.0, 1.5], [1.5, 3.0], [3.0, 6.0]]
    dataset.variables['b'][:] = [0.0, 1.0, 3.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0], [2.0, 5.0]]
    dataset.variables['ps'][:] = np.arange(1 * 2 * 2).reshape(1, 2, 2)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'

    # Cl variable
    dataset.createVariable('cl', np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables['cl'][:] = np.full((1, 3, 2, 2), 0.0, dtype=np.float32)
    dataset.variables['cl'].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables['cl'].units = '%'

    dataset.close()
    print(f"Saved {nc_path}")


def main():
    """Main function to create datasets."""
    save_path = os.path.dirname(os.path.abspath(__file__))
    save_cl_file_with_a(save_path)
    save_cl_file_with_ap(save_path)
    save_cl_file_with_height(save_path)
    save_cnrm_cm6_1_cl_file(save_path)
    save_cesm2_cl_file(save_path)
    save_cesm2_waccm_cl_file(save_path)
    save_gfdl_cm4_cl_file(save_path)


if __name__ == '__main__':
    main()
