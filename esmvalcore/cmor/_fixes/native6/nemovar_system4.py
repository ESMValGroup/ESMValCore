from ..fix import Fix

from iris.cube import CubeList
from ..native_datasets import NativeDatasetFix
import iris
import numpy as np
import dask.array as da
from cf_units import Unit

def create_vertex_lons(a):
    ny = a.shape[0]
    nx = a.shape[1]
    if nx == 1:  # Longitudes were integrated out
        if ny == 1:
            return da.zeros([a[0, 0]])
    # b=np.zeros([ny, nx, 4])
    b = da.zeros([ny, nx, 4])
    b[:, 1:nx, 0] = 0.5 * (a[:, 0:nx - 1] + a[:, 1:nx])
    b[:, 0, 0] = 1.5 * a[:, 0] - 0.5 * a[:, 1]
    b[:, 0:nx - 1, 1] = b[:, 1:nx, 0]
    b[:, nx - 1, 1] = 1.5 * a[:, nx - 1] - 0.5 * a[:, nx - 2]
    b[:, :, 2] = b[:, :, 1]
    b[:, :, 3] = b[:, :, 0]
    # b[b < 0] = b[b < 0] + 360
    b = (b+360)% 360.0

    return b
    
def create_vertex_lats(a):
    ny = a.shape[0]
    nx = a.shape[1]
    f = np.vectorize(lambda x: (x + 90) % 180 - 90)

    b = da.zeros([ny, nx, 4])
    b[1:ny, :, 0] = f(0.5 * (a[0:ny - 1, :] + a[1:ny, :]))
    b[0, :, 0] = f(2 * a[0, :] - b[1, :, 0])
    b[:, :, 1] = b[:, :, 0]
    b[0:ny - 1, :, 2] = b[1:ny, :, 0]
    b[ny - 1, :, 2] = f(1.5 * a[ny - 1, :] - 0.5 * a[ny - 2, :])
    b[:, :, 3] = b[:, :, 2]
    return b

def fix_time_coord(cube):
    old_time = cube.coord("time")
    cube.remove_coord('time')
    time=iris.coords.DimCoord(
          old_time.points,
          var_name='time',
          long_name='time',
          standard_name="time",
          units=old_time.units
        )
    cube.add_dim_coord(time, 0)
    return cube

def fix_i_j(cube):
    if len(cube.shape)==4:
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[3] +1),
                    var_name= 'i',
                    long_name='cell index along first dimension',
                    units=1,
                ),
                3
                )
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[2]+1),
                    long_name="cell index along second dimension",
                    var_name="j",
                    units=1),
                    2)

    if len(cube.shape)==3:
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[2] +1),
                    var_name= 'i',
                    long_name='cell index along first dimension',
                    units=1,
                ),
                2
                )
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[1]+1),
                    long_name="cell index along second dimension",
                    var_name="j",
                    units=1),
                    1)
    if len(cube.shape)==2:
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[1] +1),
                    var_name= 'i',
                    long_name='cell index along first dimension',
                    units=1,
                ),
                1
                )
        cube.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, cube.shape[0]+1),
                    long_name="cell index along second dimension",
                    var_name="j",
                    units=1),
                    0)
    return cube

def fix_lat_lon(cube):
    cube.coord("latitude").var_name="latitude"
    cube.coord("latitude").attributes={}
    cube.coord("longitude").var_name="longitude"
    cube.coord("longitude").attributes={}
    return cube



def fix_lat_lon_bounds(cube):
    cube.coord("longitude").bounds=create_vertex_lons(
        cube.coord("longitude").core_points())
    cube.coord("latitude").bounds=create_vertex_lats(
        cube.coord("latitude").core_points())
    return cube

def fix_vertical_level(cube):
    if cube.coords()[1].var_name=='lev':
        cube.coord("generic").rename("depth")
        cube.coord("depth").units="m"
    else:
        cube.coord("model_level_number").rename("depth")
    cube.coord("depth").var_name = "lev"
    cube.coord("depth").long_name = "ocean depth coordinate"
    cube.coord("depth").attributes={"positive":"down"}
    return(cube)

def fix_variable_info(self,cube):
    cube.attributes={}
    cube.var_name = str(self.vardef.short_name)
    cube.units=self.vardef.units
    cube.standard_name=str(self.vardef.standard_name)
    cube.long_name=str(self.vardef.long_name)
    return cube

class Thetao(NativeDatasetFix):
    """Fixes for Thetao."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes,"votemper")        
        return cube

class Tos(NativeDatasetFix):
    """Fixes for Tos."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes,"sosstsst")        
        return cube

class So(NativeDatasetFix):
    """Fixes for So."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes,"vosaline")        
        return cube

class Sos(NativeDatasetFix):
    """Fixes for So."""
    def fix_metadata(self, cubes):
        """Fix metadata."""
        cube = self.get_cube(cubes,"sosaline")
        return cube
class Areacello(NativeDatasetFix):
    """Fixes for Areacello."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        e1t = self.get_cube(cubes,"e1t")
        e2t = self.get_cube(cubes,"e2t")
        fix_i_j(e1t)
        fix_i_j(e2t)
        mask=self.get_cube(cubes,"tmask")[0][0]
        mask=da.from_array(mask.data)
        mask = ~mask.astype(bool)
        cube = e1t*e2t
        cube.data.mask = mask
        nav_lat = self.get_cube(cubes,"nav_lat")
        nav_lon = self.get_cube(cubes,"nav_lon")
        cube.add_aux_coord( iris.coords.AuxCoord(
            points=nav_lat.data, standard_name='latitude',
            long_name = "latitude", units = "degrees_north"
            ),(1,2))
        cube.add_aux_coord( iris.coords.AuxCoord(
            points=nav_lon.data, standard_name='longitude',
            units = "degrees_east",long_name = "longitude"
            ),(1,2))
        return cube
    
class Volcello(NativeDatasetFix):
    """Fixes for Volcello."""

    def fix_metadata(self, cubes):
        """Fix metadata."""
        e1t = self.get_cube(cubes,"e1t")
        e2t = self.get_cube(cubes,"e2t")
        e3t = self.get_cube(cubes,"e3t_0")[0]
        e3t.add_dim_coord(
                iris.coords.DimCoord(
                    np.arange(1, e3t.shape[0] +1),
                    var_name= 'depth',
                    long_name='depth',
                    units="m",
                ),
                0
                )
        fix_i_j(e1t)
        fix_i_j(e2t)

        mask=self.get_cube(cubes,"tmask")
        mask=da.from_array(mask.data)
        mask = ~mask.astype(bool)
        cube = e1t*e2t*e3t
        cube.data.mask = mask
        nav_lat = self.get_cube(cubes,"nav_lat")
        nav_lon = self.get_cube(cubes,"nav_lon")
        cube.add_aux_coord( iris.coords.AuxCoord(
            points=nav_lat.data, standard_name='latitude',
            long_name = "latitude", units = "degrees_north"
            ),(1,2))
        cube.add_aux_coord( iris.coords.AuxCoord(
            points=nav_lon.data, standard_name='longitude',
            units = "degrees_east",long_name = "longitude"
            ),(1,2))
        return cube
    
class AllVars(NativeDatasetFix):
    """Fixes for all variables."""
    def fix_metadata(self, cube):
        """Fix coordinates."""
        fix_variable_info(self, cube)
        if self.vardef.short_name not in ['areacello',"volcello"]:
            fix_i_j(cube)
            fix_time_coord(cube)
        fix_lat_lon(cube)
        if len(cube.shape)==4:
            fix_vertical_level(cube)
        fix_lat_lon_bounds(cube)
        if self.vardef.short_name in ['areacello']:
            cube=cube[0]
            cube=iris.util.squeeze(cube)
        return CubeList([cube])
