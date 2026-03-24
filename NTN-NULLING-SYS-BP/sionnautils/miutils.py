# -*- coding: utf-8 -*-
"""
miutils.py:  Mitsuba utilities
"""
import mitsuba as mi
import numpy as np

# class CoverageMapPlanner(object):
#     """
#     Class to plan coverage maps for a scene in Mitsuba
#     """
    
#     def __init__(self, scene, bldg_tol=1, grid_size=10, bbox=None):
#         """
#         Args:
#             scene: mitsuba scene
#                 Scene to trace rays in
#             bldg_tol: float
#                 Minimum distance between the max and min heights of objects
#                 to consider that a point has a building at that location.
#                 Value in meters
#             grid_size: float
#                 Size of the grid in each dimension in meters
#             bbox : float array of shape (4) | None
#                 Bounding box for the grid `[xmin, xmax, ymin, ymax]`.
#                 If `None` is specified, the bounding box is taken from 
#                 `scene.bbox()`.
#         """
#         self.scene = scene
#         self.bldg_tol = bldg_tol
#         self.grid_size = grid_size
#         if bbox is None:
#             b = self.scene.bbox()
#             self.bbox = np.array([b.min.x, b.max.x, b.min.y, b.max.y])
#         else:
#             self.bbox = bbox
        

#     def set_grid(self):
#         """
#         Create a grid of points on the scene's (x,y) bounding box

#         Sets the following attributes:
#         x, y: (nx,) and (ny,) arrays 
#             x and y coordinates along each axis
#         xgrid, ygrid: (ny,nx) 
#             2D mesh grid of all the points 
#         """
#         self.x = np.arange(self.bbox[0], self.bbox[1], self.grid_size)
#         self.y = np.arange(self.bbox[2], self.bbox[3], self.grid_size)
#         self.nx = len(self.x)
#         self.ny = len(self.y)
#         self.xgrid, self.ygrid = np.meshgrid(self.x, self.y)
        
#         self.yvec = self.ygrid.flatten()

 
#     def compute_grid_attributes(self):
#         """
#         Compute various attributes of the points on the grid

#         Sets:
#         zmin_grid, zmax_grid: (ny,nx) arrays
#             Min and max z coordinate of each grid point
#         bldg_grid: (ny,nx) array
#             True if the point has a building as defined
#             by `zmax_grid - zmin_grid > bldg_tol`
#         in_region: (ny,nx) array
#             True if the point is within the region defined
#             by having at least one building point on the left and righ
#         """
      
#         # Create vectors from the grid points
#         xvec = self.xgrid.flatten()
#         yvec = self.ygrid.flatten()

#         # Trace from slightly the bottom of the scene
#         z = self.scene.bbox().min.z-1
#         npts = self.nx*self.ny
#         p0 = mi.Point3f(xvec, yvec, z*np.ones(npts))
#         directions = np.zeros((npts, 3))
#         directions[:,2] = 1
        
#         directions = mi.Vector3f(directions)
#         # directions = mi.Vector3f(directions.T) 
        
#         ray = mi.Ray3f(p0, directions)
#         si = self.scene.ray_intersect(ray)
#         p = np.array(si.p)
#         zmin = p[:,2]
#         self.zmin_grid = zmin.reshape(self.xgrid.shape)


#         # Trace from slightly above the top of the scene
#         z = self.scene.bbox().max.z+1
#         p0 = mi.Point3f(xvec, yvec, z*np.ones(npts))
#         directions = np.zeros((npts, 3))
#         directions[:,2] = -1
#         directions = mi.Vector3f(directions)
#         ray = mi.Ray3f(p0, directions)
#         si = self.scene.ray_intersect(ray)
#         p = np.array(si.p)
#         zmax = p[:,2]
#         self.zmax_grid = zmax.reshape(self.xgrid.shape)

#         # Find the points that are outside
#         self.bldg_grid = (self.zmax_grid - self.zmin_grid > self.bldg_tol)

#         # Find the points that are within the region
#         # This is defined as having a point in a building on the left and right
#         mleft = np.maximum.accumulate(self.bldg_grid, axis=1)
#         u = np.fliplr(self.bldg_grid)
#         mright= np.maximum.accumulate(u, axis=1)
#         mright = np.fliplr(mright)
#         self.in_region = mleft & mright
     
  
        


# class CoverageMapPlanner(object):
#     """
#     Class to plan coverage maps for a scene in Mitsuba
#     """

#     def __init__(self, scene, bldg_tol=1, grid_size=10, bbox=None):
#         """
#         Args:
#             scene: mitsuba scene
#                 Scene to trace rays in
#             bldg_tol: float
#                 Minimum distance between the max and min heights of objects
#                 to consider that a point has a building at that location.
#                 Value in meters
#             grid_size: float
#                 Size of the grid in each dimension in meters
#             bbox : float array of shape (4) | None
#                 Bounding box for the grid `[xmin, xmax, ymin, ymax]`.
#                 If `None` is specified, the bounding box is taken from 
#                 `scene.bbox()`.
#         """
#         self.scene = scene
#         self.bldg_tol = bldg_tol
#         self.grid_size = grid_size
#         if bbox is None:
#             b = self.scene.bbox()
#             self.bbox = np.array([b.min.x, b.max.x, b.min.y, b.max.y])
#         else:
#             self.bbox = bbox

#     def set_grid(self):
#         """
#         Create a grid of points on the scene's (x,y) bounding box

#         Sets the following attributes:
#         x, y: (nx,) and (ny,) arrays 
#             x and y coordinates along each axis
#         xgrid, ygrid: (ny,nx) 
#             2D mesh grid of all the points 
#         """
#         self.x = np.arange(self.bbox[0], self.bbox[1], self.grid_size)
#         self.y = np.arange(self.bbox[2], self.bbox[3], self.grid_size)
#         self.nx = len(self.x)
#         self.ny = len(self.y)
#         self.xgrid, self.ygrid = np.meshgrid(self.x, self.y)

#         self.yvec = self.ygrid.flatten()

#     def compute_grid_attributes(self):
#         """
#         Compute various attributes of the points on the grid

#         Sets:
#         zmin_grid, zmax_grid: (ny,nx) arrays
#             Min and max z coordinate of each grid point
#         bldg_grid: (ny,nx) array
#             True if the point has a building as defined
#             by `zmax_grid - zmin_grid > bldg_tol`
#         in_region: (ny,nx) array
#             True if the point is within the region defined
#             by having at least one building point on the left and right
#         """
#         xvec = self.xgrid.flatten()
#         yvec = self.ygrid.flatten()
#         npts = self.nx * self.ny

#         # Trace from bottom of the scene
#         z = self.scene.bbox().min.z - 1
#         p0 = mi.Point3f(xvec.astype(np.float32), yvec.astype(np.float32), np.full(npts, z, dtype=np.float32))
#         directions = np.zeros((3, npts), dtype=np.float32)
#         directions[2, :] = 1.0
#         ray = mi.Ray3f(p0, mi.Vector3f(directions))
#         si = self.scene.ray_intersect(ray)
#         p = np.array(si.p)
#         zmin = p[2, :]  # Access Z component
#         self.zmin_grid = zmin.reshape(self.xgrid.shape)

#         # Trace from top of the scene
#         z = self.scene.bbox().max.z + 1
#         p0 = mi.Point3f(xvec.astype(np.float32), yvec.astype(np.float32), np.full(npts, z, dtype=np.float32))
#         directions = np.zeros((3, npts), dtype=np.float32)
#         directions[2, :] = -1.0
#         ray = mi.Ray3f(p0, mi.Vector3f(directions))
#         si = self.scene.ray_intersect(ray)
#         p = np.array(si.p)
#         zmax = p[2, :]  # Access Z component
#         self.zmax_grid = zmax.reshape(self.xgrid.shape)

#         self.bldg_grid = (self.zmax_grid - self.zmin_grid > self.bldg_tol)

#         mleft = np.maximum.accumulate(self.bldg_grid, axis=1)
#         mright = np.maximum.accumulate(np.fliplr(self.bldg_grid), axis=1)
#         mright = np.fliplr(mright)
#         self.in_region = mleft & mright



# -*- coding: utf-8 -*-
"""
miutils.py: Mitsuba utilities
"""


class CoverageMapPlanner(object):
    """
    Class to plan coverage maps for a scene in Mitsuba
    """

    def __init__(self, scene, bldg_tol=1, grid_size=10, bbox=None):
        """
        Args:
            scene: mitsuba scene
            bldg_tol: float - threshold to detect buildings by height diff
            grid_size: float - size of grid per axis [meters]
            bbox: optional bounding box override [xmin, xmax, ymin, ymax]
        """
        self.scene = scene
        self.bldg_tol = bldg_tol
        self.grid_size = grid_size
        if bbox is None:
            b = self.scene.bbox()
            self.bbox = np.array([b.min.x, b.max.x, b.min.y, b.max.y])
        else:
            self.bbox = bbox

    def set_grid(self):
        """Create a grid of points in (x, y) plane"""
        self.x = np.arange(self.bbox[0], self.bbox[1], self.grid_size)
        self.y = np.arange(self.bbox[2], self.bbox[3], self.grid_size)
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.xgrid, self.ygrid = np.meshgrid(self.x, self.y)
        self.yvec = self.ygrid.flatten()

    def compute_grid_attributes(self, batch_size=None):
        """Compute z-min/z-max/bldg_mask/in-region over grid.

        batch_size: number of rays per chunk. Use to limit GPU memory.
        """
        xvec = self.xgrid.flatten().astype(np.float32)
        yvec = self.ygrid.flatten().astype(np.float32)
        npts = self.nx * self.ny

        def make_point3f(x, y, z):
            return mi.Point3f(mi.Float(x), mi.Float(y), mi.Float(z))

        def make_vector3f(xyz_3xn):
            return mi.Vector3f(
                mi.Float(xyz_3xn[0]),
                mi.Float(xyz_3xn[1]),
                mi.Float(xyz_3xn[2])
            )

        if batch_size is None:
            # Keep memory reasonable on GPU for very large grids
            batch_size = 250000 if npts > 500000 else npts

        def trace_z(z_start, dz):
            out = np.empty(npts, dtype=np.float32)
            for i in range(0, npts, batch_size):
                j = min(i + batch_size, npts)
                xb = xvec[i:j]
                yb = yvec[i:j]
                zvec = np.full(j - i, z_start, dtype=np.float32)
                p0 = make_point3f(xb, yb, zvec)

                directions = np.zeros((3, j - i), dtype=np.float32)
                directions[2, :] = dz
                d_vec = make_vector3f(directions)
                ray = mi.Ray3f(p0, d_vec)
                si = self.scene.ray_intersect(ray)
                out[i:j] = np.array(si.p)[2, :]
            return out

        # ---------- ZMIN (trace from below)
        z_below = self.scene.bbox().min.z - 1
        zmin = trace_z(z_below, 1.0)
        self.zmin_grid = zmin.reshape(self.xgrid.shape)

        # ---------- ZMAX (trace from above)
        z_above = self.scene.bbox().max.z + 1
        zmax = trace_z(z_above, -1.0)
        self.zmax_grid = zmax.reshape(self.xgrid.shape)

        # ---------- Building mask
        self.bldg_grid = (self.zmax_grid - self.zmin_grid > self.bldg_tol)

        # ---------- Region mask (has building left & right)
        mleft = np.maximum.accumulate(self.bldg_grid, axis=1)
        mright = np.maximum.accumulate(np.fliplr(self.bldg_grid), axis=1)
        self.in_region = mleft & np.fliplr(mright)
        
                # ---------- All-region (both horizontal and vertical)
        mtop = np.maximum.accumulate(self.bldg_grid, axis=0)
        mbottom = np.maximum.accumulate(np.flipud(self.bldg_grid), axis=0)
        mbottom = np.flipud(mbottom)

        self.in_allregion = (mleft & np.fliplr(mright)) | (mtop & mbottom)
        # 或者想更严格（横纵都要满足）就改成:
        # self.in_allregion = (mleft & np.fliplr(mright)) & (mtop & mbottom)
        
        
from sionna.rt import ITURadioMaterial

def assign_material(scene, obj_name: str, itu_type: str, thickness: float = 0.1, color=None):
    """
    Assign an ITU radio material to a specific object in the scene.

    - If a material with the same ITU type already exists in the scene, reuse it.
    - Otherwise, create a new ITURadioMaterial and assign it.

    Args:
        scene: Sionna RT Scene object
        obj_name: Name of the object to modify (e.g., "building_1")
        itu_type: ITU material type string (e.g., "marble", "brick", "concrete")
        thickness: Material thickness in meters (default: 0.1)
        color: Optional RGB tuple (r, g, b)
    """
    # 1) Search for an existing material of the same type
    existing_mat = None
    for _, obj in scene.objects.items():
        mat = getattr(obj, "radio_material", None)
        if mat is not None and getattr(mat, "type", None) == itu_type:
            existing_mat = mat
            break

    # 2) Get the target object
    target_obj = scene.get(obj_name)
    if target_obj is None:
        raise ValueError(f"Object {obj_name} does not exist in the scene")

    # 3) Reuse or create material
    if existing_mat is not None:
        target_obj.radio_material = existing_mat
        print(f"{obj_name} reused existing material '{itu_type}'")
    else:
        new_mat = ITURadioMaterial(
            name=itu_type,
            itu_type=itu_type,
            thickness=thickness,
            color=color
        )
        target_obj.radio_material = new_mat
        print(f"{obj_name} created and assigned new material '{itu_type}'")


def replace_material(scene, old_name: str, new_name: str, itu_type: str = None,
                     thickness: float = 0.1, color=None):
    """
    Replace all objects in the scene with material name `old_name` by `new_name`.

    - If a material with `new_name` already exists in the scene, reuse it.
    - Otherwise, create a new ITURadioMaterial(new_name, itu_type, ...).

    Args:
        scene: Sionna RT Scene object
        old_name: Material name to replace (e.g., "glass")
        new_name: New material name (e.g., "brick")
        itu_type: ITU material type (default: use new_name)
        thickness: Thickness of the new material
        color: Optional RGB tuple (r, g, b)

    Returns:
        count: Number of objects updated
    """
    # 1) Check if new material already exists
    new_mat = None
    for _, obj in scene.objects.items():
        mat = getattr(obj, "radio_material", None)
        if mat is not None and getattr(mat, "name", None) == new_name:
            new_mat = mat
            break

    # 2) If not, create a new material
    if new_mat is None:
        if itu_type is None:
            itu_type = new_name
        new_mat = ITURadioMaterial(
            name=new_name,
            itu_type=itu_type,
            thickness=thickness,
            color=color
        )

    # 3) Replace across objects
    count = 0
    for _, obj in scene.objects.items():
        mat = getattr(obj, "radio_material", None)
        if mat is not None and getattr(mat, "name", None) == old_name:
            obj.radio_material = new_mat
            count += 1

    print(f"Replacement complete: {old_name} -> {new_name}, updated {count} objects")
    return count
