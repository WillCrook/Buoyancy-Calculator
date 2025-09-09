import trimesh
import numpy as np
from trimesh.boolean import union, difference

class WECModel:
    def __init__(self, rho_water=1000, g=9.81):
        self.rho_water = rho_water
        self.g = g
        self.parts = []
        self.extra_mass = 0

    def load_cad(self, filepath, scale=1.0, density=None):
        """Load a CAD/mesh file and scale to meters if needed."""
        mesh = trimesh.load(filepath)
        mesh.apply_scale(scale)
        if density is not None:
            mesh.density = density
        self.parts.append(mesh)
        return mesh

    def add_cube(self, size=(0.3,0.3,0.3), position=(0,0,0), density=600):
        cube = trimesh.creation.box(extents=size)
        cube.apply_translation(position)
        cube.density = density
        self.parts.append(cube)
        return cube

    def add_cylinder(self, radius=0.075, height=0.15, position=(0,0,0), orientation='z', hollow=False, wall_thickness=0.005):
        cyl = trimesh.creation.cylinder(radius=radius, height=height)
        cyl.apply_translation(position)
        if orientation == 'x':
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0,1,0], cyl.center_mass))
        elif orientation == 'y':
            cyl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1,0,0], cyl.center_mass))
        if hollow:
            inner = trimesh.creation.cylinder(radius=radius-wall_thickness, height=height)
            inner.apply_translation(position)
            cyl = difference([cyl, inner])
        self.parts.append(cyl)
        return cyl

    def set_extra_mass(self, mass):
        self.extra_mass = mass

    def compute_buoyancy(self):
        """Compute combined volume, mass, buoyancy, and waterline height using voxel-based approach."""
        if len(self.parts) == 0:
            raise ValueError("No parts added")
        combined = union(self.parts)
        volume = combined.volume
        mass = 0
        for part in self.parts:
            density = getattr(part, 'density', 600)
            mass += part.volume * density
        mass += self.extra_mass
        buoyant_force = self.rho_water * self.g * volume

        # Voxelize the combined mesh
        pitch = min(combined.extents) / 50.0  # choose resolution based on smallest dimension
        voxelized = trimesh.voxel.creation.voxelize(combined, pitch=pitch)
        coords = voxelized.points

        # Sort voxels by z height
        sorted_indices = np.argsort(coords[:, 2])
        sorted_coords = coords[sorted_indices]

        # Calculate cumulative submerged volume from bottom up
        voxel_volume = pitch ** 3
        z_values = sorted_coords[:, 2]
        unique_z = np.unique(z_values)
        submerged_volume = 0
        waterline = None
        for z in unique_z:
            count = np.sum(z_values == z)
            submerged_volume += count * voxel_volume
            if submerged_volume >= mass / self.rho_water:
                waterline = z + pitch / 2  # approximate waterline at center of voxel layer
                break
        if waterline is None:
            waterline = combined.bounds[1][2]  # if not found, set to top of mesh

        print(f"Volume: {volume:.4f} m^3, Mass: {mass:.2f} kg, Buoyant Force: {buoyant_force:.2f} N")
        print(f"Estimated Waterline Height {waterline:.4f} m")
        return combined, waterline

    def show(self):
        """Visualise the WEC and waterline using voxel-based waterline calculation."""
        combined, waterline = self.compute_buoyancy()
        # Create waterline plane
        plane = trimesh.creation.box(extents=[combined.extents[0]*1.5, combined.extents[1]*1.5, 0.002])
        plane.apply_translation([combined.center_mass[0], combined.center_mass[1], waterline])
        plane.visual.face_colors = [0, 0, 255, 100]
        scene = trimesh.Scene([combined, plane])
        scene.show()

# --- Example usage ---
wec = WECModel()

# Option 1: Load CAD
cad_mesh = wec.load_cad('/Users/willcrook/Downloads/3D Printed Play Dart.stl', scale=0.001)

# Option 2: Build primitives
# main_cube = wec.add_cube(size=(0.3,0.3,0.3))
# v_cyl = wec.add_cylinder(radius=0.075, height=0.15, orientation='z')
# h_cyl = wec.add_cylinder(radius=0.075, height=0.15, orientation='x')

# Electronics mass
wec.set_extra_mass(0.5)

# Show mesh + waterline
wec.show()