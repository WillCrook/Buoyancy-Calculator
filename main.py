import trimesh
import numpy as np
from trimesh.boolean import union
from scipy.optimize import brentq

class WECModel:
    def __init__(self, rho_water=1000, g=9.81): #density in kg/m^3 and g field strength in m/s^2
        self.rho_water = rho_water
        self.g = g
        self.parts = []
        self.extra_mass = 0

    def load_cad(self, filepath, scale=None, density=None, mass=None, ignore_holes=False):
        """Load a CAD/mesh file and scale to meters if needed."""
        mesh = trimesh.load(filepath)
        original_extents = mesh.extents.copy()
        
        print(f"Original extents: {original_extents}")
        print(f'Applying Scale {scale}')
        mesh.apply_scale(scale)
        print(f"Scaled extents: {mesh.extents}") 
        if not mesh.is_watertight:
            if ignore_holes:
                print(f"Note: Loaded mesh '{filepath}' is not watertight, but non-watertight regions are being ignored (assumed intentional holes/openings).")
            else:
                print(f"Warning: Loaded mesh '{filepath}' is not watertight. This may indicate accidental gaps or holes. Attempting to fill holes, but volume calculations may be inaccurate. Consider repairing or exporting a watertight STL.")
                filled = mesh.fill_holes()
                if filled:
                    print("Mesh holes filled to attempt repair.")
                else:
                    print("No holes filled; mesh repair not applied.")
        mesh.name = filepath

        if mass is not None and density is None:
            density = mass / mesh.volume
        elif density is not None and mass is None:
            mass = density * mesh.volume

        if density is not None:
            mesh.density = density
        if mass is not None:
            mesh.user_mass = mass

        self.parts.append(mesh)

        # --- Debug output ---
        print(f"Loaded CAD part: {filepath}")
        print(f"  Scaled size (extents): {mesh.extents}")
        print(f"  Volume: {mesh.volume:.3g} m^3")
        print(f"  Surface area: {mesh.area:.3g} m^2")
        if hasattr(mesh, 'user_mass') and mesh.user_mass is not None:
            print(f"  Explicit mass: {mesh.user_mass:.3g} kg")
        if hasattr(mesh, 'density'):
            print(f"  Density: {mesh.density:.3g} kg/m^3")
        print("---")

        return mesh

    def list_parts(self):
        """Print summary of all parts currently loaded."""
        print("All loaded WEC parts:")
        total_mass = 0
        for i, part in enumerate(self.parts, start=1):
            print(f"Part {i}:")
            print(f"  Extents: {part.extents}")
            print(f"  Volume: {part.volume:.3g} m^3")
            if hasattr(part, 'user_mass') and getattr(part, 'user_mass', None) is not None:
                print(f"  Explicit mass: {part.user_mass:.3g} kg")
            if hasattr(part, 'density'):
                print(f"  Density: {part.density} kg/m^3")
            print("---")
            total_mass += getattr(part, 'user_mass', 0)
        print(f"  Total Mass of the WEC: {total_mass:.3g}")
    
    def set_extra_mass(self, mass):
        self.extra_mass = mass

    def submerged_mesh(self, combined, waterline):
        """Return the submerged portion of the combined mesh below the waterline."""
        bounds = combined.bounds
        size_x = bounds[1][0] - bounds[0][0]
        size_y = bounds[1][1] - bounds[0][1]
        size_z = waterline - bounds[0][2]
        if size_z <= 0:
            return None
        water_box = trimesh.creation.box(extents=[size_x*2, size_y*2, size_z])
        water_box.apply_translation([combined.center_mass[0], combined.center_mass[1], bounds[0][2] + size_z/2])
        submerged = combined.intersection(water_box)
        return submerged

    def solve_equilibrium(self):
        if len(self.parts) == 0:
            raise ValueError("No parts added")
        if len(self.parts) == 1:
            combined = self.parts[0]
        else:
            try:
                combined = union(self.parts)
            except ValueError as e:
                if str(e) == "Not all meshes are volumes!":
                    print("Warning: Not all meshes are volumes; falling back to using the first mesh as the combined mesh.")
                    combined = self.parts[0]
                else:
                    raise
        mass = 0
        used_density = False
        for part in self.parts:
            part_mass = getattr(part, 'user_mass', None)
            if part_mass is not None:
                mass += part_mass
            elif hasattr(part, 'density') and part.density is not None:
                mass += part.volume * part.density
                used_density = True
            else:
                raise ValueError(f"Neither Mass or Density has been given. Unable to complete the calculations for {part.name}")

        mass += self.extra_mass
        
        print(f"Total mass (including extra mass): {mass:.3g} kg")
        weight = mass * self.g

        def func(z):
            submerged = self.submerged_mesh(combined, z)
            if submerged is None:
                vol = 0.0
            else:
                vol = submerged.volume
            buoyancy = self.rho_water * self.g * vol
            return buoyancy - weight

        z_min = combined.bounds[0][2] - combined.extents[2]
        z_max = combined.bounds[1][2] + combined.extents[2]

        f_min = func(z_min)
        f_max = func(z_max)

        if f_min * f_max > 0:
            if f_min < 0 and f_max < 0:
                print("The object is too heavy: fully sunk")
                waterline = combined.bounds[0][2]
                submerged = combined
            elif f_min > 0 and f_max > 0:
                print("The object is too light: floats without submerging")
                waterline = combined.bounds[1][2]
                submerged = None
            else:
                waterline = z_min
                submerged = None
        else:
            waterline = brentq(func, z_min, z_max, xtol=1e-5)
            submerged = self.submerged_mesh(combined, waterline)

        if submerged is None or (hasattr(submerged, 'volume') and submerged.volume == 0):
            cob = np.array([np.nan, np.nan, np.nan])
        else:
            cob = submerged.center_mass

        com = combined.center_mass

        print(f"Equilibrium Waterline: {waterline:.3g} m")
        print(f"Submerged Volume: {submerged.volume if submerged is not None else 0:.3g} m^3")
        print(f"Center of Buoyancy: {cob}")
        print(f"Center of Mass: {com}")

        return combined, waterline

    def show(self):
        """Visualise the WEC and waterline using equilibrium waterline calculation."""
        combined, waterline = self.solve_equilibrium()
        # Create waterline plane
        plane = trimesh.creation.box(extents=[combined.extents[0]*1.5, combined.extents[1]*1.5, 0.002])
        plane.apply_translation([combined.center_mass[0], combined.center_mass[1], waterline])
        plane.visual.face_colors = [0, 0, 255, 100]
        scene = trimesh.Scene([combined, plane])
        scene.show()


wec = WECModel()

#Load CAD
wec.load_cad('Data/Right_Outrigger.stl', scale = 0.001, density = None, mass = 2.4, ignore_holes=False)

# Electronics mass
wec.set_extra_mass(0) #kg for the electronics

# Show mesh + waterline
wec.show()