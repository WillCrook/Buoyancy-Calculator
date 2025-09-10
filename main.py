import trimesh
import numpy as np
from trimesh.boolean import union
from scipy.optimize import brentq

class WECModel:
    def __init__(self, fluid_density=1000, g=9.81): #density in kg/m^3 and g field strength in m/s^2
        self.fluid_density = fluid_density
        self.g = g
        self.parts = []
        self.extra_mass = 0
        
    def load_cad(self, filepath, scale=None, density=None, mass=None, ignore_holes=False, rotate=True, rotations=None):
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

        if rotate and rotations is not None:
            center = mesh.center_mass
            for rotation in rotations:
                angle_rad = np.deg2rad(rotation['angle'])
                axis = rotation['axis']
                if axis == 'x':
                    axis_vector = [1, 0, 0]
                elif axis == 'y':
                    axis_vector = [0, 1, 0]
                elif axis == 'z':
                    axis_vector = [0, 0, 1]
                else:
                    raise ValueError("rotation axis must be 'x', 'y', or 'z'")
                rot_matrix = trimesh.transformations.rotation_matrix(angle_rad, axis_vector, point=center)
                mesh.apply_transform(rot_matrix)

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

        for i, part in enumerate(self.parts, start=1):
            print(f"Part {i}:")
            print(f"  Extents: {part.extents}")
            print(f"  Volume: {part.volume:.3g} m^3")
            if hasattr(part, 'user_mass') and getattr(part, 'user_mass', None) is not None:
                print(f"  Explicit mass: {part.user_mass:.3g} kg")
            if hasattr(part, 'density'):
                print(f"  Density: {part.density} kg/m^3")
            print("---")
    
    def set_extra_mass(self, mass):
        self.extra_mass = mass

    def set_fluid_density(self, density):
        self.fluid_density = density

    def set_gravity(self, g):
        self.g = g

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
        
        weight = mass * self.g

        def func(z):
            submerged = self.submerged_mesh(combined, z)
            if submerged is None:
                vol = 0.0
            else:
                vol = submerged.volume
            buoyancy = self.fluid_density * self.g * vol
            return buoyancy - weight

        z_min = combined.bounds[0][2] - combined.extents[2]
        z_max = combined.bounds[1][2] + combined.extents[2]

        f_min = func(z_min)
        f_max = func(z_max)

        if f_min * f_max > 0:
            if f_min < 0 and f_max < 0:
                print("The object is too heavy: fully sunk")
                waterline = combined.bounds[1][2]  # top of the object
                submerged = combined
            elif f_min > 0 and f_max > 0:
                print("The object is too light: floats without submerging")
                waterline = combined.bounds[0][2]  # bottom of the object
                submerged = None
            else:
                print("The object sits in the water")
                waterline = z_min
                submerged = None
        else:
            waterline = brentq(func, z_min, z_max, xtol=1e-5)
            submerged = self.submerged_mesh(combined, waterline)

        relative_waterline = waterline - combined.bounds[0][2]

        if submerged is None or (hasattr(submerged, 'volume') and submerged.volume == 0):
            cob = np.array([np.nan, np.nan, np.nan])
        else:
            cob = submerged.center_mass

        com = combined.center_mass

        return combined, waterline, relative_waterline, submerged, cob, com, mass

    def check_stability(self):
        """Estimate roll and pitch stability of the floating object."""
        if len(self.parts) == 0:
            raise ValueError("No parts added for stability check")
        
        combined, waterline, relative_waterline, submerged, cob, com, mass = self.solve_equilibrium()
        if submerged is None:
            print("Object is fully above water, stability check not applicable.")
            return None, None, None, None

        # Waterplane extents at waterline
        bounds = combined.bounds
        width_x = bounds[1][0] - bounds[0][0]
        width_y = bounds[1][1] - bounds[0][1]

        # Approximate second moments of area
        I_x = (width_y**3 * width_x) / 12  # roll
        I_y = (width_x**3 * width_y) / 12  # pitch

        V = submerged.volume

        # Metacentric height approximation
        BM_x = I_x / V
        BM_y = I_y / V

        # Distance from COB to COM
        BG = com[2] - cob[2]

        GM_x = BM_x - BG
        GM_y = BM_y - BG

        stable_roll = GM_x > 0
        stable_pitch = GM_y > 0

        return GM_x, GM_y, stable_roll, stable_pitch

    def show(self):
        """Visualise the WEC and waterline using equilibrium waterline calculation."""
        combined, waterline, relative_waterline, submerged, cob, com, mass = self.solve_equilibrium()
        GM_x, GM_y, stable_roll, stable_pitch = self.check_stability()
        # Create the fluid
        water_depth = combined.extents[2]*5  # depth of the fluid
        plane = trimesh.creation.box(
            extents=[combined.extents[0]*5, combined.extents[1]*5, water_depth]
        )
        plane.apply_translation([
            combined.center_mass[0],
            combined.center_mass[1],
            waterline - water_depth / 2
        ])
        plane.visual.face_colors = [0, 100, 255, 100] 
        scene = trimesh.Scene([combined, plane])
        scene.set_camera(angles=(np.pi/2, 0, 0), distance=combined.extents.max() * 5, center=combined.center_mass )

        #OUTPUT
        print(f"Waterline: {relative_waterline:.3g} m above bottom of object")
        print(f"Total Mass: {mass:.3g} kg")
        print(f"Submerged Volume: {submerged.volume if submerged is not None else 0:.3g} m^3")
        print(f"Center of Buoyancy: {cob}")
        print(f"Center of Mass: {com}")

        print(f"Stability Check:")
        print(f"  Roll GM: {GM_x:.4f} m -> {'Stable' if stable_roll else 'Unstable'}")
        print(f"  Pitch GM: {GM_y:.4f} m -> {'Stable' if stable_pitch else 'Unstable'}")
        
        scene.show()

wec = WECModel()
wec.set_fluid_density(1000)  # water kg/m^3
wec.set_gravity(9.81)        # g field strength m/s^2

#Load CAD
wec.load_cad('Data/Right_Outrigger.stl', 
             scale = 0.001, 
             density = None, 
             mass = 2.4, 
             ignore_holes=False, 
             rotate=True, 
             rotations=[{'axis':'y','angle':90}, {'axis': 'z', 'angle': 90}])

wec.load_cad('Data/Keel.stl', 
             scale = 0.001, 
             density = None, 
             mass = 1.9, 
             ignore_holes=False, 
             rotate=True, 
             rotations=[{'axis':'x','angle':90}, {'axis':'y','angle':0}, {'axis': 'z', 'angle': 90}])


# Electronics mass
# wec.set_extra_mass(0) #kg for the electronics

#test
# ball = trimesh.creation.icosphere(radius=0.02)  # 40 mm diameter
# ball.user_mass = 0.020  # mass in kg
# wec.parts = [ball]

# Show mesh + waterline
wec.show()