
import trimesh
import numpy as np
from trimesh.boolean import union
from scipy.optimize import brentq

# Reusable dummy submerged mesh class for manual volume cases
class DummySubmerged:
    def __init__(self, volume, center_mass):
        self.volume = volume
        self.center_mass = center_mass

class WECModel:
    def __init__(self, fluid_density=1000, g=9.81): #density in kg/m^3 and g field strength in m/s^2
        self.fluid_density = fluid_density
        self.g = g
        self.parts = []
    
    def check_all_meshes_watertight(self, visualize=False):
        """
        Check all loaded parts for watertightness.
        Prints a summary and optionally visualizes open vertices in red.
        """
        for part in self.parts:
            if part.is_watertight:
                print(f"Mesh '{part.name}' is watertight")
            else:
                open_facets = part.facets_boundary
                print(f"Mesh '{part.name}' is NOT watertight")
                print(f"  Number of open facets: {len(open_facets)}")

                if visualize:
                    scene = trimesh.Scene()
                    scene.add_geometry(part)

                    # Collect unique vertices from first 10 open facets
                    vertices_indices = np.unique(np.concatenate(open_facets))
                    # Create a point cloud for these vertices
                    points = trimesh.points.PointCloud(vertices=part.vertices[vertices_indices])
                    points.visual.vertex_colors = [255, 0, 0, 255]  # red
                    scene.add_geometry(points)
                    
                    scene.show()
        
    def load_cad(self, filepath, scale=None, density=None, mass=None, ignore_holes=False, rotate=True, rotations=None, manual_volume=None, manual_com=None):
        """Load a CAD/mesh file and scale to meters if needed."""
        mesh = trimesh.load(filepath)
        mesh.apply_scale(scale)
        
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

        if manual_volume is not None:
            mesh._manual_volume = manual_volume
            if manual_com is not None:
                mesh.center_mass = np.array(manual_com)
        else:
            mesh._manual_volume = None

        effective_volume = manual_volume if manual_volume is not None else mesh.volume

        if mass is not None and density is None:
            density = mass / effective_volume
        elif density is not None and mass is None:
            mass = density * effective_volume

        if density is not None:
            mesh.density = density
        if mass is not None:
            mesh.user_mass = mass

        self.parts.append(mesh)

       
        print(f"Loaded CAD part: {filepath}")
        print(f"  Scaled size (extents): {mesh.extents}")
        if manual_volume is not None:
            print(f"  Manual Volume: {manual_volume:.3g} m^3")
            if manual_com is not None:
                print(f"  Manual Center of Mass: {manual_com}")
        else:
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
            vol = part._manual_volume if hasattr(part, '_manual_volume') and part._manual_volume is not None else part.volume
            print(f"  Volume: {vol:.3g} m^3")
            if hasattr(part, 'user_mass') and getattr(part, 'user_mass', None) is not None:
                print(f"  Explicit mass: {part.user_mass:.3g} kg")
            if hasattr(part, 'density'):
                print(f"  Density: {part.density} kg/m^3")
            print("---")
    
    def set_fluid_density(self, density):
        self.fluid_density = density

    def set_gravity(self, g):
        self.g = g

    def submerged_mesh(self, mesh, waterline):
        """
        Return the submerged portion of the mesh below the waterline.
        If the mesh has a manual volume (_manual_volume), return a dummy mesh-like object
        with volume/center_mass attributes for buoyancy calculations, skipping boolean operations.
        Otherwise, intersect mesh with the water box.
        """
        bounds = mesh.bounds
        size_x = bounds[1][0] - bounds[0][0]
        size_y = bounds[1][1] - bounds[0][1]
        size_z = waterline - bounds[0][2]
        if size_z <= 0:
            return None

        # If mesh has manual volume, skip boolean and return dummy
        if hasattr(mesh, "_manual_volume") and mesh._manual_volume is not None:
            # Fraction of volume below waterline
            z_min = bounds[0][2]
            z_max = bounds[1][2]
            total_height = z_max - z_min
            if total_height <= 0:
                frac_sub = 0.0
            else:
                # If waterline above the mesh, fully submerged
                if waterline >= z_max:
                    frac_sub = 1.0
                # If waterline below the mesh, not submerged
                elif waterline <= z_min:
                    frac_sub = 0.0
                else:
                    frac_sub = (waterline - z_min) / total_height
            sub_vol = mesh._manual_volume * frac_sub
            # Center of buoyancy: interpolate between bottom and center of mass
            # For a uniform vertical distribution, COB is at z_min + (frac_sub/2)*height
            if hasattr(mesh, "center_mass"):
                # Use manual center_mass if provided, but shift z to the centroid of submerged part
                center_xy = np.array(mesh.center_mass[:2])
            else:
                center_xy = np.array([(bounds[0][0] + bounds[1][0]) / 2,
                                      (bounds[0][1] + bounds[1][1]) / 2])
            # Center of buoyancy in z: halfway up the submerged height
            cob_z = z_min + (frac_sub * total_height) / 2
            cob = np.array([center_xy[0], center_xy[1], cob_z])
            return DummySubmerged(sub_vol, cob)
        
        else:
            # Fallback: use trimesh intersection
            water_box = trimesh.creation.box(extents=[size_x*2, size_y*2, size_z])
            water_box.apply_translation([mesh.center_mass[0], mesh.center_mass[1], bounds[0][2] + size_z/2])
            try:
                submerged = mesh.intersection(water_box)
            except Exception:
                # If intersection fails, return None
                return None
            return submerged

    def solve_equilibrium(self):
        if len(self.parts) == 0:
            raise ValueError("No parts added")

        # Calculate total mass and total weight
        total_mass = 0
        for part in self.parts:
            part_mass = getattr(part, 'user_mass', None)
            if part_mass is not None:
                total_mass += part_mass
            elif hasattr(part, 'density') and part.density is not None:
                vol = part._manual_volume if hasattr(part, '_manual_volume') and part._manual_volume is not None else part.volume
                total_mass += vol * part.density
            else:
                raise ValueError(f"Neither Mass or Density has been given. Unable to complete the calculations for {part.name}")

        total_weight = total_mass * self.g

        # Determine z bounds for root finding from all parts bounds
        min_z = min(part.bounds[0][2] for part in self.parts)
        max_z = max(part.bounds[1][2] for part in self.parts)
        range_z = max_z - min_z
        z_min = min_z - range_z
        z_max = max_z + range_z

        def total_buoyancy(z):
            total_buoy = 0.0
            total_vol = 0.0
            weighted_cob = np.zeros(3)
            for part in self.parts:
                submerged = self.submerged_mesh(part, z)
                if submerged is None or submerged.volume == 0:
                    vol = 0.0
                    cob = np.zeros(3)
                else:
                    vol = submerged.volume
                    cob = submerged.center_mass
                buoy = self.fluid_density * self.g * vol
                total_buoy += buoy
                total_vol += vol
                weighted_cob += cob * vol
            if total_vol > 0:
                self._current_cob = weighted_cob / total_vol
            else:
                self._current_cob = np.array([np.nan, np.nan, np.nan])
            return total_buoy - total_weight

        f_min = total_buoyancy(z_min)
        f_max = total_buoyancy(z_max)

        if f_min * f_max > 0:
            if f_min < 0 and f_max < 0:
                print("The object is too heavy: fully sunk")
                waterline = max_z
                submerged = None
            elif f_min > 0 and f_max > 0:
                print("The object is too light: floats without submerging")
                waterline = min_z
                submerged = None
            else:
                waterline = z_min
                submerged = None
        else:
            print("The object sits in the water")
            waterline = brentq(total_buoyancy, z_min, z_max, xtol=1e-5)
            # Calculate submerged meshes at equilibrium waterline
            submerged_parts = []
            for part in self.parts:
                sm = self.submerged_mesh(part, waterline)
                if sm is not None and sm.volume > 0:
                    submerged_parts.append(sm)
            if submerged_parts:
                # Combine submerged parts into one mesh for volume and center_mass
                try:
                    submerged = union(submerged_parts, 'blender')
                except Exception:
                    # fallback to first submerged part if union fails
                    submerged = submerged_parts[0]
            else:
                submerged = None

        # Calculate combined center of mass weighted by part mass
        total_mass_for_com = 0.0
        weighted_com = np.zeros(3)
        for part in self.parts:
            part_mass = getattr(part, 'user_mass', None)
            if part_mass is None and hasattr(part, 'density') and part.density is not None:
                vol = part._manual_volume if hasattr(part, '_manual_volume') and part._manual_volume is not None else part.volume
                part_mass = vol * part.density
            if part_mass is None:
                part_mass = 0.0
            weighted_com += part.center_mass * part_mass
            total_mass_for_com += part_mass
        if total_mass_for_com > 0:
            combined_com = weighted_com / total_mass_for_com
        else:
            combined_com = np.array([np.nan, np.nan, np.nan])

        # For compatibility with show(), define combined as a scene of parts
        combined = trimesh.Scene(self.parts)

        relative_waterline = waterline - min_z
        return combined, waterline, relative_waterline, submerged, self._current_cob, combined_com, total_mass

    def check_stability(self):
        """Estimate roll and pitch stability of the floating object."""
        if len(self.parts) == 0:
            raise ValueError("No parts added for stability check")
        
        combined, waterline, relative_waterline, submerged, cob, com, mass = self.solve_equilibrium()
        if submerged is None:
            print("Object is not floating, stability check not applicable.")
            return None, None, None, None

        # Waterplane extents at waterline
        bounds = combined.bounds if hasattr(combined, 'bounds') else None
        if bounds is None:
            # fallback: compute bounds from parts
            min_corner = np.min([p.bounds[0] for p in self.parts], axis=0)
            max_corner = np.max([p.bounds[1] for p in self.parts], axis=0)
            bounds = np.array([min_corner, max_corner])

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
        # Since combined is a Scene, get bounds from parts
        min_corner = np.min([p.bounds[0] for p in self.parts], axis=0)
        max_corner = np.max([p.bounds[1] for p in self.parts], axis=0)
        extents = max_corner - min_corner
        water_depth = extents[2]*5  # depth of the fluid
        plane = trimesh.creation.box(
            extents=[extents[0]*5, extents[1]*5, water_depth]
        )
        center_mass = (min_corner + max_corner) / 2
        plane.apply_translation([
            center_mass[0],
            center_mass[1],
            waterline - water_depth / 2
        ])
        plane.visual.face_colors = [0, 100, 255, 100] 
        scene = trimesh.Scene(list(self.parts) + [plane])
        scene.set_camera(angles=(np.pi/2, 0, 0), distance=extents.max() * 5, center=center_mass )

        # OUTPUT
        print(f"Waterline: {relative_waterline:.3g} m above bottom of object")
        print(f"Total Mass: {mass:.3g} kg")
        # Calculate and print overall density
        total_volume = sum([p._manual_volume if hasattr(p, '_manual_volume') and p._manual_volume is not None else p.volume for p in self.parts])
        overall_density = mass / total_volume if total_volume > 0 else float('nan')
        print(f"Overall Density: {overall_density:.3g} kg/m^3")
        print(f"Submerged Volume: {submerged.volume if submerged is not None else 0:.3g} m^3")
        print(f"Center of Buoyancy: {cob}")
        print(f"Center of Mass: {com}")

        if GM_x and GM_y is not None:
            print(f"Stability Check:")
            print(f"  Roll GM: {GM_x:.3g} m -> {'Stable' if stable_roll else 'Unstable'}")
            print(f"  Pitch GM: {GM_y:.3g} m -> {'Stable' if stable_pitch else 'Unstable'}")
        
        scene.show()

wec = WECModel()
wec.set_fluid_density(1025)  # water kg/m^3
wec.set_gravity(9.81)        # g field strength m/s^2

#Load CAD
wec.load_cad('Data/Keel Fin.stl', 
             scale = 0.001, 
             density = None, 
             mass = 1.9, 
             ignore_holes=False, 
             rotate=False,
             rotations=[{'axis' : 'y', 'angle' : 0}, {'axis' : 'x', 'angle' : 0}]
            )

wec.load_cad('Data/Left Outrigger.stl', 
             scale = 0.001, 
             density = None, 
             mass = 2.4, 
             ignore_holes=False, 
             rotate=False,
             rotations=[{'axis' : 'y', 'angle' : 0}, {'axis' : 'x', 'angle' : 0}]
            )

wec.load_cad('Data/Right Outrigger.stl', 
             scale = 0.001, 
             density = None, 
             mass = 2.4, 
             ignore_holes=False, 
             rotate=False,
             rotations=[{'axis' : 'y', 'angle' : 0}, {'axis' : 'x', 'angle' : 0}]
            )

wec.load_cad('Data/Keel Weight.stl', 
             scale = 0.001, 
             density = None, 
             mass = 4.5, 
             ignore_holes=False, 
             rotate=False,
             rotations=[{'axis' : 'y', 'angle' : 0}, {'axis' : 'x', 'angle' : 0}]
            )

wec.load_cad('Data/Hull.stl', 
             scale = 0.001, 
             density = None, 
             mass = 7.5, #with electronics 
             ignore_holes=True, 
             rotate=False,
             rotations=[{'axis' : 'y', 'angle' : 0}, {'axis' : 'x', 'angle' : 0}],
             manual_volume=0.0314,
            )

# wec.load_cad('Data/WEC 3.stl', 
#              scale = 0.001, 
#              density = None, 
#              mass = 18.7, 
#              ignore_holes=True, 
#              rotate=False,
#              rotations=[{'axis' : 'x', 'angle' : 90}],
#             )

# Check watertightness of all meshes and optionally visualize open edges
wec.check_all_meshes_watertight(visualize=True)

# Show mesh + waterline
wec.show()