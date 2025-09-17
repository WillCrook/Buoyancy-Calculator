import os
import pickle
import hashlib
import trimesh
import numpy as np
from trimesh.boolean import union
from scipy.optimize import brentq
import gmsh
from tqdm import tqdm

class DummySubmerged:
    def __init__(self, volume, center_mass):
        self.volume = volume
        self.center_mass = center_mass


def step_to_trimesh(filepath, mesh_size_factor=0.005):
    """Convert STEP file to trimesh using Gmsh, with caching."""
    file_hash = f"{filepath}_{os.path.getmtime(filepath)}"
    hash_key = hashlib.sha256(file_hash.encode()).hexdigest()
    cache_dir = ".mesh_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{hash_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            mesh = pickle.load(f)
            return mesh

    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", 0)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("step_model")
    gmsh.model.occ.importShapes(filepath)
    gmsh.model.occ.synchronize()

    bbox = gmsh.model.getBoundingBox(-1, -1)
    size = max(bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2])
    char_len = size * mesh_size_factor
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len)

    entities_3d = gmsh.model.getEntities(3)
    exact_volume = 0.0
    weighted_com = np.zeros(3)
    for dim, tag in entities_3d:
        vol = gmsh.model.occ.getMass(dim, tag)
        com = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
        exact_volume += vol
        weighted_com += com * vol
    exact_com = weighted_com / exact_volume if exact_volume > 0 else np.zeros(3)

    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)

    faces = []
    types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    for nt in node_tags:
        faces.extend(np.array(nt).reshape(-1, 3) - 1)

    gmsh.finalize()

    mesh = trimesh.Trimesh(vertices=nodes, faces=np.array(faces))
    mesh._manual_volume = exact_volume
    mesh.center_mass = exact_com
    # Save to cache before returning
    with open(cache_file, "wb") as f:
        pickle.dump(mesh, f)
    return mesh

class WECModel:
    def __init__(self, fluid_density=1000, g=9.81):
        self.fluid_density = fluid_density
        self.g = g
        self.parts = []
    
    def check_all_meshes_watertight(self, visualise=False):
        """Check all loaded parts for watertightness."""
        for part in self.parts:
            if part.is_watertight:
                print(f"'{part.name}' is watertight")
            else:
                open_facets = part.facets_boundary
                print(f"'{part.name}' is NOT watertight")
                print(f"  Number of open facets: {len(open_facets)}")

                if visualise:
                    scene = trimesh.Scene()
                    scene.add_geometry(part)

                    vertices_indices = np.unique(np.concatenate(open_facets))
                  
                    points = trimesh.points.PointCloud(vertices=part.vertices[vertices_indices])
                    points.visual.vertex_colors = [255, 0, 0, 255]  #red
                    scene.add_geometry(points)
                    
                    scene.show(block=False)
        
    def load_cad(self, filepath, scale=None, density=None, mass=None, rotate=False, rotations=None, manual_volume=None, manual_com=None):
        """Load a CAD/mesh file, with caching for STEP files."""
        if filepath.lower().endswith((".step", ".stp")):
            mesh = step_to_trimesh(filepath)
        else:
            mesh = trimesh.load(filepath)
        mesh.apply_scale(scale)
        mesh.name = os.path.basename(filepath)

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
        """Print summary of all loaded parts."""
        if self.parts == []:
            print("No Loaded Parts")
        else:
            print("All loaded WEC parts:")

            for i, part in enumerate(self.parts, start=1):
                print(f"Part {part.name}:")
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
        Uses manual volume if available, otherwise intersects mesh with water box.
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
            water_box = trimesh.creation.box(extents=[size_x*2, size_y*2, size_z])
            water_box.apply_translation([mesh.center_mass[0], mesh.center_mass[1], bounds[0][2] + size_z/2])
            try:
                submerged = mesh.intersection(water_box)
            except Exception:
                return None
            return submerged

    def solve_equilibrium(self):
        import pickle, hashlib
        if len(self.parts) == 0:
            raise ValueError("No parts added")

        # Caching key construction
        cache_items = []
        for part in self.parts:
            filename = getattr(part, "name", None)
            # Try to get absolute path for uniqueness, but fallback to name
            if filename is not None and os.path.isfile(filename):
                abs_path = os.path.abspath(filename)
                mtime = os.path.getmtime(abs_path)
            else:
                abs_path = filename if filename is not None else "nofile"
                mtime = "nofile"
            manual_volume = getattr(part, "_manual_volume", None)
            user_mass = getattr(part, "user_mass", None)
            cache_items.append(f"{abs_path}|{mtime}|{manual_volume}|{user_mass}")
        cache_items.append(f"fluid_density={self.fluid_density}")
        cache_items.append(f"gravity={self.g}")
        cache_key_str = ";".join(str(x) for x in cache_items)
        hash_key = hashlib.sha256(cache_key_str.encode()).hexdigest()
        cache_dir = ".equilibrium_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{hash_key}.pkl")

        # Check for cached result
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

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
                buoyant_message = "The object is too heavy: fully sunk"
                waterline = max_z
                submerged = None
            elif f_min > 0 and f_max > 0:
                buoyant_message = "The object is too light: floats without submerging"
                waterline = min_z
                submerged = None
            else:
                waterline = z_min
                submerged = None
        else:
            waterline = brentq(total_buoyancy, z_min, z_max, xtol=1e-5)
            buoyant_message = "The object floats: it sits in the water"
            submerged_parts = []
            for part in self.parts:
                sm = self.submerged_mesh(part, waterline)
                if sm is not None and sm.volume > 0:
                    submerged_parts.append(sm)
            if submerged_parts:
                try:
                    submerged = union(submerged_parts, 'blender')
                except Exception:
                    submerged = submerged_parts[0]
            else:
                submerged = None

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

        combined = trimesh.Scene(self.parts)

        relative_waterline = waterline - min_z
        result_tuple = (combined, waterline, relative_waterline, submerged, self._current_cob, combined_com, total_mass, buoyant_message)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result_tuple, f)
        except Exception:
            pass
        return result_tuple

    def check_stability(self, combined, waterline, relative_waterline, submerged, cob, com, mass):
        """Estimate roll and pitch stability given equilibrium results."""
        if submerged is None:
            print("Object is not floating, stability check not applicable.")
            return None, None, None, None

        bounds = combined.bounds if hasattr(combined, 'bounds') else None
        if bounds is None:
            min_corner = np.min([p.bounds[0] for p in self.parts], axis=0)
            max_corner = np.max([p.bounds[1] for p in self.parts], axis=0)
            bounds = np.array([min_corner, max_corner])

        width_x = bounds[1][0] - bounds[0][0]
        width_y = bounds[1][1] - bounds[0][1]

        I_x = (width_y**3 * width_x) / 12  # roll
        I_y = (width_x**3 * width_y) / 12  # pitch

        V = submerged.volume

        BM_x = I_x / V
        BM_y = I_y / V

        BG = com[2] - cob[2]

        GM_x = BM_x - BG
        GM_y = BM_y - BG

        stable_roll = GM_x > 0
        stable_pitch = GM_y > 0

        return GM_x, GM_y, stable_roll, stable_pitch

    def visualiser(self):
        """
        Visualise the WEC and waterline using equilibrium calculation.
        """
        combined, waterline, relative_waterline, submerged, cob, com, mass, buoyant_message = self.solve_equilibrium()
        
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
        scene.set_camera(angles=(np.pi/2, 0, 0), distance=extents.max() * 5, center=center_mass)
        scene.show(block=False)

    def show_results(self, output_to_terminal=True):
        """Print equilibrium and stability results."""
        combined, waterline, relative_waterline, submerged, cob, com, mass, buoyant_message = self.solve_equilibrium()
        GM_x, GM_y, stable_roll, stable_pitch = self.check_stability(combined, waterline, relative_waterline, submerged, cob, com, mass)
        total_volume = sum([p._manual_volume if hasattr(p, '_manual_volume') and p._manual_volume is not None else p.volume for p in self.parts])
        overall_density = mass / total_volume if total_volume > 0 else float('nan')

        if output_to_terminal:
            print(buoyant_message)
            print(f"Waterline: {relative_waterline:.3g} m above bottom of object")
            print(f"Total Mass: {mass:.3g} kg")    
            print(f"Overall Density: {overall_density:.3g} kg/m^3")
            print(f"Submerged Volume: {submerged.volume if submerged is not None else 0:.3g} m^3")
            print(f"Center of Buoyancy: {cob}")
            print(f"Center of Mass: {com}")
            if GM_x is not None and GM_y is not None:
                print(f"Stability Check:")
                print(f"  Roll GM: {GM_x:.3g} m -> {'Stable' if stable_roll else 'Unstable'}")
                print(f"  Pitch GM: {GM_y:.3g} m -> {'Stable' if stable_pitch else 'Unstable'}")
        else:
            return relative_waterline, mass, overall_density, submerged.volume, cob, com, GM_x, GM_y, stable_roll, stable_pitch, buoyant_message
    
    def clear(self):
        """Reset the model to its initial state, except for fluid_density and gravity."""
        self.parts = []
        if hasattr(self, 'total_mass'):
            del self.total_mass
        if hasattr(self, 'total_volume'):
            del self.total_volume
        if hasattr(self, '_current_cob'):
            del self._current_cob

    def clear_cache(self):
        """Delete all cached mesh and equilibrium files."""
        import shutil
        for cache_dir in [".mesh_cache", ".equilibrium_cache"]:
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                except Exception as e:
                    print(f"Failed to clear cache directory {cache_dir}: {e}")

if __name__ == "__main__":
    wec = WECModel()
    wec.set_fluid_density(1025)  # water kg/m^3
    wec.set_gravity(9.81)        # g field strength m/s^2

    files = [
        ('Data/WEC STEP 2/Keel Fin.step', 0.001, 1.9, None),
        ('Data/WEC STEP 2/L Outrigger.step', 0.001, 2.4, None),
        ('Data/WEC STEP 2/R Outrigger.step', 0.001, 2.4, None),
        ('Data/WEC STEP 2/Keel Weight.step', 0.001, 4.5, None),
        ('Data/WEC STEP 2/Hull.step', 0.001, 7, 0.03602),  # with electronics
    ]

    for path, scale, mass, manual_volume in tqdm(files, desc="Loading CAD files"):
        wec.load_cad(path,
                     scale=scale,
                     density=None,
                     mass=mass,
                     rotate=False,
                     rotations=[{'axis': 'y', 'angle': 0}, {'axis': 'x', 'angle': 0}, {'axis': 'z', 'angle': 0}],
                     manual_volume=manual_volume
                     )

    wec.check_all_meshes_watertight(visualise=True)

    wec.show_results()
    wec.visualiser()
