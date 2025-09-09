from __future__ import annotations

from math import pi, cos, sin, radians
from typing import List, Optional, Tuple, Dict, Any
import argparse
import json

# Physical constants (SI units) - defaults used when not provided in config
DEFAULT_FLUID_DENSITY = 1025.0  # kg/m^3 (seawater default)
DEFAULT_GRAVITY = 9.81          # m/s^2 (standard gravity)

def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> List[List[float]]:
    rx = radians(rx_deg)
    ry = radians(ry_deg)
    rz = radians(rz_deg)
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)
    # R = Rz * Ry * Rx
    Rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    Ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    Rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]

    def matmul(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return matmul(matmul(Rz, Ry), Rx)


def _apply_rotation(R: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z,
        R[1][0] * x + R[1][1] * y + R[1][2] * z,
        R[2][0] * x + R[2][1] * y + R[2][2] * z,
    )


class OrientedShape3D:
    def __init__(self, *, center: Tuple[float, float, float], rotation_deg: Tuple[float, float, float],
                 density: Optional[float], mass_kg: Optional[float], hollow: bool, thickness_m: float,
                 contributes_to_displacement: bool, name: str = "") -> None:
        self.center = center
        self.rotation_deg = rotation_deg
        self.R = _rotation_matrix_xyz(*rotation_deg)
        self.density = density
        self.mass_kg = mass_kg
        self.hollow = hollow
        self.thickness_m = max(0.0, thickness_m)
        self.contributes_to_displacement = contributes_to_displacement
        self.name = name

    # Interface required by voxel engine
    def is_point_inside_outer(self, p_world: Tuple[float, float, float]) -> bool:
        raise NotImplementedError

    def is_point_inside_inner(self, p_world: Tuple[float, float, float]) -> bool:
        return False

    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        raise NotImplementedError

    def effective_density(self) -> Optional[float]:
        if self.mass_kg is not None:
            return None  # mass handled separately
        return self.density


class OrientedCuboid3D(OrientedShape3D):
    def __init__(self, *, length_x: float, width_y: float, height_z: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.length_x = length_x
        self.width_y = width_y
        self.height_z = height_z

    def _to_local(self, p_world: Tuple[float, float, float]) -> Tuple[float, float, float]:
        px, py, pz = p_world
        cx, cy, cz = self.center
        v = (px - cx, py - cy, pz - cz)
        # inverse rotation is transpose for orthonormal R
        Rt = [list(row) for row in zip(*self.R)]
        return _apply_rotation(Rt, v)

    def is_point_inside_outer(self, p_world: Tuple[float, float, float]) -> bool:
        lx, ly, lz = self._to_local(p_world)
        return (abs(lx) <= self.length_x / 2) and (abs(ly) <= self.width_y / 2) and (abs(lz) <= self.height_z / 2)

    def is_point_inside_inner(self, p_world: Tuple[float, float, float]) -> bool:
        if not self.hollow or self.thickness_m <= 0:
            return False
        lx, ly, lz = self._to_local(p_world)
        return (abs(lx) <= max(0.0, self.length_x / 2 - self.thickness_m)) and \
               (abs(ly) <= max(0.0, self.width_y / 2 - self.thickness_m)) and \
               (abs(lz) <= max(0.0, self.height_z / 2 - self.thickness_m))

    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # AABB of OBB: extents = |R| * half_sizes
        hx, hy, hz = self.length_x / 2, self.width_y / 2, self.height_z / 2
        absR = [[abs(a) for a in row] for row in self.R]
        ex = absR[0][0] * hx + absR[0][1] * hy + absR[0][2] * hz
        ey = absR[1][0] * hx + absR[1][1] * hy + absR[1][2] * hz
        ez = absR[2][0] * hx + absR[2][1] * hy + absR[2][2] * hz
        cx, cy, cz = self.center
        return (cx - ex, cy - ey, cz - ez), (cx + ex, cy + ey, cz + ez)


class OrientedCylinder3D(OrientedShape3D):
    def __init__(self, *, radius: float, height_z: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radius = radius
        self.height_z = height_z

    def _to_local(self, p_world: Tuple[float, float, float]) -> Tuple[float, float, float]:
        px, py, pz = p_world
        cx, cy, cz = self.center
        v = (px - cx, py - cy, pz - cz)
        Rt = [list(row) for row in zip(*self.R)]
        return _apply_rotation(Rt, v)

    def is_point_inside_outer(self, p_world: Tuple[float, float, float]) -> bool:
        lx, ly, lz = self._to_local(p_world)
        return (lx * lx + ly * ly) <= (self.radius ** 2) and (abs(lz) <= self.height_z / 2)

    def is_point_inside_inner(self, p_world: Tuple[float, float, float]) -> bool:
        if not self.hollow or self.thickness_m <= 0:
            return False
        inner_r = max(0.0, self.radius - self.thickness_m)
        lx, ly, lz = self._to_local(p_world)
        return (lx * lx + ly * ly) <= (inner_r ** 2) and (abs(lz) <= max(0.0, self.height_z / 2 - self.thickness_m))

    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        # AABB of oriented cylinder approximated by using bounding box of rotated local box of size (2r,2r,h)
        hx, hy, hz = self.radius, self.radius, self.height_z / 2
        absR = [[abs(a) for a in row] for row in self.R]
        ex = absR[0][0] * hx + absR[0][1] * hy + absR[0][2] * hz
        ey = absR[1][0] * hx + absR[1][1] * hy + absR[1][2] * hz
        ez = absR[2][0] * hx + absR[2][1] * hy + absR[2][2] * hz
        cx, cy, cz = self.center
        return (cx - ex, cy - ey, cz - ez), (cx + ex, cy + ey, cz + ez)


class OrientedSphere3D(OrientedShape3D):
    def __init__(self, *, radius: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radius = radius

    def is_point_inside_outer(self, p_world: Tuple[float, float, float]) -> bool:
        x, y, z = p_world
        cx, cy, cz = self.center
        dx, dy, dz = x - cx, y - cy, z - cz
        return (dx * dx + dy * dy + dz * dz) <= (self.radius ** 2)

    def is_point_inside_inner(self, p_world: Tuple[float, float, float]) -> bool:
        if not self.hollow or self.thickness_m <= 0:
            return False
        inner_r = max(0.0, self.radius - self.thickness_m)
        x, y, z = p_world
        cx, cy, cz = self.center
        dx, dy, dz = x - cx, y - cy, z - cz
        return (dx * dx + dy * dy + dz * dz) <= (inner_r ** 2)

    def aabb(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        r = self.radius
        cx, cy, cz = self.center
        return (cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r)


class VoxelBody:
    def __init__(self, shapes: List[OrientedShape3D], voxel_resolution_m: float) -> None:
        self.shapes = shapes
        self.res = max(1e-4, voxel_resolution_m)
        # Compute global AABB
        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]
        for s in self.shapes:
            mn, mx = s.aabb()
            for i in range(3):
                mins[i] = min(mins[i], mn[i])
                maxs[i] = max(maxs[i], mx[i])
        self.aabb_min = tuple(mins)
        self.aabb_max = tuple(maxs)

    # Adapter surface similar to CompositeBody
    def total_volume(self) -> float:
        return self.submerged_volume(self.aabb_max[2] + 10.0)

    def submerged_volume(self, waterline_z: float) -> float:
        x0, y0, z0 = self.aabb_min
        x1, y1, z1 = self.aabb_max
        res = self.res
        vol = 0.0
        cell = res ** 3
        z = z0 + res / 2
        while z <= min(waterline_z, z1 + res / 2):
            y = y0 + res / 2
            while y <= y1 + res / 2:
                x = x0 + res / 2
                while x <= x1 + res / 2:
                    p = (x, y, z)
                    inside = False
                    for s in self.shapes:
                        if not s.contributes_to_displacement:
                            continue
                        if s.is_point_inside_outer(p) and not s.is_point_inside_inner(p):
                            inside = True
                            break
                    if inside:
                        vol += cell
                    x += res
                y += res
            z += res
        return vol

    def max_top_z(self) -> float:
        return self.aabb_max[2]

    def min_base_z(self) -> float:
        return self.aabb_min[2]

    def total_mass_from_parts(self) -> float:
        # Union mass across shapes by voxel sampling
        x0, y0, z0 = self.aabb_min
        x1, y1, z1 = self.aabb_max
        res = self.res
        cell = res ** 3
        mass = 0.0
        z = z0 + res / 2
        while z <= z1 + res / 2:
            y = y0 + res / 2
            while y <= y1 + res / 2:
                x = x0 + res / 2
                while x <= x1 + res / 2:
                    p = (x, y, z)
                    occupied = False
                    density = None
                    for s in self.shapes:
                        if s.is_point_inside_outer(p) and not s.is_point_inside_inner(p):
                            occupied = True
                            if s.mass_kg is not None:
                                density = None
                            else:
                                density = s.density if s.density is not None else density
                            break
                    if occupied:
                        if density is not None:
                            mass += density * cell
                        else:
                            # Mass from explicit mass parts handled separately below
                            mass += 0.0
                    x += res
                y += res
            z += res
        # Add explicit mass parts
        for s in self.shapes:
            if s.mass_kg is not None:
                mass += s.mass_kg
        return mass


def buoyant_force_newtons(
    submerged_volume_m3: float,
    fluid_density_kg_per_m3: float = DEFAULT_FLUID_DENSITY,
    gravity_m_per_s2: float = DEFAULT_GRAVITY,
) -> float:
    """Archimedes' principle: F_b = rho * g * V_displaced."""
    return fluid_density_kg_per_m3 * gravity_m_per_s2 * submerged_volume_m3


def solve_equilibrium_waterline(
    body,
    total_mass_kg: float,
    *,
    fluid_density_kg_per_m3: float = DEFAULT_FLUID_DENSITY,
    gravity_m_per_s2: float = DEFAULT_GRAVITY,
    atol_force_n: float = 1e-3,
    atol_height_m: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """Find waterline z where buoyant force equals weight.

    Returns (waterline_z_m, buoyant_force_N).

    If the object is too heavy to float fully submerged, raises ValueError.
    If the object is so light it floats without touching water, returns the
    minimum base_z as the waterline with zero submerged volume.
    """

    weight_n = total_mass_kg * gravity_m_per_s2

    # Helper: net force at waterline z
    def net_force_at(z: float) -> float:
        return buoyant_force_newtons(body.submerged_volume(z), fluid_density_kg_per_m3, gravity_m_per_s2) - weight_n

    # Establish bracket [lo, hi]
    lo = body.min_base_z() - max(1.0, body.max_top_z() - body.min_base_z())  # some margin below
    hi = body.max_top_z()

    f_lo = net_force_at(lo)
    f_hi = net_force_at(hi)

    # If even at the very bottom there's already enough buoyant force (object mass ~ 0)
    if f_lo >= 0:
        waterline = lo
        fb = net_force_at(waterline) + weight_n  # equals buoyant force
        return waterline, fb

    # If fully submerged at hi and still insufficient buoyancy, it sinks
    if f_hi < 0:
        # Try a bit above to ensure fully-submerged check
        fully_submerged_force = buoyant_force_newtons(body.total_volume(), fluid_density_kg_per_m3, gravity_m_per_s2)
        if fully_submerged_force < weight_n:
            raise ValueError("No equilibrium: body sinks (weight exceeds maximum buoyancy).")
        # Else, expand hi upwards to find sign change
        expand_step = max(0.1, body.max_top_z() - body.min_base_z())
        for _ in range(20):
            hi += expand_step
            f_hi = net_force_at(hi)
            if f_hi >= 0:
                break
        else:
            # Should not happen if fully_submerged_force >= weight
            raise RuntimeError("Failed to bracket root for waterline.")

    # Bisection method
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = net_force_at(mid)
        if abs(f_mid) <= atol_force_n or (hi - lo) <= atol_height_m:
            waterline = mid
            fb = weight_n  # at equilibrium, buoyant force equals weight
            # Use actual computed buoyant force at mid for better reporting
            fb_actual = buoyant_force_newtons(body.submerged_volume(waterline), fluid_density_kg_per_m3, gravity_m_per_s2)
            return waterline, fb_actual
        if f_mid > 0:
            hi = mid
        else:
            lo = mid

    # If not converged, return best effort
    waterline = 0.5 * (lo + hi)
    fb_actual = buoyant_force_newtons(body.submerged_volume(waterline), fluid_density_kg_per_m3, gravity_m_per_s2)
    return waterline, fb_actual


def build_body_from_config(config):
    """Build CompositeBody from JSON config.
    Returns: (body, main_shape_name, fluid_density, gravity)
    """
    # Support multiple key names for convenience
    fluid_density = float(config.get("FLUID_DENSITY", config.get("fluid_density_kg_per_m3", DEFAULT_FLUID_DENSITY)))
    gravity = float(config.get("GRAVITY", config.get("gravity_m_per_s2", DEFAULT_GRAVITY)))
    main_shape_name = config.get("main_shape_name")
    shapes_cfg = config.get("shapes", [])
    electronics_mass_kg = float(config.get("electronics_mass_kg", 0.0))

    parts: List[object] = []
    voxel_shapes: List[OrientedShape3D] = []
    for sc in shapes_cfg:
        stype = sc.get("type")
        name = sc.get("name", "")
        pos = sc.get("position", {})
        base_z = float(pos.get("base_z", 0.0))
        pos_x = float(pos.get("x", 0.0))
        pos_y = float(pos.get("y", 0.0))
        density = sc.get("material_density_kg_per_m3")
        mass = sc.get("mass_kg")
        if density is not None:
            density = float(density)
        if mass is not None:
            mass = float(mass)

        # Common flags for oriented shapes
        rotation = sc.get("rotation_deg", {"x": 0, "y": 0, "z": 0})
        rot_tuple = (float(rotation.get("x", 0.0)), float(rotation.get("y", 0.0)), float(rotation.get("z", 0.0)))
        center = (float(sc.get("center_x", pos_x)), float(sc.get("center_y", pos_y)), float(sc.get("center_z", base_z)))
        mass_only = bool(sc.get("mass_only", False))
        contributes = bool(sc.get("contributes_to_displacement", True)) and (not mass_only)

        if stype == "cuboid":
            dims = sc.get("dimensions", {})
            hollow = bool(sc.get("hollow", False))
            thickness_m = float(sc.get("thickness_m", 0.0))
            voxel_shapes.append(
                OrientedCuboid3D(
                    center=center,
                    rotation_deg=rot_tuple,
                    density=density,
                    mass_kg=mass,
                    hollow=hollow,
                    thickness_m=thickness_m,
                    contributes_to_displacement=contributes,
                    name=name,
                    length_x=float(dims["length_x"]),
                    width_y=float(dims["width_y"]),
                    height_z=float(dims["height_z"]),
                )
            )
        elif stype == "cylinder":
            dims = sc.get("dimensions", {})
            hollow = bool(sc.get("hollow", False))
            thickness_m = float(sc.get("thickness_m", 0.0))
            voxel_shapes.append(
                OrientedCylinder3D(
                    center=center,
                    rotation_deg=rot_tuple,
                    density=density,
                    mass_kg=mass,
                    hollow=hollow,
                    thickness_m=thickness_m,
                    contributes_to_displacement=contributes,
                    name=name,
                    radius=float(dims["radius"]),
                    height_z=float(dims["height_z"]),
                )
            )
        elif stype == "sphere":
            dims = sc.get("dimensions", {})
            hollow = bool(sc.get("hollow", False))
            thickness_m = float(sc.get("thickness_m", 0.0))
            voxel_shapes.append(
                OrientedSphere3D(
                    center=center,
                    rotation_deg=rot_tuple,
                    density=density,
                    mass_kg=mass,
                    hollow=hollow,
                    thickness_m=thickness_m,
                    contributes_to_displacement=contributes,
                    name=name,
                    radius=float(dims["radius"]),
                )
            )
        else:
            raise ValueError(f"Unsupported shape type: {stype}")

    # Decide body modeling approach: voxel union if requested, else analytic sum
    voxel_cfg = config.get("voxel_union", {})
    use_voxel = bool(voxel_cfg.get("enabled", True))
    voxel_res = float(voxel_cfg.get("voxel_resolution_m", 0.01))
    # Always use voxel body for unified behavior
    body = VoxelBody(voxel_shapes, voxel_res)
    # Add electronics mass as a mass-only, non-displacing part
    if electronics_mass_kg > 0:
        # Add as a mass-only oriented tiny cuboid
        voxel_shapes.append(
            OrientedCuboid3D(
                center=(0.0, 0.0, body.min_base_z()),
                rotation_deg=(0.0, 0.0, 0.0),
                density=None,
                mass_kg=electronics_mass_kg,
                hollow=False,
                thickness_m=0.0,
                contributes_to_displacement=False,
                name="electronics",
                length_x=1e-6,
                width_y=1e-6,
                height_z=1e-6,
            )
        )
    return body, main_shape_name, fluid_density, gravity


def run_from_config(path: str) -> None:
    with open(path, "r") as f:
        cfg = json.load(f)
    body, main_name, fluid_density, gravity = build_body_from_config(cfg)
    total_mass = body.total_mass_from_parts()
    if total_mass <= 0:
        raise ValueError("Total mass must be positive. Provide payload_mass_kg or per-part mass/density.")

    try:
        waterline_z, buoyant_force_n = solve_equilibrium_waterline(
            body,
            total_mass,
            fluid_density_kg_per_m3=fluid_density,
            gravity_m_per_s2=gravity,
        )
    except ValueError as e:
        print(str(e))
        return

    print(f"Total mass (WEC): {total_mass:.3f} kg")
    print(f"Equilibrium waterline height (z): {waterline_z:.6f} m")
    print(f"Total buoyant force at equilibrium: {buoyant_force_n:.3f} N")


def main() -> None:
    parser = argparse.ArgumentParser(description="Buoyancy calculator")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()
    if args.config:
        run_from_config(args.config)
    else:
        print("no config file found")

if __name__ == "__main__":
    main()