from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from math import pi
from typing import List, Iterable, Optional, Tuple, Dict, Any
import argparse
import json


# Physical constants (SI units) - defaults used when not provided in config
DEFAULT_FLUID_DENSITY = 1025.0  # kg/m^3 (seawater default)
DEFAULT_GRAVITY = 9.81          # m/s^2 (standard gravity)


def cap(value, min_value, max_value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


@dataclass(kw_only=True)
class Shape(ABC):
    """Base class for buoyancy shapes.

    Coordinate system:
    - z-axis is vertical; waterline is a horizontal plane at z = waterline_z.
    - Each shape has its geometric dimensions and a vertical position given by base_z,
      the z-coordinate of its bottom-most point.

    Extensibility: Subclasses should implement total_volume() and submerged_volume().
    Hollow or composite shapes can override submerged_volume() with custom logic.
    """

    # Position
    base_z: float  # z position of the shape's bottom
    pos_x: float = 0.0
    pos_y: float = 0.0
    name: str = ""
    # Mass properties (optional)
    material_density_kg_per_m3: Optional[float] = None  # if provided, mass can be derived from volume
    mass_kg: Optional[float] = None  # explicit mass overrides derived density
    contributes_to_displacement: bool = True

    @abstractmethod
    def total_volume(self) -> float:
        """Return the total geometric volume of the shape (m^3)."""

    @abstractmethod
    def submerged_volume(self, waterline_z: float) -> float:
        """Return the submerged volume below waterline z (m^3)."""

    @property
    @abstractmethod
    def top_z(self):
        """Return the z position of the shape's top face (m)."""

    def derived_mass_kg(self):
        if self.mass_kg is not None:
            return self.mass_kg
        if self.material_density_kg_per_m3 is not None:
            return self.material_density_kg_per_m3 * self.mass_volume()
        return None

    def mass_volume(self) -> float:
        """Volume used to compute mass if density is provided. Defaults to total volume.
        Override for shells or mass-only parts.
        """
        return self.total_volume()

    def waterline_depth_on_shape(self, waterline_z: float):
        """How far up the shape the waterline reaches (m), capped to shape height. 
        To determine whether the shape is submerged, partially submerged or not submerged at all"""
        return cap(waterline_z - self.base_z, 0.0, max(0.0, self.top_z - self.base_z))


@dataclass(kw_only=True)
class Cuboid(Shape):
    """Axis-aligned rectangular prism aligned with z-axis."""

    length_x: float  # length along x (m)
    width_y: float   # width along y (m)
    height_z: float  # height along z (m)

    def total_volume(self) -> float:
        return self.length_x * self.width_y * self.height_z

    def submerged_volume(self, waterline_z: float) -> float:
        submerged_height = cap(waterline_z - self.base_z, 0.0, self.height_z)
        return self.length_x * self.width_y * submerged_height

    @property
    def top_z(self) -> float:
        return self.base_z + self.height_z


@dataclass(kw_only=True)
class VerticalCylinder(Shape):
    """Right circular cylinder oriented along z-axis."""

    radius: float
    height_z: float

    def total_volume(self) -> float:
        return pi * (self.radius ** 2) * self.height_z

    def submerged_volume(self, waterline_z: float) -> float:
        submerged_height = cap(waterline_z - self.base_z, 0.0, self.height_z)
        return pi * (self.radius ** 2) * submerged_height

    @property
    def top_z(self) -> float:
        return self.base_z + self.height_z


@dataclass(kw_only=True)
class HollowCuboid(Shape):
    """A cuboid shell: outer minus inner cuboid. Displacement can be outer-only if desired.

    By default contributes to displacement with its outer volume but mass uses shell volume.
    Set contributes_to_displacement=False to make it mass-only.
    """

    outer_length_x: float
    outer_width_y: float
    outer_height_z: float
    inner_length_x: float
    inner_width_y: float
    inner_height_z: float

    def total_volume(self) -> float:
        # Displacement volume: outer only (shell encloses water otherwise)
        return self.outer_length_x * self.outer_width_y * self.outer_height_z

    def mass_volume(self) -> float:
        outer = self.outer_length_x * self.outer_width_y * self.outer_height_z
        inner = max(0.0, self.inner_length_x) * max(0.0, self.inner_width_y) * max(0.0, self.inner_height_z)
        inner = min(inner, outer)
        return outer - inner

    def submerged_volume(self, waterline_z: float) -> float:
        submerged_height = cap(waterline_z - self.base_z, 0.0, self.outer_height_z)
        return self.outer_length_x * self.outer_width_y * submerged_height

    @property
    def top_z(self) -> float:
        return self.base_z + self.outer_height_z


@dataclass(kw_only=True)
class HollowVerticalCylinder(Shape):
    """Cylindrical shell: outer radius minus inner radius, vertical."""

    outer_radius: float
    inner_radius: float
    height_z: float

    def total_volume(self) -> float:
        # Displacement from outer cylinder
        return pi * (self.outer_radius ** 2) * self.height_z

    def mass_volume(self) -> float:
        outer = pi * (self.outer_radius ** 2) * self.height_z
        inner = pi * (max(0.0, min(self.inner_radius, self.outer_radius)) ** 2) * self.height_z
        return max(0.0, outer - inner)

    def submerged_volume(self, waterline_z: float) -> float:
        submerged_height = cap(waterline_z - self.base_z, 0.0, self.height_z)
        return pi * (self.outer_radius ** 2) * submerged_height

    @property
    def top_z(self) -> float:
        return self.base_z + self.height_z


@dataclass
class CompositeBody:
    """Collection of shapes acting as a single floating body.

    This supports multiple parts (e.g., hull and outriggers). New shapes
    simply need to implement the Shape API and can be added here.
    """

    parts: List[Shape]

    def total_volume(self) -> float:
        return sum(p.total_volume() for p in self.parts)

    def submerged_volume(self, waterline_z: float) -> float:
        return sum((p.submerged_volume(waterline_z) if p.contributes_to_displacement else 0.0) for p in self.parts)

    def max_top_z(self) -> float:
        return max((p.top_z for p in self.parts), default=0.0)

    def min_base_z(self) -> float:
        return min((p.base_z for p in self.parts), default=0.0)

    def total_mass_from_parts(self) -> float:
        total = 0.0
        for p in self.parts:
            m = p.derived_mass_kg()
            if m is not None:
                total += m
        return total


def buoyant_force_newtons(
    submerged_volume_m3: float,
    fluid_density_kg_per_m3: float = DEFAULT_FLUID_DENSITY,
    gravity_m_per_s2: float = DEFAULT_GRAVITY,
) -> float:
    """Archimedes' principle: F_b = rho * g * V_displaced."""
    return fluid_density_kg_per_m3 * gravity_m_per_s2 * submerged_volume_m3


def solve_equilibrium_waterline(
    body: CompositeBody,
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


def build_body_from_config(config: Dict[str, Any]) -> Tuple[CompositeBody, Optional[str], float, float]:
    """Build CompositeBody from JSON config.

    Returns: (body, main_shape_name, fluid_density, gravity)
    """
    # Support multiple key names for convenience
    fluid_density = float(config.get("FLUID_DENSITY", config.get("fluid_density_kg_per_m3", DEFAULT_FLUID_DENSITY)))
    gravity = float(config.get("GRAVITY", config.get("gravity_m_per_s2", DEFAULT_GRAVITY)))
    main_shape_name = config.get("main_shape_name")
    shapes_cfg = config.get("shapes", [])
    electronics_mass_kg = float(config.get("electronics_mass_kg", 0.0))

    parts: List[Shape] = []
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

        if stype == "cuboid":
            dims = sc.get("dimensions", {})
            part = Cuboid(
                base_z=base_z,
                length_x=float(dims["length_x"]),
                width_y=float(dims["width_y"]),
                height_z=float(dims["height_z"]),
                pos_x=pos_x,
                pos_y=pos_y,
                name=name,
                material_density_kg_per_m3=density,
                mass_kg=mass,
            )
        elif stype == "vertical_cylinder":
            dims = sc.get("dimensions", {})
            part = VerticalCylinder(
                base_z=base_z,
                radius=float(dims["radius"]),
                height_z=float(dims["height_z"]),
                pos_x=pos_x,
                pos_y=pos_y,
                name=name,
                material_density_kg_per_m3=density,
                mass_kg=mass,
            )
        elif stype == "hollow_cuboid":
            dims = sc.get("dimensions", {})
            inner = sc.get("inner_dimensions", {})
            part = HollowCuboid(
                base_z=base_z,
                outer_length_x=float(dims["length_x"]),
                outer_width_y=float(dims["width_y"]),
                outer_height_z=float(dims["height_z"]),
                inner_length_x=float(inner["length_x"]),
                inner_width_y=float(inner["width_y"]),
                inner_height_z=float(inner["height_z"]),
                pos_x=pos_x,
                pos_y=pos_y,
                name=name,
                material_density_kg_per_m3=density,
                mass_kg=mass,
            )
        elif stype == "hollow_vertical_cylinder":
            dims = sc.get("dimensions", {})
            inner = sc.get("inner_dimensions", {})
            part = HollowVerticalCylinder(
                base_z=base_z,
                outer_radius=float(dims["radius"]),
                inner_radius=float(inner["radius"]),
                height_z=float(dims["height_z"]),
                pos_x=pos_x,
                pos_y=pos_y,
                name=name,
                material_density_kg_per_m3=density,
                mass_kg=mass,
            )
        else:
            raise ValueError(f"Unsupported shape type: {stype}")
        parts.append(part)

    body = CompositeBody(parts=parts)
    # Add electronics mass as a mass-only, non-displacing part
    if electronics_mass_kg > 0:
        parts.append(
            Cuboid(
                base_z=body.min_base_z(),
                length_x=0.0,
                width_y=0.0,
                height_z=0.0,
                name="electronics",
                pos_x=0.0,
                pos_y=0.0,
                mass_kg=electronics_mass_kg,
                contributes_to_displacement=False,
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
    if main_name:
        main = next((p for p in body.parts if p.name == main_name), None)
        if main is not None:
            depth = main.waterline_depth_on_shape(waterline_z)
            print(f"Waterline depth on main body '{main_name}': {depth:.6f} m")
        else:
            print(f"Main body '{main_name}' not found in shapes.")
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