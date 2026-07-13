## Buoyancy Calculator v1.0

This is the first public release of the Buoyancy Calculator, an open-source application for analysing
buoyancy and equilibrium of STEP/STL CAD models.

Why did the buoyancy calculator get promoted? Because it always knew how to rise to the occasion!

### Features:
- Calculates volume, centre of mass and centre of buoyancy
- Load multiple STEP/STL files via drag-and-drop or file dialog
- Run equilibrium solver and view waterline and stabillity results in a scrollable output panel
- Set part parameters: scale, mass, density, rotations, manual volume, and center of mass
- Check CAD models for watertightness
- Environment settings: customize fluid density and gravity
- 3D Visualiser: view your loaded STEP/STL CAD models in an interactive 3D viewer.
  - Rotate, pan, and zoom the model to inspect geometry.
  - Works alongside the solver to check and visualise waterline depth. 
  - Optional watertight edge visualization highlights any holes or open facets in the mesh.
- Clear cache and list loaded parts
- Save and load project configurations (JSON)
- Recent configuration quick-access menu

### Known Issues:
- Large load time on application startup due to heavy python dependencies
- Large STEP/STL files may increase load times
- Windows taskbar icon may appear inconsistently due to caching (restart may be required)

### Installation:
- Download the installer `.exe` from this release
- Follow the setup wizard
- Launch and have a play!

### License:
- MIT License – free to use, modify, and distribute

### Images:
<table>
  <tr>
    <td align="centre"><b>Real life Wave Energy Converter </b></td>
    <td align="centre"><b>Simulated WEC from Buoyancy Calculator</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/5358b3af-436e-456f-b58e-7979ef116825" width="400">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/63461eff-d379-49e9-b107-5d123ebb1de7" width="400">
    </td>
  </tr>
</table>
