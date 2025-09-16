import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QLineEdit, QFileDialog, QAbstractItemView, QFormLayout, QDoubleSpinBox,
    QSpinBox, QGroupBox, QGridLayout, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from tqdm import tqdm

import os
import traceback
import json

# Import WECModel from main.py
from main import WECModel

class PartsListWidget(QListWidget):
    """A QListWidget that accepts drag-and-drop for STEP/STL files and calls a callback when files are dropped."""
    def __init__(self, file_loaded_callback):
        super().__init__()
        self.file_loaded_callback = file_loaded_callback
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        files = MainWindow._collect_supported_files(paths)
        if files:
            self.file_loaded_callback(files)
        event.acceptProposedAction()

class DragDropWidget(QWidget):
    """A QWidget that accepts drag-and-drop for STEP/STL files."""
    def __init__(self, file_loaded_callback):
        super().__init__()
        self.file_loaded_callback = file_loaded_callback
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.label = QLabel("Drag and drop STEP or STL files here\nor click to open file dialog", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(
            url.toLocalFile().lower().endswith(('.stl', '.step', '.stp')) or os.path.isdir(url.toLocalFile())
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        files = MainWindow._collect_supported_files(paths)
        if files:
            self.file_loaded_callback(files)
        event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Select STEP/STL files", "",
                "3D Files (*.stl *.step *.stp);;All Files (*)"
            )
            if files:
                self.file_loaded_callback(files)

class PartParameterWidget(QWidget):
    """Widget for displaying and editing parameters for a part."""
    def __init__(self, part_name, defaults):
        super().__init__()
        self.part_name = part_name
        layout = QGridLayout()

        # Scale
        layout.addWidget(QLabel("Scale (0.001 = mm → m):"), 0, 0)
        self.scale = QLineEdit()
        self.scale.setText(str(defaults.get('scale', 0.001)))
        layout.addWidget(self.scale, 0, 1)

        # Mass vs Density toggle
        self.use_density_cb = QCheckBox("Use Density Instead of Mass")
        self.use_density_cb.setChecked(defaults.get('density', 0.0) > 0)
        layout.addWidget(self.use_density_cb, 1, 0, 1, 2)

        # Mass
        layout.addWidget(QLabel("Mass (kg):"), 2, 0)
        self.mass = QLineEdit()
        self.mass.setText(str(defaults.get('mass', 0.0)))
        layout.addWidget(self.mass, 2, 1)

        # Density
        layout.addWidget(QLabel("Density (kg/m³):"), 3, 0)
        self.density = QLineEdit()
        self.density.setText(str(defaults.get('density', 0.0)))
        layout.addWidget(self.density, 3, 1)

        # Rotation toggle
        self.apply_rotations_cb = QCheckBox("Apply Rotations")
        self.apply_rotations_cb.setChecked(
            any(defaults.get(k, 0.0) != 0.0 for k in ['rot_x', 'rot_y', 'rot_z'])
        )
        layout.addWidget(self.apply_rotations_cb, 4, 0, 1, 2)

        # Rotations
        layout.addWidget(QLabel("Rotation X (deg):"), 5, 0)
        self.rot_x = QLineEdit()
        self.rot_x.setText(str(defaults.get('rot_x', 0.0)))
        layout.addWidget(self.rot_x, 5, 1)

        layout.addWidget(QLabel("Rotation Y (deg):"), 6, 0)
        self.rot_y = QLineEdit()
        self.rot_y.setText(str(defaults.get('rot_y', 0.0)))
        layout.addWidget(self.rot_y, 6, 1)

        layout.addWidget(QLabel("Rotation Z (deg):"), 7, 0)
        self.rot_z = QLineEdit()
        self.rot_z.setText(str(defaults.get('rot_z', 0.0)))
        layout.addWidget(self.rot_z, 7, 1)

        # Manual override toggles: separate volume and COM
        self.override_volume_cb = QCheckBox("Override Volume")
        self.override_volume_cb.setChecked(defaults.get('manual_volume', None) is not None)
        layout.addWidget(self.override_volume_cb, 8, 0, 1, 2)

        self.override_com_cb = QCheckBox("Override COM (Only toggle if your calculating the COM for every part, otherwise leave it!)")
        self.override_com_cb.setChecked(defaults.get('manual_com', None) is not None)
        layout.addWidget(self.override_com_cb, 9, 0, 1, 2)

        # Manual volume
        layout.addWidget(QLabel("Manual Volume (m³):"), 10, 0)
        self.manual_volume = QLineEdit()
        self.manual_volume.setText(
            str(defaults.get('manual_volume', 0.0) if defaults.get('manual_volume', None) is not None else 0.0)
        )
        layout.addWidget(self.manual_volume, 10, 1)

        # Manual center of mass (x,y,z)
        layout.addWidget(QLabel("Manual Center of Mass X (m):"), 11, 0)
        self.manual_com_x = QLineEdit()
        mc = defaults.get('manual_com', [0.0, 0.0, 0.0])
        self.manual_com_x.setText(str(mc[0] if mc is not None else 0.0))
        layout.addWidget(self.manual_com_x, 11, 1)

        layout.addWidget(QLabel("Manual Center of Mass Y (m):"), 12, 0)
        self.manual_com_y = QLineEdit()
        self.manual_com_y.setText(str(mc[1] if mc is not None else 0.0))
        layout.addWidget(self.manual_com_y, 12, 1)

        layout.addWidget(QLabel("Manual Center of Mass Z (m):"), 13, 0)
        self.manual_com_z = QLineEdit()
        self.manual_com_z.setText(str(mc[2] if mc is not None else 0.0))
        layout.addWidget(self.manual_com_z, 13, 1)

        self.setLayout(layout)

        # Connect toggles to enable/disable relevant inputs
        self.use_density_cb.stateChanged.connect(self.update_mass_density_state)
        self.apply_rotations_cb.stateChanged.connect(self.update_rotation_state)
        self.override_volume_cb.stateChanged.connect(self.update_manual_volume_override_state)
        self.override_com_cb.stateChanged.connect(self.update_manual_com_override_state)

        # Initialize states
        self.update_mass_density_state()
        self.update_rotation_state()
        self.update_manual_volume_override_state()
        self.update_manual_com_override_state()

    def update_mass_density_state(self):
        use_density = self.use_density_cb.isChecked()
        self.density.setEnabled(use_density)
        self.mass.setEnabled(not use_density)

    def update_rotation_state(self):
        enabled = self.apply_rotations_cb.isChecked()
        self.rot_x.setEnabled(enabled)
        self.rot_y.setEnabled(enabled)
        self.rot_z.setEnabled(enabled)

    def update_manual_volume_override_state(self):
        enabled = self.override_volume_cb.isChecked()
        self.manual_volume.setEnabled(enabled)

    def update_manual_com_override_state(self):
        enabled = self.override_com_cb.isChecked()
        self.manual_com_x.setEnabled(enabled)
        self.manual_com_y.setEnabled(enabled)
        self.manual_com_z.setEnabled(enabled)

    def get_params(self):
        params = {
            'scale': float(self.scale.text()),
            'mass': float(self.mass.text()) if not self.use_density_cb.isChecked() else 0.0,
            'density': float(self.density.text()) if self.use_density_cb.isChecked() else 0.0,
            'rotate': self.apply_rotations_cb.isChecked(),
            'rotations': [],
            'manual_volume': None,
            'manual_com': None
        }
        if params['rotate']:
            params['rotations'] = [
                {'axis': 'x', 'angle': float(self.rot_x.text())},
                {'axis': 'y', 'angle': float(self.rot_y.text())},
                {'axis': 'z', 'angle': float(self.rot_z.text())},
            ]
        if self.override_volume_cb.isChecked():
            params['manual_volume'] = float(self.manual_volume.text())
        if self.override_com_cb.isChecked():
            params['manual_com'] = [
                float(self.manual_com_x.text()),
                float(self.manual_com_y.text()),
                float(self.manual_com_z.text())
            ]
        return params

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Buoyancy Calculator")
        self.resize(900, 600)
        self.parts = []  # List of dicts: {'filename':..., 'volume':..., 'params':..., ...}
        self.part_widgets = {}  # filename -> PartParameterWidget
        self.history = {}

        # Load history json if exists
        try:
            with open("history.json", "r") as f:
                self.history = json.load(f)
        except Exception:
            self.history = {}

        central = QWidget()
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Left: File drag/drop and part list
        left_layout = QVBoxLayout()
        self.dragdrop = DragDropWidget(self.load_files)
        left_layout.addWidget(self.dragdrop)

        self.parts_list = PartsListWidget(self.load_files)
        left_layout.addWidget(QLabel("Loaded Parts:"))
        left_layout.addWidget(self.parts_list, stretch=1)

        # Add: Horizontal layout for Add/Remove File buttons
        btn_layout = QHBoxLayout()
        self.add_file_btn = QPushButton("Add File")
        self.add_file_btn.clicked.connect(self.add_file_dialog)
        btn_layout.addWidget(self.add_file_btn)
        self.remove_file_btn = QPushButton("Remove File")
        self.remove_file_btn.clicked.connect(self.remove_selected_file)
        btn_layout.addWidget(self.remove_file_btn)
        left_layout.addLayout(btn_layout)

        # Enable drag and drop on the central widget so drops anywhere are handled
        central.setAcceptDrops(True)
        central.dragEnterEvent = self.dragEnterEvent
        central.dropEvent = self.dropEvent

        # Center: Parameter widgets for selected part
        center_layout = QVBoxLayout()
        self.param_group = QGroupBox("Part Parameters")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)
        center_layout.addWidget(self.param_group)

        # Add: Save/Load Config buttons
        config_btn_layout = QHBoxLayout()
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.save_config)
        config_btn_layout.addWidget(self.save_config_btn)
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self.load_config)
        config_btn_layout.addWidget(self.load_config_btn)
        center_layout.addLayout(config_btn_layout)

        # Right: Solver and results
        right_layout = QVBoxLayout()
        self.solve_btn = QPushButton("Run Equilibrium Solver")
        self.solve_btn.clicked.connect(self.run_solver)
        right_layout.addWidget(self.solve_btn)

        self.visualize_btn = QPushButton("Visualize in Trimesh Viewer")
        self.visualize_btn.clicked.connect(self.launch_viewer)
        right_layout.addWidget(self.visualize_btn)

        # Progress bar for solver
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.result_group = QGroupBox("Solver Outputs")
        self.result_layout = QFormLayout()
        self.result_group.setLayout(self.result_layout)
        right_layout.addWidget(self.result_group)

        # Add layouts to main
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(center_layout, 2)
        main_layout.addLayout(right_layout, 2)

        # Connections
        self.parts_list.currentItemChanged.connect(self.update_param_widget)

        # Initialize
        self.update_param_widget()

    def save_config(self):
        # Gather parameters
        self.collect_parameters()
        # Open save file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filename:
            return
        # Write self.parts list to JSON
        try:
            with open(filename, "w") as f:
                json.dump(self.parts, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Config Error", str(e))

    def load_config(self):
        # Open open file dialog
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if not filename:
            return
        # Load JSON into list of parts
        try:
            with open(filename, "r") as f:
                loaded_parts = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Config Error", str(e))
            return
        # Clear current parts, widgets, and list
        self.parts = []
        for w in self.part_widgets.values():
            w.setParent(None)
        self.part_widgets = {}
        self.parts_list.clear()
        # Recreate parts and widgets
        for part in loaded_parts:
            filename = part.get('filename')
            params = part.get('params', {})
            self.parts.append({'filename': filename, 'params': params})
            item = QListWidgetItem(os.path.basename(filename))
            item.setData(Qt.ItemDataRole.UserRole, filename)
            self.parts_list.addItem(item)
            self.part_widgets[filename] = PartParameterWidget(
                os.path.basename(filename), params
            )
        if self.parts:
            self.parts_list.setCurrentRow(0)
        self.update_param_widget()

    @staticmethod
    def _collect_supported_files(paths):
        """Given a list of file/folder paths, collect all supported files (.stl, .step, .stp).
        If a folder is given, add all supported files in that folder (non-recursive)."""
        supported_exts = ('.stl', '.step', '.stp')
        files = []
        for path in paths:
            if os.path.isdir(path):
                for fname in os.listdir(path):
                    full_path = os.path.join(path, fname)
                    if os.path.isfile(full_path) and full_path.lower().endswith(supported_exts):
                        files.append(full_path)
            elif os.path.isfile(path) and path.lower().endswith(supported_exts):
                files.append(path)
        return files

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept drag if any file/folder with supported extension or is a directory
        if event.mimeData().hasUrls() and any(
            url.toLocalFile().lower().endswith(('.stl', '.step', '.stp')) or os.path.isdir(url.toLocalFile())
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        files = MainWindow._collect_supported_files(paths)
        if files:
            self.load_files(files)
        event.acceptProposedAction()

    def load_files(self, filepaths):
        for filepath in filepaths:
            if not any(p['filename'] == filepath for p in self.parts):
                # Use parameters from history if available, else default
                defaults = {
                    'scale': 0.001,
                    'mass': 0.0,
                    'density': 0.0,
                    'rot_x': 0.0,
                    'rot_y': 0.0,
                    'rot_z': 0.0,
                    'manual_volume': None,
                    'manual_com': None
                }
                if filepath in self.history:
                    # update defaults with saved params
                    saved = self.history[filepath]
                    for key in defaults.keys():
                        if key in saved:
                            defaults[key] = saved[key]
                part = {
                    'filename': filepath,
                    'params': defaults
                }
                self.parts.append(part)
                item = QListWidgetItem(os.path.basename(filepath))
                item.setData(Qt.ItemDataRole.UserRole, filepath)
                self.parts_list.addItem(item)
                # Create parameter widget
                self.part_widgets[filepath] = PartParameterWidget(
                    os.path.basename(filepath), part['params']
                )
        if self.parts:
            self.parts_list.setCurrentRow(0)
        self.update_param_widget()

    def update_param_widget(self):
        # Clear old widgets
        for i in reversed(range(self.param_layout.count())):
            widget = self.param_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        current_item = self.parts_list.currentItem()
        if not current_item:
            self.param_group.setTitle("Part Parameters")
            return
        filepath = current_item.data(Qt.ItemDataRole.UserRole)
        widget = self.part_widgets[filepath]
        self.param_layout.addWidget(widget)
        self.param_group.setTitle(f"Parameters: {os.path.basename(filepath)}")

    def collect_parameters(self):
        # Gather parameters for all parts from their widgets
        for part in self.parts:
            widget = self.part_widgets[part['filename']]
            part['params'] = widget.get_params()
        return self.parts

    def run_solver(self):
        self.collect_parameters()
        filepaths = [p['filename'] for p in self.parts]
        param_list = []
        for p in self.parts:
            param_list.append((p['filename'],
                               p['params'].get('scale', 0.001),
                               p['params'].get('mass', 0.0),
                               p['params'].get('density', 0.0),
                               p['params'].get('rotate', False),
                               p['params'].get('rotations', []),
                               p['params'].get('manual_volume', None),
                               p['params'].get('manual_com', None)))
        try:
            wec = WECModel()
            self.progress_bar.setMaximum(len(param_list))
            self.progress_bar.setValue(0)
            for i, (filepath, scale, mass, density, rotate, rotations, manual_volume, manual_com) in enumerate(param_list):
                wec.load_cad(
                    filepath,
                    scale=scale,
                    density=density,
                    mass=mass,
                    rotate=rotate,
                    rotations=rotations,
                    manual_volume=manual_volume,
                    manual_com=manual_com
                )
                self.progress_bar.setValue(i+1)
            # Run equilibrium solver
            (relative_waterline, 
             total_mass, 
             overall_density, 
             submerged_volume, 
             cob, com, 
             GM_x, GM_y, 
             stable_roll, stable_pitch
             ) = wec.show_results(output_to_terminal=False)

            # Update GUI display to show results in self.result_layout
            self.result_layout.clear()
            # Waterline
            self.result_layout.addRow(QLabel("Waterline:"), QLabel(f"{relative_waterline:.3g} m above bottom of object"))
            # Total Mass
            self.result_layout.addRow(QLabel("Total Mass:"), QLabel(f"{total_mass:.3g} kg"))
            # Overall Density
            # Compute total volume as in show_results (mass / density)
            if overall_density > 0:
                total_volume = total_mass / overall_density
            else:
                total_volume = 0.0
            self.result_layout.addRow(QLabel("Overall Density:"), QLabel(f"{overall_density:.3g} kg/m^3"))
            # Submerged Volume
            self.result_layout.addRow(QLabel("Submerged Volume:"), QLabel(f"{submerged_volume:.3g} m^3"))
            # Center of Buoyancy
            self.result_layout.addRow(QLabel("Center of Buoyancy:"), QLabel(str(cob)))
            # Center of Mass
            self.result_layout.addRow(QLabel("Center of Mass:"), QLabel(str(com)))
            # Stability Check
            stability_label = QLabel("Stability Check:")
            stability_label.setStyleSheet("font-weight: bold;")
            self.result_layout.addRow(stability_label)
            # Roll GM and stability
            roll_stable_str = "Stable" if stable_roll else "Unstable"
            self.result_layout.addRow(QLabel("  Roll GM:"), QLabel(f"{GM_x:.3g} m -> {roll_stable_str}"))
            # Pitch GM and stability
            pitch_stable_str = "Stable" if stable_pitch else "Unstable"
            self.result_layout.addRow(QLabel("  Pitch GM:"), QLabel(f"{GM_y:.3g} m -> {pitch_stable_str}"))
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Solver Error", f"{e}\n{tb}")
        except:
            pass
        
    def launch_viewer(self):
        self.collect_parameters()
        param_list = []
        for p in self.parts:
            param_list.append((p['filename'],
                               p['params'].get('scale', 0.001),
                               p['params'].get('mass', 0.0),
                               p['params'].get('density', 0.0),
                               p['params'].get('rotate', False),
                               p['params'].get('rotations', []),
                               p['params'].get('manual_volume', None),
                               p['params'].get('manual_com', None)))
        try:
            wec = WECModel()
            for (filepath, scale, mass, density, rotate, rotations, manual_volume, manual_com) in param_list:
                wec.load_cad(
                    filepath,
                    scale=scale,
                    density=density,
                    mass=mass,
                    rotate=rotate,
                    rotations=rotations,
                    manual_volume=manual_volume,
                    manual_com=manual_com
                )
            wec.visualiser()
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Viewer Error", f"{e}\n{tb}")

    def add_file_dialog(self):
        # Open file dialog and load selected files
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select STEP/STL files", "",
            "3D Files (*.stl *.step *.stp);;All Files (*)"
        )
        if files:
            self.load_files(files)

    def remove_selected_file(self):
        # Remove the currently selected file from parts, part_widgets, and parts_list
        current_item = self.parts_list.currentItem()
        if not current_item:
            return
        filepath = current_item.data(Qt.ItemDataRole.UserRole)
        # Remove from self.parts
        self.parts = [p for p in self.parts if p['filename'] != filepath]
        # Remove from self.part_widgets
        if filepath in self.part_widgets:
            widget = self.part_widgets.pop(filepath)
            widget.setParent(None)
        # Remove item from parts_list
        row = self.parts_list.row(current_item)
        self.parts_list.takeItem(row)
        # Update parameter widget display
        self.update_param_widget()

# Add clear method to QFormLayout if missing
def _formlayout_clear(self):
    while self.rowCount():
        self.removeRow(0)
if not hasattr(QFormLayout, "clear"):
    QFormLayout.clear = _formlayout_clear

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()