import sys
import os
import traceback
import json
import io
from contextlib import contextmanager
import ctypes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "Data", "Assets", "history.json")

# Helper for resource paths (PyInstaller-friendly)
def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Third-party imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QLineEdit, QFileDialog, QAbstractItemView, QFormLayout, QDoubleSpinBox,
    QSpinBox, QGroupBox, QGridLayout, QMessageBox, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon

# Local imports
from main import WECModel

class PartsListWidget(QListWidget):
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
    def __init__(self, part_name, defaults):
        super().__init__()
        self.part_name = part_name
        layout = QGridLayout()

        layout.addWidget(QLabel("Scale (0.001 = mm → m):"), 0, 0)
        self.scale = QLineEdit()
        self.scale.setText(str(defaults.get('scale', 0.001)))
        layout.addWidget(self.scale, 0, 1)

        self.use_density_cb = QCheckBox("Use Density Instead of Mass")
        self.use_density_cb.setChecked(defaults.get('density', 0.0) > 0)
        layout.addWidget(self.use_density_cb, 1, 0, 1, 2)

        layout.addWidget(QLabel("Mass (kg):"), 2, 0)
        self.mass = QLineEdit()
        self.mass.setText(str(defaults.get('mass', 0.0)))
        layout.addWidget(self.mass, 2, 1)

        layout.addWidget(QLabel("Density (kg/m³):"), 3, 0)
        self.density = QLineEdit()
        self.density.setText(str(defaults.get('density', 0.0)))
        layout.addWidget(self.density, 3, 1)

        self.apply_rotations_cb = QCheckBox("Apply Rotations")
        self.apply_rotations_cb.setChecked(
            any(defaults.get(k, 0.0) != 0.0 for k in ['rot_x', 'rot_y', 'rot_z'])
        )
        layout.addWidget(self.apply_rotations_cb, 4, 0, 1, 2)

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

        self.override_volume_cb = QCheckBox("Override Volume")
        self.override_volume_cb.setChecked(defaults.get('manual_volume', None) is not None)
        layout.addWidget(self.override_volume_cb, 8, 0, 1, 2)

        self.override_com_cb = QCheckBox("Override COM (Toggle only if computing COM per part)")
        self.override_com_cb.setChecked(defaults.get('manual_com', None) is not None)
        layout.addWidget(self.override_com_cb, 9, 0, 1, 2)

        layout.addWidget(QLabel("Manual Volume (m³):"), 10, 0)
        self.manual_volume = QLineEdit()
        self.manual_volume.setText(
            str(defaults.get('manual_volume', 0.0) if defaults.get('manual_volume', None) is not None else 0.0)
        )
        layout.addWidget(self.manual_volume, 10, 1)

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

        self.use_density_cb.stateChanged.connect(self.update_mass_density_state)
        self.apply_rotations_cb.stateChanged.connect(self.update_rotation_state)
        self.override_volume_cb.stateChanged.connect(self.update_manual_volume_override_state)
        self.override_com_cb.stateChanged.connect(self.update_manual_com_override_state)

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
        self.current_config_file = None  # Track current config file
        icon_path = resource_path(os.path.join("Data", "Assets", "fetch.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # Load history json if exists
        try:
            with open(HISTORY_FILE, "r") as f:
                self.history = json.load(f)
        except Exception:
            self.history = {}

        from PyQt6.QtWidgets import QScrollArea, QSplitter
        central = QWidget()
        # Prepare widgets for left, center, right sections
        # Left: File drag/drop and part list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
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

        # Center: Parameter widgets for selected part
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
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

        # Add: Config file label below Save/Load Config buttons
        self.current_config_label = QLabel("No config loaded")
        self.current_config_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.current_config_label)

        # Right: Solver and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.env_group = QGroupBox("Environment Settings")
        env_layout = QGridLayout()

        try:
            wec_temp = WECModel()
            fluid_density_default = getattr(wec_temp, "fluid_density", 1025)
            gravity_default = getattr(wec_temp, "g", 9.81)
        except Exception:
            fluid_density_default = 1025
            gravity_default = 9.81

        env_layout.addWidget(QLabel("Fluid Density (kg/m³):"), 0, 0)
        self.fluid_density_spin = QDoubleSpinBox()
        self.fluid_density_spin.setRange(0, 2000)
        self.fluid_density_spin.setDecimals(1)
        self.fluid_density_spin.setSingleStep(1)
        self.fluid_density_spin.setValue(fluid_density_default)
        env_layout.addWidget(self.fluid_density_spin, 0, 1)

        env_layout.addWidget(QLabel("Gravity (m/s²):"), 1, 0)
        self.gravity_spin = QDoubleSpinBox()
        self.gravity_spin.setRange(0, 20)
        self.gravity_spin.setDecimals(3)
        self.gravity_spin.setSingleStep(0.01)
        self.gravity_spin.setValue(gravity_default)
        env_layout.addWidget(self.gravity_spin, 1, 1)

        self.env_group.setLayout(env_layout)
        right_layout.addWidget(self.env_group)

        self.watertight_no_vis_btn = QPushButton("Check CAD Models are Watertight")
        self.watertight_no_vis_btn.clicked.connect(self.check_watertight_no_vis)
        right_layout.addWidget(self.watertight_no_vis_btn)
        self.watertight_vis_btn = QPushButton("Show Open Facets")
        self.watertight_vis_btn.clicked.connect(self.check_watertight_vis)
        right_layout.addWidget(self.watertight_vis_btn)

        self.solve_btn = QPushButton("Run Equilibrium Solver")
        self.solve_btn.clicked.connect(self.run_solver)
        right_layout.addWidget(self.solve_btn)

        self.visualize_btn = QPushButton("Visualize in Trimesh Viewer")
        self.visualize_btn.clicked.connect(self.launch_viewer)
        right_layout.addWidget(self.visualize_btn)

        # --- Clear Cache Button ---
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache_gui)
        right_layout.addWidget(self.clear_cache_btn)

        # --- List Loaded Parts Button ---
        self.list_parts_btn = QPushButton("List Loaded Parts")
        self.list_parts_btn.clicked.connect(self.list_loaded_parts_gui)
        right_layout.addWidget(self.list_parts_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.result_group = QGroupBox("Solver Outputs")
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.result_layout = QFormLayout(scroll_content)
        scroll_content.setLayout(self.result_layout)
        scroll_area.setWidget(scroll_content)
        group_layout = QVBoxLayout()
        group_layout.addWidget(scroll_area)
        self.result_group.setLayout(group_layout)
        right_layout.addWidget(self.result_group)

        self.full_screen_output_btn = QPushButton("Full Screen Output")
        self.full_screen_output_btn.clicked.connect(self.show_solver_output_fullscreen)
        right_layout.addWidget(self.full_screen_output_btn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([200, 200, 200])
        splitter.setHandleWidth(6)  # default is 1 or 2
        splitter.setStyleSheet("QSplitter::handle { background-color: gray; }")

        central.setAcceptDrops(True)
        central.dragEnterEvent = self.dragEnterEvent
        central.dropEvent = self.dropEvent
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.wec = WECModel()
        self.wec.set_fluid_density(self.fluid_density_spin.value())
        self.wec.set_gravity(self.gravity_spin.value())
        self.fluid_density_spin.valueChanged.connect(lambda v: self.wec.set_fluid_density(v))
        self.gravity_spin.valueChanged.connect(lambda v: self.wec.set_gravity(v))

        self.parts_list.currentItemChanged.connect(self.update_param_widget)

        self.update_param_widget()

    def save_config(self):
        # Gather parameters
        self.collect_parameters()
        config = {
            "parts": self.parts,
            "environment": {
                "fluid_density": self.fluid_density_spin.value(),
                "gravity": self.gravity_spin.value()
            }
        }
        # If a config file is loaded, prompt to overwrite or save as new
        filename = None
        if self.current_config_file is not None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Save Configuration")
            msg.setText(f"Do you want to overwrite the current config file?\n\n{self.current_config_file}")
            overwrite_btn = msg.addButton("Overwrite", QMessageBox.ButtonRole.YesRole)
            saveas_btn = msg.addButton("Save As New...", QMessageBox.ButtonRole.NoRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(overwrite_btn)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == overwrite_btn:
                filename = self.current_config_file
            elif clicked == saveas_btn:
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
                )
            else:
                return
        else:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
            )
        if not filename:
            return
        try:
            with open(filename, "w") as f:
                json.dump(config, f, indent=2)
            self.current_config_file = filename
            self.update_current_config_label()
            self.save_recent_config(filename)
        except Exception as e:
            QMessageBox.critical(self, "Save Config Error", str(e))

    def load_config(self):
        # Show menu of recent configs and "Other File..." option
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        menu = QMenu(self)
        # Get recent configs from history
        recent_configs = self.history.get("recent_configs", [])
        # Only keep files that still exist
        recent_configs = [f for f in recent_configs if os.path.isfile(f)]
        # Show up to 5 most recent
        shown_configs = recent_configs[:5]
        actions = []
        for path in shown_configs:
            act = QAction(os.path.basename(path), self)
            act.setToolTip(path)
            actions.append((act, path))
            menu.addAction(act)
        if shown_configs:
            menu.addSeparator()
        other_action = QAction("Other File...", self)
        menu.addAction(other_action)
        # Popup menu under the button
        pos = self.load_config_btn.mapToGlobal(self.load_config_btn.rect().bottomLeft())
        chosen_action = menu.exec(pos)
        if chosen_action is None:
            return
        chosen_path = None
        for act, path in actions:
            if chosen_action == act:
                chosen_path = path
                break
        if chosen_action == other_action:
            chosen_path = self.open_config_file_dialog()
        if not chosen_path:
            return
        self.load_config_file(chosen_path)

    def save_recent_config(self, path):
        """Save the config file path to recent configs history."""
        if not path:
            return
        # Load existing recent configs from history
        recent = self.history.get("recent_configs", [])
        # Remove if already present, then insert at front
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        # Keep only 10 most recent
        recent = recent[:10]
        self.history["recent_configs"] = recent
        # Save to disk
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass

    def open_config_file_dialog(self):
        """Show file open dialog for config JSON, return path or None."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        return filename if filename else None

    def load_config_file(self, filename):
        """Load config from the given file, update recent configs and current config label."""
        if not filename:
            return
        # Load JSON and handle both legacy (list of parts) and new (dict with parts & environment)
        try:
            with open(filename, "r") as f:
                config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Config Error", str(e))
            return
        # Determine if config is dict (new format) or list (legacy)
        if isinstance(config, dict):
            loaded_parts = config.get("parts", [])
            environment = config.get("environment", {})
        else:
            loaded_parts = config
            environment = {}
        # Clear current parts, widgets, and list
        self.parts = []
        for w in self.part_widgets.values():
            w.setParent(None)
        self.part_widgets = {}
        self.parts_list.clear()
        # Recreate parts and widgets
        for part in loaded_parts:
            part_filename = part.get('filename')
            params = part.get('params', {})
            self.parts.append({'filename': part_filename, 'params': params})
            item = QListWidgetItem(os.path.basename(part_filename))
            item.setData(Qt.ItemDataRole.UserRole, part_filename)
            self.parts_list.addItem(item)
            self.part_widgets[part_filename] = PartParameterWidget(
                os.path.basename(part_filename), params
            )
        if self.parts:
            self.parts_list.setCurrentRow(0)
        self.update_param_widget()
        # Set environment settings if present
        if environment:
            fluid_density = environment.get("fluid_density")
            gravity = environment.get("gravity")
            if fluid_density is not None:
                self.fluid_density_spin.setValue(fluid_density)
                self.wec.set_fluid_density(fluid_density)
            if gravity is not None:
                self.gravity_spin.setValue(gravity)
                self.wec.set_gravity(gravity)
        # Save to recent configs
        self.save_recent_config(filename)
        # Track current config file and update label
        self.current_config_file = filename
        self.update_current_config_label()

    def update_current_config_label(self):
        """Update the label showing the current loaded config file."""
        if self.current_config_file is None:
            self.current_config_label.setText("No config loaded")
        else:
            self.current_config_label.setText(
                f"Config: {os.path.basename(self.current_config_file)}"
            )

    @staticmethod
    def _collect_supported_files(paths):
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
            wec = self.wec
            if hasattr(wec, "clear"):
                wec.clear()
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
            self.result_layout.clear()
            with self.redirect_stdout_to_gui():
                wec.show_results(output_to_terminal=True)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Solver Error", f"{e}\n{tb}")
        
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
            wec = self.wec
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

    def check_watertight_no_vis(self):
        # Clear previous outputs at the very start and reset WEC model state
        self.result_layout.clear()
        if hasattr(self.wec, "clear"):
            self.wec.clear()
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
            wec = self.wec
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
            # Redirect stdout to GUI and call watertight check
            with self.redirect_stdout_to_gui():
                wec.check_all_meshes_watertight(visualise=False)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Watertight Check Error", f"{e}\n{tb}")

    def check_watertight_vis(self):
        # Clear previous outputs at the very start and reset WEC model state
        self.result_layout.clear()
        if hasattr(self.wec, "clear"):
            self.wec.clear()
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
            wec = self.wec
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
            # Redirect stdout to GUI and call watertight check
            with self.redirect_stdout_to_gui():
                wec.check_all_meshes_watertight(visualise=True)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Watertight Check Error", f"{e}\n{tb}")

    def append_solver_output(self, text):
        label = QLabel(str(text))
        self.result_layout.addRow(label)

    @contextmanager
    def redirect_stdout_to_gui(self):
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        try:
            yield
        finally:
            sys.stdout = old_stdout
            buffer.seek(0)
            for line in buffer.read().splitlines():
                if line.strip():
                    self.append_solver_output(line)
    
    def clear_cache_gui(self):
        try:
            self.wec.clear_cache()
            QMessageBox.information(self, "Cache Cleared", "The cache has been cleared successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Cache Clear Error", f"Failed to clear cache:\n{e}")

    def list_loaded_parts_gui(self):
        self.result_layout.clear()
        if not self.parts:
            self.append_solver_output("No parts are currently loaded.")
            return
        try:
            if not hasattr(self, "wec") or self.wec is None:
                self.wec = WECModel()
            wec = self.wec
            if hasattr(wec, "clear"):
                wec.clear()
            for p in self.parts:
                params = p.get("params", {})
                wec.load_cad(
                    p["filename"],
                    scale=params.get("scale", 0.001),
                    density=params.get("density", 0.0),
                    mass=params.get("mass", 0.0),
                    rotate=params.get("rotate", False),
                    rotations=params.get("rotations", []),
                    manual_volume=params.get("manual_volume", None),
                    manual_com=params.get("manual_com", None)
                )
            import io
            buf = io.StringIO()
            import sys
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                wec.list_parts()
            finally:
                sys.stdout = old_stdout
            output = buf.getvalue().strip()
            if not output:
                self.append_solver_output("No parts are currently loaded.")
            else:
                for line in output.splitlines():
                    if line.strip():
                        self.append_solver_output(line)
        except Exception as e:
            self.append_solver_output(f"Failed to display parts: {e}")
    
    def show_solver_output_fullscreen(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit
        dialog = QDialog(self)
        dialog.setWindowTitle("Solver Outputs - Full Screen")
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        lines = []
        for i in range(self.result_layout.count()):
            item = self.result_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                if widget is not None and hasattr(widget, "text"):
                    lines.append(widget.text())
        text_edit.setPlainText('\n'.join(lines))
        layout.addWidget(text_edit)
        dialog.setLayout(layout)
        try:
            dialog.showMaximized()
        except Exception:
            dialog.showFullScreen()
        dialog.exec()

def _formlayout_clear(self):
    while self.rowCount():
        self.removeRow(0)
if not hasattr(QFormLayout, "clear"):
    QFormLayout.clear = _formlayout_clear

def main():
    if os.name == "nt":
        myappid = "BuoyancyCalculator.Fetch.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)

    # Load and set the icon globally
    icon = QIcon(resource_path(os.path.join("Data", "Assets", "fetch.ico")))
    app.setWindowIcon(icon)

    window = MainWindow()
    window.setWindowIcon(icon)  # also set explicitly on the main window
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()