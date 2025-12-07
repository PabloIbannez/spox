import numpy as np
import re
import sys
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, colorchooser

# PlotOptiX imports for ray tracing functionality
from plotoptix import TkOptiX
from plotoptix.materials import (
    m_plastic, m_matt_plastic, m_diffuse, m_flat,
    m_shadow_catcher, make_material
)
from plotoptix.enums import MaterialType, Camera


class ParticleViewer:
    """
    A class to visualize particle simulation data using NVIDIA OptiX via plotoptix.
    It provides a Tkinter GUI for real-time interaction, camera control, 
    and rendering settings adjustments.
    """

    def __init__(self, data_file=None, data_string=None):
        """
        Initialize the viewer.
        
        Args:
            data_file (str): Path to the data file to load.
            data_string (str): Raw string content of data to load.
        """
        # --- State Management ---
        self.frames = []
        self.box_sizes = []
        self.current_frame = 0
        self.show_ground_plane = False
        self.show_box = False
        self.screenshot_counter = 0
        self.playback_active = False
        self.playback_after_id = None
        
        # --- Rendering Parameters ---
        self.ambient_level = 0.85
        self.rt = None  # The Ray Tracer instance
        self.surface_z_position = 0.0
        self.light_intensity = 1.0
        self.scene_light_intensity = 1.0
        self.max_accumulation_frames = 256
        self.playback_interval_ms = 500
        self.loop_playback = False

        # --- Camera Parameters ---
        self.aperture_radius = 0.0
        self.focal_scale = 0.95
        self.camera_eye = None
        self.camera_target = None
        self.camera_up_vector = np.array([0, 0, 1])
        self.camera_speed = 2.0
        self.camera_mult = 1.0
        self.mouse_look_active = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.005
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        
        # Camera configuration mapping
        self.camera_names = {
            'dof': "main_cam_dof",
            'pinhole': "main_cam_pinhole",
            'ortho': "main_cam_orthographic",
        }
        self.active_camera_name = None
        self._registered_cameras = set()
        self.camera_projection = "perspective"
        self.projection_labels = {
            "perspective": "Perspective",
            "orthographic": "Orthographic",
        }
        self.projection_from_label = {v: k for k, v in self.projection_labels.items()}
        self.ortho_scale = 60.0
        self.camera_view_presets = ["Free", "Isometric", "Top", "Front", "Side"]
        self.selected_view_preset = "Isometric"

        # --- UI Variables (Tkinter) ---
        self.control_frame = None
        self.camera_speed_var = None
        self.camera_view_var = None
        self.loop_var = None
        self.ray_quality_var = None
        self.surface_color_hex = None
        self.box_color_hex = None
        self.background_color_hex = None

        # --- Styling and Coloring ---
        self.type_colors = {}
        self.type_color_overrides = {}
        self.type_color_displays = {}
        self.particle_material = "cartoon"
        self.surface_color = [0.92, 0.92, 0.92]
        self.box_color = [0.6, 0.6, 0.6]
        self.background_color = [0.98, 0.98, 0.98]
        
        self.style_presets = self._build_style_presets()
        self.current_style_key = "Soft Pastel"
        # Determine max types based on the preset palette size
        self.max_type_count = len(next(iter(self.style_presets.values()))["palette"])
        
        # Apply initial styles
        self.apply_style(self.current_style_key, refresh=False)
        self.surface_color_hex = self._float_rgb_to_hex(self.surface_color)
        self.background_color_hex = self._float_rgb_to_hex(self.background_color)
        self.box_color_hex = self._float_rgb_to_hex(self.box_color)

        # --- Data Loading ---
        if data_file:
            self.load_from_file(data_file)
        elif data_string:
            self.load_from_string(data_string)

    def _build_style_presets(self):
        """
        Defines the available color palettes and material styles for the visualization.
        Returns:
            dict: Dictionary of style configurations.
        """
        pastel = [
            [0.45, 0.65, 0.85], [0.85, 0.45, 0.45], [0.45, 0.85, 0.55],
            [0.85, 0.75, 0.45], [0.75, 0.45, 0.85], [0.45, 0.85, 0.85],
            [0.85, 0.55, 0.45], [0.65, 0.65, 0.65],
        ]
        vibrant = [
            [0.15, 0.45, 1.0], [1.0, 0.2, 0.2], [0.1, 0.8, 0.3],
            [1.0, 0.7, 0.0], [0.8, 0.0, 0.9], [0.0, 0.85, 0.85],
            [1.0, 0.4, 0.1], [0.9, 0.9, 0.9],
        ]
        vivid_detail = [
            [0.05, 0.35, 0.95], [0.95, 0.05, 0.05], [0.0, 0.7, 0.25],
            [0.95, 0.6, 0.0], [0.75, 0.1, 0.9], [0.0, 0.75, 0.75],
            [0.95, 0.3, 0.0], [0.95, 0.95, 0.95],
        ]
        return {
            "Soft Pastel": {
                "palette": pastel,
                "material": "cartoon",
                "description": "Soft tones suited for publication graphics with a matte finish.",
            },
            "Vibrant": {
                "palette": vibrant,
                "material": "cartoon",
                "description": "More intense colors to highlight differences between particle types.",
            },
            "Defined Vibrant": {
                "palette": vivid_detail,
                "material": "vivid_plastic",
                "description": "Saturated colors with a plastic material to emphasize edges.",
            },
        }

    def apply_style(self, style_key, refresh=True):
        """
        Applies a visual style preset (colors and materials).
        """
        preset = self.style_presets.get(style_key)
        if not preset:
            return
        self.type_colors = {i: list(color) for i, color in enumerate(preset["palette"])}
        self._apply_type_overrides()
        self.particle_material = preset["material"]
        self.current_style_key = style_key
        
        # Update UI if initialized
        if hasattr(self, "style_var"):
            if self.style_var.get() != style_key:
                self.style_var.set(style_key)
        if hasattr(self, "style_desc_var"):
            self.style_desc_var.set(preset.get("description", ""))
            
        self._refresh_type_color_controls()
        if refresh and self.rt:
            self.update_particles()

    def _apply_type_overrides(self):
        """Re-applies user-defined color overrides over the preset."""
        for idx, color in self.type_color_overrides.items():
            if idx in self.type_colors:
                self.type_colors[idx] = list(color)

    def _get_scene_reference(self):
        """Calculates the center and bounding distance of the scene for camera positioning."""
        if self.box_sizes:
            idx = min(self.current_frame, len(self.box_sizes) - 1)
            box = self.box_sizes[idx]
        elif self.box_sizes:
            box = self.box_sizes[0]
        else:
            return np.zeros(3), 50.0
            
        Lx, Ly, Lz = box['Lx'], box['Ly'], box['Lz']
        center = np.array([0.0, 0.0, Lz / 4.0])
        distance = max(Lx, Ly, Lz, 20.0)
        return center, distance

    def apply_camera_view_preset(self, preset, refresh=True):
        """Moves the camera to a predefined standard view (Top, Front, Iso, etc.)."""
        if preset not in self.camera_view_presets:
            preset = "Free"
        self.selected_view_preset = preset
        
        if hasattr(self, "camera_view_var") and self.camera_view_var:
            self.camera_view_var.set(preset)
        if preset == "Free":
            return

        center, distance = self._get_scene_reference()
        up = np.array([0, 0, 1])
        span = distance * 1.5

        if preset == "Isometric":
            eye = center + np.array([span, -span, span])
            up = np.array([0, 0, 1])
        elif preset == "Top":
            eye = center + np.array([0, 0, span])
            up = np.array([0, 1, 0])
        elif preset == "Front":
            eye = center + np.array([0, -span, 0])
        elif preset == "Side":
            eye = center + np.array([span, 0, 0])
        else:
            return

        self.camera_eye = eye
        self.camera_target = center
        self.camera_up_vector = up
        
        if refresh:
            self.update_camera()
            self.update_camera_speed_display()

    def has_geometry(self, name):
        """Checks if a named geometry exists in the ray tracer."""
        return (
            hasattr(self, "rt")
            and self.rt is not None
            and hasattr(self.rt, "geometry_data")
            and name in getattr(self.rt, "geometry_data")
        )

    def _float_rgb_to_hex(self, rgb):
        """Converts normalized float RGB [0..1] to Hex string."""
        vals = []
        for c in rgb:
            if c <= 1.0:
                vals.append(max(0, min(255, int(round(c * 255)))))
            else:
                vals.append(max(0, min(255, int(round(c)))))
        return "#{:02x}{:02x}{:02x}".format(*vals)

    def _hex_to_float_rgb(self, hex_color):
        """Converts Hex string to normalized float RGB [0..1]."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return [1.0, 1.0, 1.0]
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return [r, g, b]

    def apply_background_color(self):
        """Updates the background color in the ray tracer."""
        if hasattr(self, "background_color_display"):
            self.background_color_display.config(background=self.background_color_hex)
        if hasattr(self, 'rt') and self.rt is not None:
            try:
                self.rt.set_background(self.background_color)
            except Exception:
                # Fallback for older API versions: calculate mean value
                avg = float(np.mean(self.background_color))
                self.rt.set_background(avg)

    def parse_header(self, header_line):
        """
        Parses the simulation box dimensions from a header line.
        Expected format: '... Lx=10.0;Ly=10.0;Lz=5.0 ...'
        """
        pattern = r'Lx=([0-9.]+);Ly=([0-9.]+);Lz=([0-9.]+)'
        match = re.search(pattern, header_line)
        if match:
            return {
                'Lx': float(match.group(1)),
                'Ly': float(match.group(2)),
                'Lz': float(match.group(3))
            }
        return {'Lx': 50, 'Ly': 50, 'Lz': 12}

    def load_from_string(self, data_string):
        """Loads data from a raw string."""
        lines = data_string.strip().split('\n')
        self._parse_lines(lines)

    def load_from_file(self, filename):
        """Loads data from a file path."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        self._parse_lines(lines)

    def _parse_lines(self, lines):
        """
        Parses lines of particle data into frames and automatically scales down 
        large systems to avoid ray tracing precision artifacts (shadow acne/rings).
        """
        current_particles = []
        current_box = None
        
        # Ensure lists are clear before parsing
        self.frames = []
        self.box_sizes = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                # Header line implies start of a new frame
                if current_particles and current_box:
                    # Save previous frame
                    self.frames.append(np.array(current_particles))
                    self.box_sizes.append(current_box)

                current_box = self.parse_header(line)
                current_particles = []
            else:
                # Particle data: x y z radius type
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        radius = float(parts[3])
                        ptype = int(float(parts[4]))
                        current_particles.append([x, y, z, radius, ptype])
                    except ValueError:
                        continue

        # Save the final frame
        if current_particles and current_box:
            self.frames.append(np.array(current_particles))
            self.box_sizes.append(current_box)

        # --- Auto-scaling Logic ---
        if self.box_sizes:
            # Determine the maximum dimension from the first frame's box
            box0 = self.box_sizes[0]
            max_dim = max(box0['Lx'], box0['Ly'], box0['Lz'])
            
            # Threshold: if the system is larger than 100.0 units, we scale it down.
            # Large coordinates (>1000) cause float precision errors in ray tracing.
            target_max = 100.0
            
            if max_dim > target_max:
                scale_factor = target_max / max_dim
                print(f"Large system detected (Max Dim: {max_dim:.2f}). Scaling by {scale_factor:.6f} to avoid rendering artifacts...")
                
                # Apply scaling to all frames (positions and radii)
                for i in range(len(self.frames)):
                    # Columns 0,1,2 are x,y,z; Column 3 is radius
                    self.frames[i][:, 0:4] *= scale_factor
                
                # Apply scaling to box dimensions
                for i in range(len(self.box_sizes)):
                    self.box_sizes[i]['Lx'] *= scale_factor
                    self.box_sizes[i]['Ly'] *= scale_factor
                    self.box_sizes[i]['Lz'] *= scale_factor
            else:
                print(f"System size is within safe render limits (Max Dim: {max_dim:.2f}). No scaling applied.")

        print(f"Loaded {len(self.frames)} frames")
        for i, frame in enumerate(self.frames):
            print(f"  Frame {i}: {len(frame)} particles")

    def get_particle_data(self, frame_idx):
        """Extracts positions, radii, and colors for a specific frame."""
        if not self.frames or frame_idx >= len(self.frames):
            return None, None, None

        frame = self.frames[frame_idx]
        positions = frame[:, :3]
        radii = frame[:, 3]
        types = frame[:, 4].astype(int)

        palette_size = max(1, len(self.type_colors))
        default_color = [0.5, 0.5, 0.5]
        colors = np.array([
            self.type_colors.get(t % palette_size, default_color)
            for t in types
        ])

        return positions, radii, colors

    def setup_materials(self):
        """Initializes OptiX materials."""
        # Cartoon/Diffuse material
        m_cartoon = make_material(
            MaterialType.Diffuse,
            color=[1.0, 1.0, 1.0, 1.0],
            roughness=0.3,
        )
        self.rt.setup_material("cartoon", m_cartoon)

        # Soft Plastic material
        m_soft_plastic = m_matt_plastic.copy()
        m_soft_plastic["VarFloat"] = {"base_roughness": 0.4}
        self.rt.setup_material("soft_plastic", m_soft_plastic)

        # Vivid Plastic material
        m_vivid_plastic = m_plastic.copy()
        m_vivid_plastic["VarFloat"] = {"base_roughness": 0.15}
        self.rt.setup_material("vivid_plastic", m_vivid_plastic)

        # Shadow Catcher (Ground)
        m_ground = m_shadow_catcher.copy()
        self.rt.setup_material("ground", m_ground)

        # Floor material
        m_floor = make_material(
            MaterialType.Diffuse,
            color=[0.95, 0.95, 0.95, 1.0],
            roughness=0.5,
        )
        self.rt.setup_material("floor", m_floor)

    def setup_scene(self):
        """Configures the initial scene environment, lighting, and camera."""
        if not self.frames:
            print("No frames loaded!")
            return

        # Attempt to set light shading (Hard vs Soft shadows)
        try:
            self.rt.set_param(light_shading="Soft")
        except:
            # Fallback for API variation
            try:
                self.rt.set_light_shading("Soft")
            except:
                pass

        # Calculate initial camera position
        box = self.box_sizes[0]
        Lx, Ly, Lz = box['Lx'], box['Ly'], box['Lz']

        center = [0, 0, Lz / 4]
        distance = max(Lx, Ly) * 1.5

        self.camera_eye = np.array([distance * 0.8, -distance * 0.8, distance * 0.5])
        self.camera_target = np.array(center)
        self.camera_up_vector = np.array([0, 0, 1])
        self.apply_camera_view_preset(self.selected_view_preset, refresh=False)
        self.update_camera()

        self.apply_background_color()

        # Lighting setup
        ambient_intensity = self.ambient_level * self.scene_light_intensity
        self.rt.set_ambient(ambient_intensity)

        # Accumulation (quality) settings
        self.rt.set_param(min_accumulation_step=8)
        self._apply_accumulation_setting()

        # Post-processing
        self.rt.set_float("tonemap_exposure", 0.9)
        self.rt.set_float("tonemap_gamma", 2.2)
        self.rt.add_postproc("Gamma")

    def add_ground_plane(self):
        """Adds or updates the ground plane geometry."""
        if not self.box_sizes:
            return

        box = self.box_sizes[self.current_frame]
        Lx, Ly = box['Lx'], box['Ly']

        self.rt.set_data(
            "ground_plane",
            geom="Parallelograms",
            pos=[[-Lx, -Ly, self.surface_z_position]],
            u=[Lx * 2, 0, 0],
            v=[0, Ly * 2, 0],
            c=self.surface_color,
            mat="floor",
        )

    def add_box_outline(self):
        """Adds or updates the wireframe bounding box."""
        if not self.show_box or not self.box_sizes:
            return

        box = self.box_sizes[self.current_frame]
        Lx, Ly, Lz = box['Lx'], box['Ly'], box['Lz']

        corners = np.array([
            [-Lx, -Ly, -Lz], [ Lx, -Ly, -Lz], [ Lx,  Ly, -Lz], [-Lx,  Ly, -Lz],
            [-Lx, -Ly,  Lz], [ Lx, -Ly,  Lz], [ Lx,  Ly,  Lz], [-Lx,  Ly,  Lz],
        ], dtype=np.float32)

        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],   # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],   # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],   # Vertical pillars
        ], dtype=np.int32)

        radii = np.full(corners.shape[0], 0.1, dtype=np.float32)
        colors = np.full((corners.shape[0], 3), self.box_color, dtype=np.float32)
        edge_indices = edges.flatten()

        self.rt.set_graph(
            "bounding_box",
            pos=corners,
            edges=edge_indices,
            r=radii,
            c=colors,
            mat="diffuse",
        )

    def update_particles(self):
        """Updates the particle geometry in the ray tracer for the current frame."""
        positions, radii, colors = self.get_particle_data(self.current_frame)

        if positions is None:
            return

        # Update main particle data
        self.rt.set_data(
            "particles",
            pos=positions,
            r=radii,
            c=colors,
            mat=self.particle_material,
        )

        if self.show_ground_plane:
            self.add_ground_plane()
        else:
            if self.has_geometry("ground_plane"):
                self.rt.delete_geometry("ground_plane")

        if self.show_box:
            self.add_box_outline()
        else:
            if self.has_geometry("bounding_box"):
                self.rt.delete_geometry("bounding_box")

        if hasattr(self.rt, '_root'):
            self.rt._root.title(f"Particle Viewer - Frame {self.current_frame + 1}/{len(self.frames)}")

    def next_frame(self):
        """Advances to the next frame."""
        if not self.frames:
            return
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            self.update_particles()
            self.update_frame_display()
            print(f"Frame: {self.current_frame + 1}/{len(self.frames)}")
        elif self.loop_playback:
            self.goto_frame(0)

    def prev_frame(self):
        """Returns to the previous frame."""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_particles()
            self.update_frame_display()
            print(f"Frame: {self.current_frame + 1}/{len(self.frames)}")

    def start_playback(self):
        """Starts automated frame playback."""
        if self.playback_active or len(self.frames) <= 1:
            return
        if not hasattr(self, "rt") or self.rt is None or not hasattr(self.rt, "_root"):
            return
        self.playback_active = True
        self._schedule_playback_step()

    def pause_playback(self):
        """Pauses automated frame playback."""
        self.playback_active = False
        if self.playback_after_id and hasattr(self, "rt") and self.rt is not None and hasattr(self.rt, "_root"):
            try:
                self.rt._root.after_cancel(self.playback_after_id)
            except Exception:
                pass
        self.playback_after_id = None

    def _schedule_playback_step(self):
        if not self.playback_active or not hasattr(self, "rt") or self.rt is None:
            return
        if not hasattr(self.rt, "_root"):
            return
        self.playback_after_id = self.rt._root.after(self.playback_interval_ms, self._advance_playback)

    def _advance_playback(self):
        if not self.playback_active:
            return
        previous_frame = self.current_frame
        self.next_frame()
        if len(self.frames) == 0:
            self.pause_playback()
            return
        if (not self.loop_playback and self.current_frame == len(self.frames) - 1
                and previous_frame == self.current_frame):
            self.pause_playback()
            return
        self._schedule_playback_step()

    def goto_frame(self, frame_idx):
        """Jumps to a specific frame index."""
        if 0 <= frame_idx < len(self.frames):
            self.current_frame = frame_idx
            self.update_particles()
            self.update_frame_display()
            print(f"Frame: {self.current_frame + 1}/{len(self.frames)}")

    def save_screenshot(self):
        """Saves the current view to a PNG file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_frame{self.current_frame + 1}_{timestamp}.png"

        # The accumulation needs to settle for a good screenshot, though 
        # this method just grabs the current buffer.
        self.rt.save_image(filename)
        print(f"Screenshot saved: {filename}")
        self.screenshot_counter += 1

    def toggle_ground_plane(self):
        self.show_ground_plane = not self.show_ground_plane
        if self.show_ground_plane:
            self.add_ground_plane()
            print("Ground plane: ON")
        else:
            if self.has_geometry("ground_plane"):
                self.rt.delete_geometry("ground_plane")
            print("Ground plane: OFF")
        self._refresh_type_color_controls()
        self._refresh_surface_controls()

    def set_light_intensity(self, intensity):
        self.light_intensity = intensity
        self.update_lights()

    def set_surface_position(self, z_pos):
        if not self.box_sizes:
            return

        box = self.box_sizes[self.current_frame]
        Lz = box['Lz']

        # Clamp Z to box dimensions
        self.surface_z_position = np.clip(z_pos, -Lz, Lz)

        if self.show_ground_plane:
            self.add_ground_plane()
        if self.show_box:
            self.add_box_outline()

        print(f"Surface position: {self.surface_z_position:.2f}")

    def update_lights(self):
        if not self.rt:
            return
        ambient_intensity = self.ambient_level * self.scene_light_intensity
        try:
            self.rt.set_ambient(ambient_intensity, refresh=True)
        except Exception as e:
            print(f"Warning: Could not update ambient lighting: {e}")

    def _apply_accumulation_setting(self):
        if not self.rt:
            return
        try:
            self.rt.set_param(max_accumulation_frames=int(self.max_accumulation_frames))
        except Exception as exc:
            print(f"Warning: Could not update accumulation frames: {exc}")

    def setup_key_bindings(self):
        """Binds keyboard and mouse events to handlers."""
        root = self.rt._root

        def on_key_press(event):
            key = event.keysym.lower()

            if key == 'right':
                self.next_frame()
            elif key == 'left':
                self.prev_frame()
            elif key == 'c':
                self.save_screenshot()
            elif key == 'p':
                self.toggle_ground_plane()
            elif key == 'g':
                self.show_box = not self.show_box
                if hasattr(self, 'box_var'):
                    self.box_var.set(self.show_box)
                self._refresh_box_controls()
                self.update_particles()
            elif key == 'escape':
                self.safe_exit()
            elif key == 'home':
                self.goto_frame(0)
            elif key == 'end':
                self.goto_frame(len(self.frames) - 1)
            elif key == 'w':
                self.move_camera_simple('forward')
            elif key == 's':
                self.move_camera_simple('backward')
            elif key == 'a':
                self.move_camera_simple('left')
            elif key == 'd':
                self.move_camera_simple('right')
            elif key == 'q':
                self.roll_camera('left')
            elif key == 'e':
                self.roll_camera('right')
            elif key == 'r':
                self.reset_camera_simple()
            elif key == 'shift_l':
                self.move_camera_simple('up')
            elif key == 'control_l':
                self.move_camera_simple('down')
        
        root.bind('<KeyPress>', on_key_press)
        root.bind('<F12>', lambda e: self.save_screenshot())

        def on_mouse_wheel(event):
            # Determine if the widget that triggered the event is part of the control panel
            is_over_panel = False
            w = event.widget
            while w:
                if w == self.control_container:
                    is_over_panel = True
                    break
                w = w.master

            if is_over_panel:
                # Handle vertical scrolling for the panel
                if event.num == 5 or event.delta < 0:
                    self.control_panel_canvas.yview_scroll(1, "units")
                elif event.num == 4 or event.delta > 0:
                    self.control_panel_canvas.yview_scroll(-1, "units")
            else:
                # Handle camera zoom
                if event.num == 4 or event.delta > 0:
                    self.zoom_camera(zoom_in=True)
                elif event.num == 5 or event.delta < 0:
                    self.zoom_camera(zoom_in=False)

        def on_mouse_motion(event):
            if self.mouse_look_active:
                dx = event.x - self.last_mouse_x
                dy = event.y - self.last_mouse_y
                self.camera_yaw = -dx * self.mouse_sensitivity
                self.camera_pitch = -dy * self.mouse_sensitivity
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

        def on_mouse_button_press(event):
            root.after_idle(self.sync_camera_state)

        def on_mouse_button_release(event):
            root.after_idle(self.sync_camera_state)

        def on_mouse_drag(event):
            root.after_idle(self.sync_camera_state)

        root.bind('<MouseWheel>', on_mouse_wheel)
        root.bind('<Motion>', on_mouse_motion)
        root.bind('<Button-1>', on_mouse_button_press)
        root.bind('<Button-3>', on_mouse_button_press)
        root.bind('<ButtonRelease-1>', on_mouse_button_release)
        root.bind('<ButtonRelease-3>', on_mouse_button_release)
        root.bind('<B1-Motion>', on_mouse_drag)
        root.bind('<B3-Motion>', on_mouse_drag)

        # Unified scroll bindings for Linux/X11
        root.bind('<Button-4>', on_mouse_wheel)
        root.bind('<Button-5>', on_mouse_wheel)

    def create_control_panel(self):
        """Constructs the Tkinter control panel."""
        if not self.rt or not hasattr(self.rt, '_root'):
            return

        root = self.rt._root

        # Layout Setup
        self.control_container = ttk.Frame(root, padding=0, relief="flat", borderwidth=0)
        self.control_panel_canvas = tk.Canvas(self.control_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.control_container, orient="vertical", command=self.control_panel_canvas.yview)
        self.control_panel_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_frame = ttk.Frame(self.control_panel_canvas, padding=0, relief="flat", borderwidth=0)
        self.control_panel_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")

        def on_control_configure(event):
            self.control_panel_canvas.configure(scrollregion=self.control_panel_canvas.bbox("all"))
        self.control_frame.bind("<Configure>", on_control_configure)

        def position_control_panel():
            try:
                root.update_idletasks()
                self.control_container.place(relx=1.0, rely=0.5, anchor="e", relheight=0.9, width=340)
            except Exception:
                self.control_container.grid(row=0, column=1, sticky="nse", padx=0, pady=0)
                root.grid_columnconfigure(0, weight=1)
                root.grid_columnconfigure(1, weight=0)
                root.grid_rowconfigure(0, weight=1)

        root.after(100, position_control_panel)

        # --- Widgets ---
        ttk.Label(self.control_frame, text="Controls", font=("Arial", 12, "bold")).pack(pady=(80, 10))

        # Camera Section
        camera_frame = ttk.LabelFrame(self.control_frame, text="Camera", padding=10)
        camera_frame.pack(fill=tk.X, pady=(0, 10))

        proj_frame = ttk.Frame(camera_frame)
        proj_frame.pack(fill=tk.X, pady=2)
        ttk.Label(proj_frame, text="Projection:").pack(side=tk.LEFT)
        self.camera_projection_var = tk.StringVar(value=self.projection_labels[self.camera_projection])
        projection_combo = ttk.Combobox(
            proj_frame,
            textvariable=self.camera_projection_var,
            values=list(self.projection_labels.values()),
            state="readonly",
            width=14,
        )
        projection_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        projection_combo.bind("<<ComboboxSelected>>", self.on_projection_change)

        view_frame = ttk.Frame(camera_frame)
        view_frame.pack(fill=tk.X, pady=2)
        ttk.Label(view_frame, text="View:").pack(side=tk.LEFT)
        self.camera_view_var = tk.StringVar(value=self.selected_view_preset)
        view_combo = ttk.Combobox(
            view_frame,
            textvariable=self.camera_view_var,
            values=self.camera_view_presets,
            state="readonly",
            width=14,
        )
        view_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        view_combo.bind("<<ComboboxSelected>>", self.on_view_preset_change)

        self.camera_speed_label = ttk.Label(camera_frame, text=f"Speed: {self.camera_mult:.2f}")
        self.camera_speed_label.pack(anchor=tk.W, pady=(0, 2))
        self.camera_speed_var = tk.DoubleVar(value=self.camera_mult)
        speed_slider_frame = ttk.Frame(camera_frame)
        speed_slider_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(
            speed_slider_frame,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.camera_speed_var,
            command=self.on_camera_speed_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_speed_entry = ttk.Entry(speed_slider_frame, width=6, textvariable=self.camera_speed_var)
        self.camera_speed_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.camera_speed_entry.bind('<Return>', self.on_camera_speed_entry_change)
        self.camera_speed_entry.bind('<FocusOut>', self.on_camera_speed_entry_change)

        blur_frame = ttk.Frame(camera_frame)
        blur_frame.pack(fill=tk.X, pady=2)
        ttk.Label(blur_frame, text="Blur:").pack(side=tk.LEFT)

        self.blur_var = tk.DoubleVar(value=self.aperture_radius)
        blur_slider_frame = ttk.Frame(blur_frame)
        blur_slider_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Scale(
            blur_slider_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.blur_var,
            command=self.on_blur_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.blur_entry = ttk.Entry(blur_slider_frame, width=6, textvariable=self.blur_var)
        self.blur_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.blur_entry.bind('<Return>', self.on_blur_entry_change)
        self.blur_entry.bind('<FocusOut>', self.on_blur_entry_change)

        focus_frame = ttk.Frame(camera_frame)
        focus_frame.pack(fill=tk.X, pady=2)
        ttk.Label(focus_frame, text="Focus:").pack(side=tk.LEFT)

        self.focus_var = tk.DoubleVar(value=self.focal_scale)
        focus_slider_frame = ttk.Frame(focus_frame)
        focus_slider_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Scale(
            focus_slider_frame,
            from_=0.5,
            to=1.5,
            orient=tk.HORIZONTAL,
            variable=self.focus_var,
            command=self.on_focus_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.focus_entry = ttk.Entry(focus_slider_frame, width=6, textvariable=self.focus_var)
        self.focus_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.focus_entry.bind('<Return>', self.on_focus_entry_change)
        self.focus_entry.bind('<FocusOut>', self.on_focus_entry_change)

        # Lighting Section
        lights_frame = ttk.LabelFrame(self.control_frame, text="Scene Lighting", padding=10)
        lights_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(lights_frame, text="Light Intensity:").pack(anchor=tk.W)
        self.scene_light_var = tk.DoubleVar(value=self.scene_light_intensity)
        scene_slider_frame = ttk.Frame(lights_frame)
        scene_slider_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(
            scene_slider_frame,
            from_=0.1,
            to=3.0,
            orient=tk.HORIZONTAL,
            variable=self.scene_light_var,
            command=self.on_scene_light_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scene_light_entry = ttk.Entry(scene_slider_frame, width=6, textvariable=self.scene_light_var)
        self.scene_light_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.scene_light_entry.bind('<Return>', self.on_scene_light_entry_change)
        self.scene_light_entry.bind('<FocusOut>', self.on_scene_light_entry_change)

        ttk.Label(lights_frame, text="Ambient Level:").pack(anchor=tk.W, pady=(6, 0))
        self.ambient_var = tk.DoubleVar(value=self.ambient_level)
        ambient_slider_frame = ttk.Frame(lights_frame)
        ambient_slider_frame.pack(fill=tk.X, pady=2)
        ttk.Scale(
            ambient_slider_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.ambient_var,
            command=self.on_ambient_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ambient_entry = ttk.Entry(ambient_slider_frame, width=6, textvariable=self.ambient_var)
        self.ambient_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.ambient_entry.bind('<Return>', self.on_ambient_entry_change)
        self.ambient_entry.bind('<FocusOut>', self.on_ambient_entry_change)

        # Quality/Ray Tracing Section
        ray_frame = ttk.LabelFrame(self.control_frame, text="Ray tracing quality", padding=10)
        ray_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(ray_frame, text="Maximum accumulations:").pack(anchor=tk.W)
        self.ray_quality_var = tk.IntVar(value=self.max_accumulation_frames)
        ray_slider_row = ttk.Frame(ray_frame)
        ray_slider_row.pack(fill=tk.X, pady=2)
        ttk.Scale(
            ray_slider_row,
            from_=32,
            to=512,
            orient=tk.HORIZONTAL,
            variable=self.ray_quality_var,
            command=self.on_ray_quality_change
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ray_quality_entry = ttk.Entry(ray_slider_row, width=6, textvariable=self.ray_quality_var)
        self.ray_quality_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.ray_quality_entry.bind('<Return>', self.on_ray_quality_entry_change)
        self.ray_quality_entry.bind('<FocusOut>', self.on_ray_quality_entry_change)
        ttk.Button(ray_frame, text="Relaunch rays", command=self.on_rerun_rays).pack(fill=tk.X, pady=(6, 0))

        # Visual Style Section
        style_frame = ttk.LabelFrame(self.control_frame, text="Visual Style", padding=10)
        style_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(style_frame, text="Style selector:").pack(anchor=tk.W)
        self.style_var = tk.StringVar(value=self.current_style_key)
        style_combo = ttk.Combobox(
            style_frame,
            textvariable=self.style_var,
            values=list(self.style_presets.keys()),
            state="readonly",
        )
        style_combo.pack(fill=tk.X, pady=2)
        style_combo.bind("<<ComboboxSelected>>", self.on_style_combo_change)
        self.style_desc_var = tk.StringVar(value=self.style_presets[self.current_style_key].get("description", ""))
        ttk.Label(style_frame, textvariable=self.style_desc_var, wraplength=300, font=("Arial", 8)).pack(anchor=tk.W, pady=(4, 0))
        
        bg_row = ttk.Frame(style_frame)
        bg_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(bg_row, text="Background color:").pack(side=tk.LEFT)
        self.background_color_display = tk.Label(bg_row, width=4, background=self.background_color_hex, relief="groove")
        self.background_color_display.pack(side=tk.LEFT, padx=4)
        ttk.Button(bg_row, text="Change", command=self.on_background_color_pick).pack(side=tk.RIGHT)

        # Colors by Type Section
        type_frame = ttk.LabelFrame(self.control_frame, text="Colors by type", padding=10)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        self.type_color_displays = {}
        for idx in range(self.max_type_count):
            row = ttk.Frame(type_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"Type {idx}:").pack(side=tk.LEFT)
            color = self.type_colors.get(idx, [0.5, 0.5, 0.5])
            hex_color = self._float_rgb_to_hex(color)
            label = tk.Label(row, width=4, background=hex_color, relief="groove")
            label.pack(side=tk.LEFT, padx=4)
            ttk.Button(row, text="Change", command=lambda i=idx: self.on_type_color_pick(i)).pack(side=tk.RIGHT)
            self.type_color_displays[idx] = label

        # Navigation Section
        frame_frame = ttk.LabelFrame(self.control_frame, text="Frame Navigation", padding=10)
        frame_frame.pack(fill=tk.X, pady=(0, 10))

        nav_frame = ttk.Frame(frame_frame)
        nav_frame.pack(fill=tk.X)
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(anchor=tk.CENTER)
        ttk.Button(nav_buttons, text="<<", command=lambda: self.goto_frame(0), width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons, text="<", command=self.prev_frame, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons, text=">", command=self.next_frame, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons, text=">>", command=lambda: self.goto_frame(len(self.frames)-1), width=4).pack(side=tk.LEFT, padx=2)

        playback_frame = ttk.Frame(frame_frame)
        playback_frame.pack(anchor=tk.CENTER, pady=(6, 0))
        ttk.Button(playback_frame, text="Play", command=self.start_playback, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Button(playback_frame, text="Pause", command=self.pause_playback, width=6).pack(side=tk.LEFT, padx=4)

        self.loop_var = tk.BooleanVar(value=self.loop_playback)
        ttk.Checkbutton(frame_frame, text="Loop", variable=self.loop_var,
                        command=self.on_loop_toggle).pack(anchor=tk.CENTER, pady=(6, 0))

        self.frame_label = ttk.Label(frame_frame, text=f"Frame: {self.current_frame+1}/{len(self.frames)}")
        self.frame_label.pack(anchor=tk.CENTER, pady=5)

        ttk.Button(self.control_frame, text="Save Screenshot", command=self.save_screenshot).pack(fill=tk.X, pady=5)

        # Box Options
        box_frame = ttk.LabelFrame(self.control_frame, text="Show Box", padding=10)
        box_frame.pack(fill=tk.X, pady=(0, 10))
        self.box_var = tk.BooleanVar(value=self.show_box)
        ttk.Checkbutton(box_frame, text="Show box", variable=self.box_var,
                        command=self.on_box_toggle).pack(anchor=tk.W)
        box_color_row = ttk.Frame(box_frame)
        box_color_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(box_color_row, text="Color:").pack(side=tk.LEFT)
        self.box_color_display = tk.Label(box_color_row, width=4, background=self.box_color_hex, relief="groove")
        self.box_color_display.pack(side=tk.LEFT, padx=4)
        ttk.Button(box_color_row, text="Choose...", command=self.on_box_color_pick).pack(side=tk.RIGHT)

        # Surface Options
        surface_frame = ttk.LabelFrame(self.control_frame, text="Surface Position (Z)", padding=10)
        surface_frame.pack(fill=tk.X, pady=(0, 10))

        if self.box_sizes:
            Lz = self.box_sizes[self.current_frame]['Lz']
        else:
            Lz = 12

        ttk.Label(surface_frame, text="Z Position:").pack(anchor=tk.W)
        ttk.Label(surface_frame, text=f"Range: {-Lz:.1f} to {Lz:.1f}", font=("Arial", 8)).pack(anchor=tk.W)
        self.surface_var = tk.DoubleVar(value=self.surface_z_position)
        surface_slider_frame = ttk.Frame(surface_frame)
        surface_slider_frame.pack(fill=tk.X, pady=2)
        surface_slider = ttk.Scale(
            surface_slider_frame,
            from_=-Lz,
            to=Lz,
            orient=tk.HORIZONTAL,
            variable=self.surface_var,
            command=self.on_surface_change
        )
        surface_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.surface_entry = ttk.Entry(surface_slider_frame, width=7, textvariable=self.surface_var)
        self.surface_entry.pack(side=tk.RIGHT, padx=(6, 0))
        self.surface_entry.bind('<Return>', self.on_surface_entry_change)
        self.surface_entry.bind('<FocusOut>', self.on_surface_entry_change)

        color_row = ttk.Frame(surface_frame)
        color_row.pack(fill=tk.X, pady=2)
        ttk.Label(color_row, text="Color:").pack(side=tk.LEFT)
        self.surface_color_display = tk.Label(color_row, width=4, background=self.surface_color_hex, relief="groove")
        self.surface_color_display.pack(side=tk.LEFT, padx=4)
        ttk.Button(color_row, text="Choose...", command=self.on_surface_color_pick).pack(side=tk.RIGHT)

        self.surface_toggle_btn = ttk.Button(surface_frame, text=self._surface_button_text(), command=self.toggle_ground_plane)
        self.surface_toggle_btn.pack(fill=tk.X, pady=(4, 0))

        self._refresh_box_controls()
        self._refresh_surface_controls()

        ttk.Button(self.control_frame, text="Exit", command=self.safe_exit).pack(fill=tk.X, pady=(10, 5))

    def on_scene_light_change(self, value):
        try:
            intensity = float(value)
        except (TypeError, ValueError):
            return
        intensity = float(np.clip(intensity, 0.1, 3.0))
        self.scene_light_intensity = intensity
        if hasattr(self, "scene_light_var"):
            current = self.scene_light_var.get()
            if abs(current - intensity) > 1e-6:
                self.scene_light_var.set(intensity)
        self.update_lights()

    def on_blur_change(self, value):
        try:
            blur = float(value)
        except (TypeError, ValueError):
            return
        blur = float(np.clip(blur, 0.0, 1.0))
        self.aperture_radius = blur
        if hasattr(self, "blur_var"):
            current = self.blur_var.get()
            if abs(current - blur) > 1e-6:
                self.blur_var.set(blur)
        self.update_camera()

    def on_focus_change(self, value):
        try:
            focus = float(value)
        except (TypeError, ValueError):
            return
        focus = float(np.clip(focus, 0.5, 1.5))
        self.focal_scale = focus
        if hasattr(self, "focus_var"):
            current = self.focus_var.get()
            if abs(current - focus) > 1e-6:
                self.focus_var.set(focus)
        self.update_camera()

    def on_surface_change(self, value):
        z_pos = float(value)
        self.set_surface_position(z_pos)

    def on_surface_entry_change(self, event):
        try:
            if self.box_sizes:
                Lz = self.box_sizes[self.current_frame]['Lz']
                z_pos = float(self.surface_entry.get())
                z_pos = np.clip(z_pos, -Lz, Lz)
                self.surface_var.set(z_pos)
                self.set_surface_position(z_pos)
        except ValueError:
            self.surface_var.set(self.surface_z_position)

    def on_surface_color_pick(self):
        initial = self.surface_color_hex or self._float_rgb_to_hex(self.surface_color)
        color = colorchooser.askcolor(color=initial, title="Select surface color")
        if color and color[1]:
            rgb = color[0]
            self.surface_color = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]
            self.surface_color_hex = color[1]
            if self.show_ground_plane:
                self.add_ground_plane()
            self._refresh_surface_controls()

    def on_box_color_pick(self):
        initial = self.box_color_hex or self._float_rgb_to_hex(self.box_color)
        color = colorchooser.askcolor(color=initial, title="Select box color")
        if color and color[1]:
            rgb = color[0]
            self.box_color = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]
            self.box_color_hex = color[1]
            if self.show_box:
                self.add_box_outline()
            self._refresh_box_controls()

    def on_background_color_pick(self):
        initial = self.background_color_hex or self._float_rgb_to_hex(self.background_color)
        color = colorchooser.askcolor(color=initial, title="Select background color")
        if color and color[1]:
            rgb = color[0]
            self.background_color = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]
            self.background_color_hex = color[1]
            self.apply_background_color()

    def on_view_preset_change(self, event=None):
        if not hasattr(self, "camera_view_var"):
            return
        preset = self.camera_view_var.get()
        self.apply_camera_view_preset(preset)

    def on_projection_change(self, event=None):
        label = self.camera_projection_var.get()
        new_mode = self.projection_from_label.get(label, "perspective")
        if new_mode != self.camera_projection:
            self.camera_projection = new_mode
            self.active_camera_name = None
            self.update_camera()

    def on_type_color_pick(self, type_idx):
        base_color = self.type_colors.get(type_idx, [0.5, 0.5, 0.5])
        initial = self._float_rgb_to_hex(base_color)
        color = colorchooser.askcolor(color=initial, title=f"Color for type {type_idx}")
        if color and color[1]:
            rgb = color[0]
            new_color = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]
            self.type_color_overrides[type_idx] = new_color
            self._apply_type_overrides()
            self._refresh_type_color_controls()
            self.update_particles()

    def _surface_button_text(self):
        return "Show surface" if not self.show_ground_plane else "Hide surface"

    def _refresh_surface_controls(self):
        if hasattr(self, "surface_toggle_btn"):
            self.surface_toggle_btn.config(text=self._surface_button_text())
        if hasattr(self, "surface_color_display") and self.surface_color_hex:
            self.surface_color_display.config(background=self.surface_color_hex)

    def _refresh_box_controls(self):
        if hasattr(self, "box_color_display") and self.box_color_hex:
            self.box_color_display.config(background=self.box_color_hex)
        if hasattr(self, "box_var"):
            self.box_var.set(self.show_box)

    def _refresh_type_color_controls(self):
        if not self.type_color_displays:
            return
        for idx, label in self.type_color_displays.items():
            color = self.type_colors.get(idx, [0.5, 0.5, 0.5])
            label.config(background=self._float_rgb_to_hex(color))

    def on_style_combo_change(self, event=None):
        selected = self.style_var.get()
        self.apply_style(selected)

    def _handle_slider_entry_change(self, entry_widget, variable, min_val, max_val, callback, fallback_value):
        try:
            value = float(entry_widget.get())
        except ValueError:
            variable.set(fallback_value)
            return

        value = float(np.clip(value, min_val, max_val))
        variable.set(value)
        callback(value)

    def on_scene_light_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.scene_light_entry,
            self.scene_light_var,
            0.1,
            3.0,
            self.on_scene_light_change,
            self.scene_light_intensity
        )

    def on_camera_speed_change(self, value):
        try:
            speed = float(value)
        except (TypeError, ValueError):
            return
        speed = float(np.clip(speed, 0.1, 5.0))
        self.camera_mult = speed
        self.update_camera_speed_display()

    def on_camera_speed_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.camera_speed_entry,
            self.camera_speed_var,
            0.1,
            5.0,
            self.on_camera_speed_change,
            self.camera_mult
        )

    def on_ambient_change(self, value):
        try:
            ambient = float(value)
        except (TypeError, ValueError):
            return
        ambient = float(np.clip(ambient, 0.1, 1.0))
        self.ambient_level = ambient
        if hasattr(self, "ambient_var"):
            current = self.ambient_var.get()
            if abs(current - ambient) > 1e-6:
                self.ambient_var.set(ambient)
        self.update_lights()

    def on_ambient_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.ambient_entry,
            self.ambient_var,
            0.1,
            1.0,
            self.on_ambient_change,
            self.ambient_level
        )

    def on_ray_quality_change(self, value):
        try:
            frames = int(float(value))
        except (TypeError, ValueError):
            return
        frames = int(np.clip(frames, 32, 512))
        self.max_accumulation_frames = frames
        if hasattr(self, "ray_quality_var"):
            current = int(self.ray_quality_var.get())
            if current != frames:
                self.ray_quality_var.set(frames)
        self._apply_accumulation_setting()

    def on_ray_quality_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.ray_quality_entry,
            self.ray_quality_var,
            32,
            512,
            self.on_ray_quality_change,
            self.max_accumulation_frames
        )

    def on_rerun_rays(self):
        if not hasattr(self, "rt") or self.rt is None:
            return
        try:
            self.rt.refresh_scene()
        except Exception as exc:
            print(f"Warning: Could not refresh scene: {exc}")

    def on_blur_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.blur_entry,
            self.blur_var,
            0.0,
            1.0,
            self.on_blur_change,
            self.aperture_radius
        )

    def on_focus_entry_change(self, event=None):
        self._handle_slider_entry_change(
            self.focus_entry,
            self.focus_var,
            0.5,
            1.5,
            self.on_focus_change,
            self.focal_scale
        )

    def on_box_toggle(self):
        self.show_box = self.box_var.get()
        if self.show_box:
            self.add_box_outline()
        else:
            if self.has_geometry("bounding_box"):
                self.rt.delete_geometry("bounding_box")
        self._refresh_box_controls()

    def on_loop_toggle(self):
        if hasattr(self, "loop_var"):
            self.loop_playback = bool(self.loop_var.get())

    def update_frame_display(self):
        if hasattr(self, 'frame_label'):
            self.frame_label.config(text=f"Frame: {self.current_frame+1}/{len(self.frames)}")

    def init_camera_if_needed(self):
        """Sets default camera values if none exist."""
        if self.camera_eye is None:
            if self.box_sizes:
                box = self.box_sizes[0]
                Lx, Ly, Lz = box['Lx'], box['Ly'], box['Lz']
                distance = max(Lx, Ly) * 1.5
                self.camera_eye = np.array([distance * 0.8, -distance * 0.8, distance * 0.5])
                self.camera_target = np.array([0, 0, Lz / 4])
            else:
                self.camera_eye = np.array([60.0, -60.0, 60.0])
                self.camera_target = np.array([0, 0, 0])

    def sync_camera_state(self):
        """Synchronizes internal camera state with the Ray Tracer's current state."""
        if not hasattr(self, 'rt') or self.rt is None:
            return

        try:
            cam_name = self.get_current_camera_name()
            # Retrieve latest positions from engine
            eye = self.rt.get_camera_eye(cam_name)
            target = self.rt.get_camera_target(cam_name)

            if eye is not None:
                self.camera_eye = np.array(eye)

            if target is not None:
                self.camera_target = np.array(target)

            # Try to get up-vector
            try:
                camera_params = self.rt.get_camera(cam_name)
                if camera_params and isinstance(camera_params, dict):
                    if 'up' in camera_params:
                        self.camera_up_vector = np.array(camera_params['up'])
            except:
                pass

        except AttributeError:
            # Fallback for older versions
            self._sync_camera_state_fallback()
        except Exception as e:
            print(f"Camera sync failed: {e}, using stored camera state")

    def _sync_camera_state_fallback(self):
        """Fallback method for camera synchronization if standard API fails."""
        try:
            if hasattr(self.rt, '_camera') and self.rt._camera is not None:
                if hasattr(self.rt._camera, 'eye'):
                    self.camera_eye = np.array(self.rt._camera.eye)
                if hasattr(self.rt._camera, 'target'):
                    self.camera_target = np.array(self.rt._camera.target)
                if hasattr(self.rt._camera, 'up'):
                    self.camera_up_vector = np.array(self.rt._camera.up)

            elif hasattr(self.rt, '_optix_data'):
                # Direct access to internal data structure (risky but effective)
                optix_data = self.rt._optix_data
                cam_name = self.get_current_camera_name()
                if 'camera' in optix_data and cam_name in optix_data['camera']:
                    cam_data = optix_data['camera'][cam_name]
                    if 'eye' in cam_data:
                        self.camera_eye = np.array(cam_data['eye'])
                    if 'target' in cam_data:
                        self.camera_target = np.array(cam_data['target'])
                    if 'up' in cam_data:
                        self.camera_up_vector = np.array(cam_data['up'])

        except Exception as e:
            print(f"Fallback camera sync failed: {e}")

    def get_current_camera_name(self):
        if self.active_camera_name:
            return self.active_camera_name
        if hasattr(self, 'rt') and self.rt is not None:
            try:
                name = self.rt.get_current_camera_name()
                if name:
                    self.active_camera_name = name
                    return name
            except Exception:
                pass
        
        if self.camera_projection == "orthographic":
            return self.camera_names['ortho']
        return self.camera_names['dof']

    def ensure_camera(self, name, cam_type, make_current=False):
        """Ensures the camera exists in the engine; creates it if not."""
        if name in self._registered_cameras:
            return
        self.init_camera_if_needed()
        fov_value = self.ortho_scale if cam_type == Camera.Ortho else 45
        camera_kwargs = dict(
            eye=self.camera_eye.tolist(),
            target=self.camera_target.tolist(),
            up=self.camera_up_vector.tolist(),
            fov=fov_value,
            make_current=make_current,
        )
        if cam_type == Camera.DoF:
            camera_kwargs.update(
                focal_scale=self.focal_scale,
                aperture_radius=max(self.aperture_radius, 0.0),
            )
        elif cam_type == Camera.Pinhole:
            camera_kwargs.update(
                focal_scale=self.focal_scale,
                aperture_radius=0.0,
            )
        else:
            camera_kwargs.update(
                focal_scale=1.0,
                aperture_radius=0.0,
            )
        self.rt.setup_camera(name, cam_type=cam_type, **camera_kwargs)
        self._registered_cameras.add(name)
        if make_current:
            self.active_camera_name = name

    def reset_camera_simple(self):
        """Resets the camera to the default position relative to the scene center."""
        center, distance = self._get_scene_reference()
        if np.linalg.norm(center) == 0 and distance == 50.0:
            self.camera_eye = np.array([60.0, -60.0, 60.0])
            self.camera_target = np.array([0, 0, 0])
        else:
            self.camera_target = center
            self.camera_eye = center + np.array([distance, -distance, distance])
        self.camera_mult = 1.0
        self.camera_up_vector = np.array([0, 0, 1])
        self.apply_camera_view_preset(self.selected_view_preset, refresh=False)
        self.update_camera()
        self.update_camera_speed_display()

    def move_camera_simple(self, direction):
        """Calculates camera vectors and applies movement."""
        if not hasattr(self, 'rt') or self.rt is None:
            return

        self.selected_view_preset = "Free"
        if hasattr(self, "camera_view_var") and self.camera_view_var:
            self.camera_view_var.set("Free")

        self.init_camera_if_needed()
        # Fetch current state to ensure relative movement is accurate
        self.sync_camera_state()

        # Calculate forward vector
        forward = self.camera_target - self.camera_eye
        forward = forward / np.linalg.norm(forward)
        
        # Calculate right vector
        right = np.cross(forward, self.camera_up_vector)
        if np.linalg.norm(right) > 0:
            right = right / np.linalg.norm(right)
        else:
            # Handle gimbal lock case (looking straight up/down)
            right = np.cross(forward, [0, 0, 1])
            if np.linalg.norm(right) > 0:
                right = right / np.linalg.norm(right)
            else:
                right = np.array([1, 0, 0])

        # Calculate local up vector
        up = np.cross(right, forward)
        if np.linalg.norm(up) > 0:
            up = up / np.linalg.norm(up)
        else:
            up = self.camera_up_vector

        speed = self.camera_speed * self.camera_mult

        # Handle orthographic zoom via movement keys
        if self.camera_projection == "orthographic" and direction in ('forward', 'backward'):
            zoom_step = max(0.5, speed * 0.2)
            if direction == 'forward':
                self.ortho_scale = max(1.0, self.ortho_scale - zoom_step)
            else:
                self.ortho_scale = min(1000.0, self.ortho_scale + zoom_step)
            self.update_camera()
            return

        if direction == 'forward':
            self.camera_eye += forward * speed
            self.camera_target += forward * speed
        elif direction == 'backward':
            self.camera_eye -= forward * speed
            self.camera_target -= forward * speed
        elif direction == 'left':
            self.camera_eye -= right * speed
            self.camera_target -= right * speed
        elif direction == 'right':
            self.camera_eye += right * speed
            self.camera_target += right * speed
        elif direction == 'up':
            self.camera_eye += up * speed
            self.camera_target += up * speed
        elif direction == 'down':
            self.camera_eye -= up * speed
            self.camera_target -= up * speed

        self.update_camera()

    def zoom_camera(self, zoom_in=True):
        """Handles camera zoom logic for both perspective and orthographic modes."""
        if not hasattr(self, 'rt') or self.rt is None:
            return
        self.selected_view_preset = "Free"
        if hasattr(self, "camera_view_var") and self.camera_view_var:
            self.camera_view_var.set("Free")

        self.init_camera_if_needed()
        self.sync_camera_state()

        forward = self.camera_target - self.camera_eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm == 0:
            return
        forward = forward / forward_norm
        speed = self.camera_speed * self.camera_mult * 0.5
        speed = max(0.05, speed)

        if self.camera_projection == "orthographic":
            zoom_step = max(0.5, speed * 0.2)
            if zoom_in:
                self.ortho_scale = max(1.0, self.ortho_scale - zoom_step)
            else:
                self.ortho_scale = min(1000.0, self.ortho_scale + zoom_step)
            self.update_camera()
            return

        delta = forward * speed
        if zoom_in:
            self.camera_eye += delta
            self.camera_target += delta
        else:
            self.camera_eye -= delta
            self.camera_target -= delta

        self.update_camera()

    def roll_camera(self, direction):
        """Rolls the camera around its forward axis."""
        if not hasattr(self, 'rt') or self.rt is None:
            return

        self.selected_view_preset = "Free"
        if hasattr(self, "camera_view_var") and self.camera_view_var:
            self.camera_view_var.set("Free")

        self.init_camera_if_needed()
        self.sync_camera_state()

        # Calculate vectors
        forward = self.camera_target - self.camera_eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.camera_up_vector)
        if np.linalg.norm(right) > 0:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1, 0, 0])

        # Rotation angle
        roll_angle = np.pi / 18
        if direction == 'left':
            roll_angle = -roll_angle

        # Apply rotation matrix logic to the UP vector
        cos_r, sin_r = np.cos(roll_angle), np.sin(roll_angle)
        self.camera_up_vector = cos_r * self.camera_up_vector + sin_r * right
        if np.linalg.norm(self.camera_up_vector) > 0:
            self.camera_up_vector = self.camera_up_vector / np.linalg.norm(self.camera_up_vector)

        self.update_camera()

    def update_camera_speed_display(self):
        if hasattr(self, 'camera_speed_label'):
            self.camera_speed_label.config(text=f"Speed: {self.camera_mult:.2f}")
        if hasattr(self, "camera_speed_var") and self.camera_speed_var is not None:
            current = self.camera_speed_var.get()
            if abs(current - self.camera_mult) > 1e-6:
                self.camera_speed_var.set(self.camera_mult)

    def update_camera(self):
        """Applies internal camera state to the rendering engine."""
        if not hasattr(self, 'rt') or self.rt is None:
            return

        self.init_camera_if_needed()
        if self.camera_eye is None or self.camera_target is None:
            return

        # Determine which camera type to use
        if self.camera_projection == "orthographic":
            use_dof = False
            cam_key = 'ortho'
            cam_type = Camera.Ortho
        else:
            use_dof = self.aperture_radius > 1e-4
            cam_key = 'dof' if use_dof else 'pinhole'
            cam_type = Camera.DoF if use_dof else Camera.Pinhole
        cam_name = self.camera_names[cam_key]

        make_current = self.active_camera_name is None
        self.ensure_camera(cam_name, cam_type, make_current=make_current)

        fov_value = self.ortho_scale if cam_type == Camera.Ortho else 45
        update_kwargs = dict(
            eye=self.camera_eye.tolist(),
            target=self.camera_target.tolist(),
            up=self.camera_up_vector.tolist(),
            fov=fov_value,
        )
        
        # Apply Depth of Field settings only if relevant
        if cam_type == Camera.Ortho:
            update_kwargs.update(focal_scale=1.0, aperture_radius=0.0)
        elif use_dof:
            update_kwargs.update(focal_scale=self.focal_scale, aperture_radius=self.aperture_radius)
        elif cam_type == Camera.Pinhole:
            update_kwargs.update(focal_scale=self.focal_scale, aperture_radius=0.0)

        # Update or Setup
        try:
            self.rt.update_camera(cam_name, **update_kwargs)
        except Exception:
            self.rt.setup_camera(cam_name, cam_type=cam_type, **update_kwargs, make_current=False)

        # Switch active camera if changed
        if self.active_camera_name != cam_name:
            try:
                self.rt.set_current_camera(cam_name)
            except Exception as exc:
                print(f"Warning: Could not switch camera: {exc}")
            else:
                self.active_camera_name = cam_name

    def run(self):
        """Starts the main application loop."""
        if not self.frames:
            print("No data loaded. Please load data first.")
            return

        # Initialize the Ray Tracer window
        self.rt = TkOptiX(
            width=1280,
            height=720,
        )

        # Attempt to set light shading with fallbacks
        try:
            self.rt.set_param(light_shading="Hard")
        except:
            try:
                self.rt.setup_light("dummy_light", light_type="Environment")
                self.rt.delete_light("dummy_light")
            except:
                try:
                    self.rt.set_background(0.0)
                    self.rt.set_param(light_shading="Hard")
                except:
                    print("Warning: Could not set light shading - will try alternative approach")

        self.setup_materials()
        self.setup_scene()
        self.update_particles()

        if self.show_ground_plane:
            self.add_ground_plane()

        self.rt.start()

        import atexit
        import signal
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.setup_key_bindings()
        self.create_control_panel()

        print("\n" + "="*50)
        print("PARTICLE VIEWER CONTROLS")
        print("="*50)
        print("Left/Right Arrow : Previous/Next frame")
        print("Home/End         : First/Last frame")
        print("C or F12         : Save screenshot")
        print("P                : Toggle ground plane")
        print("G                : Toggle bounding box")
        print("W/A/S/D          : Move camera forward/left/backward/right")
        print("Q/E              : Roll camera left/right")
        print("L.Shift/L.Ctrl   : Move camera up/down")
        print("R                : Reset camera")
        print("Mouse wheel      : Zoom in/out")
        print("Escape           : Exit")
        print("="*50 + "\n")

    def cleanup(self):
        """Cleanup hook for application exit."""
        pass

    def signal_handler(self, signum, frame):
        """Handles system interrupt signals."""
        self.safe_exit()

    def safe_exit(self):
        """Safely shuts down the renderer and exits."""
        try:
            if hasattr(self, 'rt') and self.rt is not None and hasattr(self.rt, '_root'):
                self.rt._root.quit()
        except:
            pass
        os._exit(0)

def main():
    print("="*50)
    print("SPOX: SuperPunto format renderer based on plotOptiX.")
    print("="*50)

    viewer = None
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        if os.path.exists(data_file):
            print(f"Loading data from: {data_file}")
            viewer = ParticleViewer(data_file=data_file)
        else:
            print(f"File not found: {data_file}")
    else:
        print("No file provided.")
    
    if viewer:
        viewer.run()

if __name__ == "__main__":
    main()
