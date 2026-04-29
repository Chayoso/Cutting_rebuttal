"""
Rendering utilities for the MLS‑MPM cutting simulation.
- Sets up Taichi UI (window, camera, lighting)
- Renders particles, knife/board proxies, optional grid, and overlay HUD
"""
import numpy as np
import taichi as ti

def should_force_full_recolor(prev_z_off: float, new_z_off: float, threshold: float = 0.01) -> bool:
    """Return True when |Δz| is large enough to warrant a full recolor pass."""
    return abs(float(new_z_off) - float(prev_z_off)) >= float(threshold)

@ti.data_oriented
class MPMRenderer:
    """Renderer class for MPM cutting simulation."""
    def __init__(self, sim):
        self.sim = sim
        self._board_proxy_cache = None
        self._grid_proxy_cache = None
        # Camera state tracking for smooth transitions
        self._camera_current_eye = None
        self._camera_current_lookat = None
        self._camera_current_up = None

    def _pick_scene(self, window):
        if hasattr(window, "get_scene"):
            return window.get_scene()
        return ti.ui.Scene()

    def _init_viewer(self):
        self.sim.window = ti.ui.Window("MLS‑MPM Cutting", res=(1280, 800), vsync=False)
        self.sim.canvas = self.sim.window.get_canvas()
        self.sim.scene  = self._pick_scene(self.sim.window)
        try:
            self.sim.camera = self.sim.window.get_camera()
        except Exception:
            self.sim.camera = ti.ui.Camera()

    def _apply_initial_camera(self):
        mode = self.sim.viewer_camera_mode
        camera_presets = {
            "top":   ([0.0, 0.7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            "front": ([1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 1.0, 0.0]),
            "auto":  ([0.4, 0.3, 0.4], [0.0, 0.1, 0.0], [0.0, 1.0, 0.0])
        }
        if mode in camera_presets:
            eye, look, up = camera_presets[mode]
        else:
            pose = self.sim.viewer_initial_pose
            eye  = pose.get("eye",    [0.3, 0.22, 0.3])
            look = pose.get("lookat", [0.0, 0.1, 0.0])
            up   = pose.get("up",     [0.0, 1.0, 0.0])
        try:
            self.sim.camera.position(*map(float, eye))
            self.sim.camera.lookat(*map(float, look))
            self.sim.camera.up(*map(float, up))
            self.sim.camera.fov(float(self.sim.viewer_initial_pose.get("fov", 35.0)))
        except Exception:
            pass

    @ti.func
    def _get_cutting_color(self, cut_intensity: ti.f32):
        """Yellow → Orange → Red → Dark Red."""
        red = 1.0
        green = ti.max(0.0, 1.0 - cut_intensity)
        blue = ti.max(0.0, 0.2 * cut_intensity - 0.1)
        return ti.Vector([red, green, blue])

    @ti.kernel
    def _recolor_full_kernel(self, band_eps: ti.f32, hide_damage_band: ti.i32):
        for p in range(self.sim.pcount[None]):
            part = self.sim.particles[p]
            X = part.x
            damage = part.D
            
            # During pause, hide damage band gate (red coloring near blade)
            if hide_damage_band == 1:
                # Only show yellow particles (no red damage band gate)
                self.sim.p_color[p] = ti.Vector([1.0, 1.0, 0.0])
            else:
                # Normal rendering: Priority: If damage exists, always show red color (permanent cutting surface)
                if damage > 1e-6:
                    self.sim.p_color[p] = self._get_cutting_color(damage)
                elif self.sim.knife.is_blade(X) == 1 and ti.abs(self.sim.knife.sample(X)) < band_eps:
                    # Near blade but not damaged yet - show distance-based color (damage band gate)
                    nd = ti.abs(self.sim.knife.sample(X)) / band_eps
                    base_intensity = 1.0 - nd
                    self.sim.p_color[p] = self._get_cutting_color(base_intensity)
                else:
                    # Default yellow color
                    self.sim.p_color[p] = ti.Vector([1.0, 1.0, 0.0])

    def _update_cut_colors(self, band_eps: float):
        # Hide damage band gate during pause
        hide_damage_band = 1 if self.sim._transparency_pause_active else 0
        self._recolor_full_kernel(float(band_eps), hide_damage_band)
        self.sim._force_full_recolor = 0

    def _setup_scene_lighting(self):
        self.sim.scene.ambient_light((0.45, 0.45, 0.45))
        self.sim.scene.point_light(pos=(0.6, 0.8, 0.6), color=(1, 1, 1))

    def _handle_camera_input(self):
        if not self.sim.viewer_lock_on_run:
            self.sim.camera.track_user_inputs(self.sim.window, movement_speed=0.02, hold_key=ti.ui.RMB)

    def _render_particles(self):
        radius = self.sim.particle_render_radius * self.sim.viewer_radius_scale
        
        # During pause, hide particles with Z > knife_z AND hide red particles (damaged + damage band gate)
        if self.sim._transparency_pause_active:
            knife_z_off = float(self.sim.knife.z_off[None])
            # Get particle positions and damage as numpy array
            particle_positions = self.sim.particles.x.to_numpy()
            particle_damage = self.sim.particles.D.to_numpy()
            
            # Update colors first (with damage band gate hidden)
            if self.sim.p_color is not None and self.sim.damage_visualization:
                self._update_cut_colors(float(self.sim.damage_band_f[None]))
            
            # Filter: keep only particles with Z <= knife_z AND damage == 0 (no red particles, no damage band gate)
            mask = (particle_positions[:, 2] <= knife_z_off) & (particle_damage < 1e-6)
            filtered_positions = particle_positions[mask]
            
            if self.sim.p_color is not None and self.sim.damage_visualization:
                filtered_colors = self.sim.p_color.to_numpy()[mask]
                if len(filtered_positions) > 0:
                    self.sim.scene.particles(filtered_positions, radius=radius, per_vertex_color=filtered_colors)
            else:
                if len(filtered_positions) > 0:
                    self.sim.scene.particles(filtered_positions, radius=radius, color=(1.0, 1.0, 0.0))
        else:
            # Normal rendering: show all particles
            if self.sim.p_color is not None and self.sim.damage_visualization:
                # Update colors every frame during transparency pause, or at regular interval otherwise
                if self.sim._frame % self.sim.color_update_every == 0 or self.sim._force_full_recolor == 1:
                    self._update_cut_colors(float(self.sim.damage_band_f[None]))
                    self.sim._last_color_update_frame = self.sim._frame
                self.sim.scene.particles(self.sim.particles.x, radius=radius, per_vertex_color=self.sim.p_color)
            else:
                self.sim.scene.particles(self.sim.particles.x, radius=radius, color=(1.0, 1.0, 0.0))

    def _render_eef_position(self):
        # Hide EEF during pause only
        if self.sim._transparency_pause_active:
            return
        if not self.sim._ee_available:
            return
        ee_state = self.sim.get_end_effector_state(0.0)
        if ee_state is not None:
            eef_pos = np.array(ee_state["pos"], dtype=np.float32)
            self.sim.scene.particles(eef_pos.reshape(1,3), radius=self.sim.particle_render_radius * 2.0, color=(0.0, 0.0, 1.0))

    def _render_knife_proxy(self):
        # Hide knife during pause only
        if self.sim._transparency_pause_active:
            return
        if self.sim._knife_base_xyz is None:
            return
        y_anim = float(self.sim.knife.y[None])
        z_off  = float(self.sim.knife.z_off[None])
        knife_origin_y = float(self.sim.knife.origin[None][1])
        dy = y_anim - knife_origin_y

        base = self.sim._knife_base_xyz
        xyz = np.empty_like(base)
        xyz[:, 0] = base[:, 0]
        xyz[:, 1] = base[:, 1] + dy
        xyz[:, 2] = base[:, 2] + z_off
        self.sim.scene.particles(xyz, radius=self.sim.particle_render_radius * 0.8, color=(0.8, 0.8, 0.8))

    def _render_board_proxy(self):
        # Board rendering disabled
        return
        if self._board_proxy_cache is None:
            board_sdf = self.sim.board._sdf_np
            board_origin = self.sim.board.origin[None].to_numpy()
            board_voxel  = self.sim.board._voxel_py
            Nz, Ny, Nx = board_sdf.shape
            zs = np.arange(0, Nz, 4, dtype=np.int32)
            ys = np.arange(0, Ny, 4, dtype=np.int32)
            xs = np.arange(0, Nx, 4, dtype=np.int32)
            sdf_sub = board_sdf[np.ix_(zs, ys, xs)]
            surf = np.abs(sdf_sub) < (0.5 * board_voxel)
            if np.any(surf):
                Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
                pts = np.column_stack([
                    (X[surf] + 0.5) * board_voxel + board_origin[0],
                    (Y[surf] + 0.5) * board_voxel + board_origin[1],
                    (Z[surf] + 0.5) * board_voxel + board_origin[2],
                ]).astype(np.float32)
            else:
                pts = np.empty((0,3), np.float32)
            self._board_proxy_cache = pts
        if self._board_proxy_cache.shape[0] > 0:
            self.sim.scene.particles(self._board_proxy_cache, radius=self.sim.particle_render_radius * 0.8, color=(0.6, 0.4, 0.2))

    def _render_grid_visualization(self):
        if not self.sim.show_grid:
            return
        if self._grid_proxy_cache is None:
            n = int(self.sim.n_grid)
            dx = float(self.sim.dx_s[None])
            bmin = self.sim.bounds_min[None].to_numpy()
            k = max(1, n // 16)
            xs = np.arange(0, n, k, dtype=np.int32)
            ys = np.arange(0, n, k, dtype=np.int32)
            zs = np.arange(0, n, k, dtype=np.int32)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            pts = np.column_stack([
                bmin[0] + (X.flatten() + 0.5) * dx,
                bmin[1] + (Y.flatten() + 0.5) * dx,
                bmin[2] + (Z.flatten() + 0.5) * dx,
            ]).astype(np.float32)
            self._grid_proxy_cache = pts
        self.sim.scene.particles(self._grid_proxy_cache, radius=self.sim.particle_render_radius * 0.5, color=(1.0, 0.0, 0.0))

    def _draw_info_overlay(self):
        if not self.sim.viewer_enabled or self.sim.window is None:
            return
        try:
            gui = self.sim.window.get_gui()
            with gui.sub_window("Simulation Info", 0.58, 0.58, 0.4, 0.4):
                gui.text("=== PERFORMANCE ===")
                gui.text(f"FPS: {self.sim._fps:.1f}")
                gui.text(f"Frame: {self.sim._frame}")

                gui.text("")
                gui.text("=== END EFFECTOR ===")
                gui.text(f"Position: [{self.sim._ee_position[0]:.3f}, {self.sim._ee_position[1]:.3f}, {self.sim._ee_position[2]:.3f}] m")
                gui.text(f"Velocity: [{self.sim._ee_velocity[0]:.3f}, {self.sim._ee_velocity[1]:.3f}, {self.sim._ee_velocity[2]:.3f}] m/s")
                gui.text(f"Speed: {np.linalg.norm(self.sim._ee_velocity):.3f} m/s")

                gui.text("")
                gui.text("=== FORCES ===")
                gui.text(f"Knife Applies X [ManiSkills X]: {self.sim._knife_applies_force[0]:.3f} N")
                gui.text(f"Knife Applies Y [ManiSkills Z]: {self.sim._knife_applies_force[1]:.3f} N")
                gui.text(f"Knife Applies Z [ManiSkills Y]: {self.sim._knife_applies_force[2]:.3f} N")
                gui.text(f"Force Magnitude: {np.linalg.norm(self.sim._knife_applies_force):.3f} N")

                gui.text("")
                gui.text("=== SIMULATION STATUS ===")
                gui.text("Status: Running")
                gui.text("Physics: MPM + CPIC")
        except Exception as e:
            print(f"GUI Error: {e}")

    def _update_camera_transition(self):
        """Smoothly transition camera: vertical -> horizontal -> original during pause."""
        import time
        
        # Continue camera transition even after pause ends until return is complete
        if not self.sim._transparency_pause_active:
            if self.sim._camera_transition_start_time is not None and not self.sim._camera_return_complete:
                # Continue return transition after pause ends
                current_time = time.time()
                # Use stored camera transition start time (set during pause initialization)
                pause_start_time = self.sim._camera_transition_start_time
                pause_elapsed = current_time - pause_start_time
                
                # Calculate return transition timing
                vertical_hold_end = self.sim._camera_transition_duration + self.sim._camera_vertical_duration
                horizontal_transition_end = vertical_hold_end + self.sim._camera_transition_duration
                horizontal_hold_end = horizontal_transition_end + self.sim._camera_horizontal_duration
                cut_parallel_transition_end = horizontal_hold_end + self.sim._camera_transition_duration
                cut_parallel_hold_end = cut_parallel_transition_end + self.sim._camera_cut_parallel_duration
                return_start = cut_parallel_hold_end
                return_end = return_start + self.sim._camera_return_duration
                
                if pause_elapsed < return_end:
                    # Continue smooth return transition
                    return_elapsed = pause_elapsed - return_start
                    progress = min(1.0, max(0.0, return_elapsed / self.sim._camera_return_duration))
                    t = progress * progress * (3.0 - 2.0 * progress)  # Smoothstep
                    eye = self.sim._camera_cut_parallel_eye * (1.0 - t) + self.sim._camera_original_eye * t
                    lookat = self.sim._camera_cut_parallel_lookat * (1.0 - t) + self.sim._camera_original_lookat * t
                    up = self.sim._camera_cut_parallel_up * (1.0 - t) + self.sim._camera_original_up * t
                    
                    # Normalize up vector
                    up = up / (np.linalg.norm(up) + 1e-8)
                    
                    # Store current camera state
                    self._camera_current_eye = eye.copy()
                    self._camera_current_lookat = lookat.copy()
                    self._camera_current_up = up.copy()
                    
                    # Apply camera transformation
                    try:
                        self.sim.camera.position(*eye)
                        self.sim.camera.lookat(*lookat)
                        self.sim.camera.up(*up)
                    except:
                        pass
                else:
                    # Return transition complete
                    self.sim._camera_return_complete = True
                    self.sim._camera_transition_start_time = None
                    self.sim._camera_phase = 0
                    # Ensure camera is at original position
                    try:
                        self.sim.camera.position(*self.sim._camera_original_eye)
                        self.sim.camera.lookat(*self.sim._camera_original_lookat)
                        self.sim.camera.up(*self.sim._camera_original_up)
                        self._camera_current_eye = self.sim._camera_original_eye.copy()
                        self._camera_current_lookat = self.sim._camera_original_lookat.copy()
                        self._camera_current_up = self.sim._camera_original_up.copy()
                    except:
                        pass
                    print(f"[PAUSE] Camera return transition complete")
            elif self.sim._camera_return_complete:
                # Transition complete, reset state
                if self.sim._camera_transition_start_time is not None:
                    self.sim._camera_transition_start_time = None
                    self.sim._camera_phase = 0
            return
        
        current_time = time.time()
        pause_elapsed = current_time - self.sim._transparency_pause_start_time
        
        # Initialize transition on first frame of pause
        if self.sim._camera_transition_start_time is None:
            self.sim._camera_transition_start_time = current_time
            self.sim._camera_phase = 0
            self.sim._camera_return_complete = False  # Reset return completion flag
            # Save current camera state (use stored state or fallback)
            if self._camera_current_eye is not None:
                self.sim._camera_original_eye = self._camera_current_eye.copy()
                self.sim._camera_original_lookat = self._camera_current_lookat.copy()
                self.sim._camera_original_up = self._camera_current_up.copy()
            else:
                # Fallback: use initial camera pose from config
                pose = self.sim.viewer_initial_pose
                self.sim._camera_original_eye = np.array(pose.get("eye", [0.5, 0.3, 0.5]), dtype=np.float32)
                self.sim._camera_original_lookat = np.array(pose.get("lookat", [0.0, 0.1, 0.0]), dtype=np.float32)
                self.sim._camera_original_up = np.array(pose.get("up", [0.0, 1.0, 0.0]), dtype=np.float32)
            
            # Calculate camera positions
            knife_z_off = float(self.sim.knife.z_off[None])
            knife_y = float(self.sim.knife.y[None])
            
            # Calculate mesh center from particles (for horizontal view)
            particle_positions = self.sim.particles.x.to_numpy()
            if len(particle_positions) > 0:
                # Filter particles with Z <= knife_z (visible particles during pause)
                visible_mask = particle_positions[:, 2] <= knife_z_off
                if np.any(visible_mask):
                    visible_particles = particle_positions[visible_mask]
                    mesh_center = np.mean(visible_particles, axis=0)
                else:
                    mesh_center = np.array([0.0, knife_y, knife_z_off], dtype=np.float32)
            else:
                mesh_center = np.array([0.0, knife_y, knife_z_off], dtype=np.float32)
            
            # Phase 0: Vertical view (top-down) - move object up significantly (raise lookat so object appears higher and cut surface is more visible)
            vertical_offset = 0.30  # Raise lookat by 15cm so object appears much higher and cut surface is more visible
            self.sim._camera_target_lookat = np.array([mesh_center[0], mesh_center[1] + vertical_offset, mesh_center[2]], dtype=np.float32)
            self.sim._camera_target_eye = np.array([mesh_center[0], mesh_center[1] + vertical_offset + 0.3, mesh_center[2]], dtype=np.float32)
            self.sim._camera_target_up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            
            # Phase 1: Horizontal view (side view) - centered on mesh
            self.sim._camera_horizontal_lookat = np.array([mesh_center[0], mesh_center[1], mesh_center[2]], dtype=np.float32)
            # Position camera to the side, looking at mesh center
            side_distance = 0.3
            self.sim._camera_horizontal_eye = np.array([mesh_center[0] + side_distance, mesh_center[1], mesh_center[2]], dtype=np.float32)
            self.sim._camera_horizontal_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Phase 2: Cut-parallel view (looking at cut surface from front) - parallel to cut plane
            # Cut plane is parallel to Y-Z plane, so view from Z direction (positive Z looking at negative Z)
            # Look at the cut surface (at knife Z position)
            self.sim._camera_cut_parallel_lookat = np.array([mesh_center[0], mesh_center[1], knife_z_off], dtype=np.float32)
            # Position camera in positive Z direction looking at cut surface
            z_distance = 0.3
            self.sim._camera_cut_parallel_eye = np.array([mesh_center[0], mesh_center[1], knife_z_off + z_distance], dtype=np.float32)
            self.sim._camera_cut_parallel_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Determine current phase based on pause elapsed time (more reliable)
        # Phase 0: 0-1s (transition to vertical), 1-5s (hold vertical)
        # Phase 1: 5-6s (transition to horizontal), 6-10s (hold horizontal)
        # Phase 2: 10-11s (transition to cut-parallel), 11-14s (hold cut-parallel)
        # Phase 3: 14-16s (return to original - smooth transition, happens DURING pause end)
        
        vertical_hold_start = self.sim._camera_transition_duration
        vertical_hold_end = vertical_hold_start + self.sim._camera_vertical_duration
        horizontal_transition_end = vertical_hold_end + self.sim._camera_transition_duration
        horizontal_hold_end = horizontal_transition_end + self.sim._camera_horizontal_duration
        cut_parallel_transition_end = horizontal_hold_end + self.sim._camera_transition_duration
        cut_parallel_hold_end = cut_parallel_transition_end + self.sim._camera_cut_parallel_duration
        # Return starts during pause (before pause ends)
        return_start = cut_parallel_hold_end
        return_end = return_start + self.sim._camera_return_duration
        
        if pause_elapsed < self.sim._camera_transition_duration:
            # Phase 0: Transition to vertical
            if self.sim._camera_phase != 0:
                self.sim._camera_phase = 0
            progress = pause_elapsed / self.sim._camera_transition_duration
            t = progress * progress * (3.0 - 2.0 * progress)  # Smoothstep
            eye = self.sim._camera_original_eye * (1.0 - t) + self.sim._camera_target_eye * t
            lookat = self.sim._camera_original_lookat * (1.0 - t) + self.sim._camera_target_lookat * t
            up = self.sim._camera_original_up * (1.0 - t) + self.sim._camera_target_up * t
        elif pause_elapsed < vertical_hold_end:
            # Hold vertical view
            eye = self.sim._camera_target_eye.copy()
            lookat = self.sim._camera_target_lookat.copy()
            up = self.sim._camera_target_up.copy()
        elif pause_elapsed < horizontal_transition_end:
            # Phase 1: Transition to horizontal
            if self.sim._camera_phase != 1:
                self.sim._camera_phase = 1
            progress = (pause_elapsed - vertical_hold_end) / self.sim._camera_transition_duration
            t = progress * progress * (3.0 - 2.0 * progress)  # Smoothstep
            eye = self.sim._camera_target_eye * (1.0 - t) + self.sim._camera_horizontal_eye * t
            lookat = self.sim._camera_target_lookat * (1.0 - t) + self.sim._camera_horizontal_lookat * t
            up = self.sim._camera_target_up * (1.0 - t) + self.sim._camera_horizontal_up * t
        elif pause_elapsed < horizontal_hold_end:
            # Hold horizontal view
            eye = self.sim._camera_horizontal_eye.copy()
            lookat = self.sim._camera_horizontal_lookat.copy()
            up = self.sim._camera_horizontal_up.copy()
        elif pause_elapsed < cut_parallel_transition_end:
            # Phase 2: Transition to cut-parallel
            if self.sim._camera_phase != 2:
                self.sim._camera_phase = 2
            progress = (pause_elapsed - horizontal_hold_end) / self.sim._camera_transition_duration
            t = progress * progress * (3.0 - 2.0 * progress)  # Smoothstep
            eye = self.sim._camera_horizontal_eye * (1.0 - t) + self.sim._camera_cut_parallel_eye * t
            lookat = self.sim._camera_horizontal_lookat * (1.0 - t) + self.sim._camera_cut_parallel_lookat * t
            up = self.sim._camera_horizontal_up * (1.0 - t) + self.sim._camera_cut_parallel_up * t
        elif pause_elapsed < cut_parallel_hold_end:
            # Hold cut-parallel view
            eye = self.sim._camera_cut_parallel_eye.copy()
            lookat = self.sim._camera_cut_parallel_lookat.copy()
            up = self.sim._camera_cut_parallel_up.copy()
        elif pause_elapsed < return_end:
            # Phase 3: Return to original (smooth transition)
            if self.sim._camera_phase != 3:
                self.sim._camera_phase = 3
            return_elapsed = pause_elapsed - return_start
            progress = min(1.0, return_elapsed / self.sim._camera_return_duration)
            t = progress * progress * (3.0 - 2.0 * progress)  # Smoothstep
            eye = self.sim._camera_cut_parallel_eye * (1.0 - t) + self.sim._camera_original_eye * t
            lookat = self.sim._camera_cut_parallel_lookat * (1.0 - t) + self.sim._camera_original_lookat * t
            up = self.sim._camera_cut_parallel_up * (1.0 - t) + self.sim._camera_original_up * t
        else:
            # After return transition, keep original position
            self.sim._camera_return_complete = True
            eye = self.sim._camera_original_eye.copy()
            lookat = self.sim._camera_original_lookat.copy()
            up = self.sim._camera_original_up.copy()
        
        # Normalize up vector
        up = up / (np.linalg.norm(up) + 1e-8)
        
        # Store current camera state
        self._camera_current_eye = eye.copy()
        self._camera_current_lookat = lookat.copy()
        self._camera_current_up = up.copy()
        
        # Apply camera transformation
        try:
            self.sim.camera.position(*eye)
            self.sim.camera.lookat(*lookat)
            self.sim.camera.up(*up)
        except:
            pass

    def _draw(self):
        # Update camera transition during pause
        self._update_camera_transition()
        
        # Only allow camera input when not in pause
        if not self.sim._transparency_pause_active:
            self._handle_camera_input()
            # Track camera state when user can control it (for transition start)
            try:
                # Try to get current camera state (may not be available in all Taichi versions)
                if hasattr(self.sim.camera, 'curr_position'):
                    self._camera_current_eye = np.array(self.sim.camera.curr_position, dtype=np.float32)
                    self._camera_current_lookat = np.array(self.sim.camera.curr_lookat, dtype=np.float32)
                    self._camera_current_up = np.array(self.sim.camera.curr_up, dtype=np.float32)
            except:
                pass
        self.sim.scene.set_camera(self.sim.camera)
        self._setup_scene_lighting()
        self._render_particles()
        self._render_knife_proxy()
        self._render_board_proxy()
        self._render_eef_position()
        self._render_grid_visualization()
        self._draw_info_overlay()
        self.sim.canvas.scene(self.sim.scene)
        self.sim.window.show()

    def render(self):
        self._draw()
