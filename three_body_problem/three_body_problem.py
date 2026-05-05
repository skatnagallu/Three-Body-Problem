"""three body problem"""

from panda3d.core import NodePath, AmbientLight, DirectionalLight, LineSegs
from panda3d.core import TextNode, Material
from direct.filter.CommonFilters import CommonFilters
from direct.gui.DirectGui import DGG
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectButton import DirectButton
from direct.gui.DirectOptionMenu import DirectOptionMenu
from direct.gui.OnscreenText import OnscreenText
import numpy as np


class ThreeBodyApp(ShowBase):
    """A simple simulation of 3 body problem"""

    def __init__(self):
        super().__init__()
        self.win.setClearColor((0.02, 0.02, 0.05, 1))
        self.is_paused = False
        self.pause_start_time = 0
        self.paused_duration = 0
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.with_planet = False
        self.mode = "random"
        self.masses = np.array([2e22, 2e22, 2e22])
        self.G = 6.67430e-11
        self.time_scale = 1.0
        
        self.orbits = {
            'Figure-8': (0.347111, 0.532728),
            'Butterfly I': (0.30689, 0.12551),
            'Butterfly II': (0.39295, 0.09758),
            'Bumblebee': (0.18428, 0.58719),
            'Moth I': (0.46444, 0.39606),
            'Moth II': (0.43917, 0.45297),
            'Butterfly III': (0.40592, 0.23016),
            'Moth III': (0.38344, 0.37736),
            'Goggles': (0.08330, 0.12789),
            'Butterfly IV': (0.350112, 0.07934),
            'Dragonfly': (0.08058, 0.58884),
            'Yarn': (0.55906, 0.34919),
            'Yin-Yang I': (0.51394, 0.30474),
            'Yin-Yang II': (0.41682, 0.33033)
        }
        
        self.filters = CommonFilters(self.win, self.cam)
        self.filters.setBloom(blend=(0, 0, 0, 1), desat=-0.5, intensity=2.0, size="small")
        self.setup_ui()

    def setup_ui(self):
        """setup ui with buttons to start an restart"""
        btn_args = dict(
            text_fg=(1, 1, 1, 1),
            text_bg=(0, 0, 0, 0.5),
            frameColor=((0.2, 0.2, 0.8, 1), (0.3, 0.3, 0.9, 1), (0.1, 0.1, 0.7, 1), (0.5, 0.5, 0.5, 1)),
            scale=0.06,
        )
        self.btn_random = DirectButton(text=("Random", "Random", "Random", "disabled"),
            command=lambda: self.set_mode("random"), pos=(-1.2, 0, -0.9), **btn_args)
        
        self.mode_menu = DirectOptionMenu(
            text="Stable Orbits",
            scale=0.06,
            items=["Lagrange", "Figure-8", "Butterfly I", "Butterfly II", "Bumblebee", "Moth I", "Moth II", "Butterfly III", "Moth III", "Goggles", "Butterfly IV", "Dragonfly", "Yarn", "Yin-Yang I", "Yin-Yang II"],
            initialitem=0,
            highlightColor=(0.65, 0.65, 0.65, 1),
            command=self.set_mode,
            pos=(-0.75, 0, -0.9),
            text_bg=(0,0,0,0.5),
            text_fg=(1,1,1,1)
        )
        self.addTrisolarisButton = DirectButton(text=("Add Trisolaris", "Add Trisolaris", "Add Trisolaris", "disabled"),
            command=self.add_trisolaris, pos=(0.25, 0.0, -0.9), **btn_args)
        self.pauseButton = DirectButton(text=("Pause", "Resume", "Pause", "disabled"),
            command=self.toggle_pause, pos=(0.8, 0, -0.9), **btn_args)
        self.restartButton = DirectButton(text=("Restart", "Restart", "Restart", "disabled"),
            command=self.restart_simulation, pos=(1.2, 0, -0.9), **btn_args)

    def set_mode(self, mode):
        self.mode = mode
        self.with_planet = False
        self.restart_simulation()

    def start_simulation(self):
        """start simulation"""
        try:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            music_path = os.path.join(base_dir, "music", "space.mp3")
            self.backgroundMusic = self.loader.loadMusic(music_path)
            self.backgroundMusic.setLoop(True)
            self.backgroundMusic.setVolume(0.5)
            self.backgroundMusic.play()
        except:
            self.backgroundMusic = None

        self.pause_start_time = 0
        self.paused_duration = 0
        directional_light = DirectionalLight("directional_light")
        directional_light.set_direction((-5, -5, -5))
        directional_light.set_color((1, 1, 1, 1))
        directional_light.set_specular_color((1, 1, 1, 1))

        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color((0.3, 0.3, 0.2, 1))
        alnp = self.render.attach_new_node(ambient_light)
        self.render.set_light(alnp)
        dlnp = self.render.attach_new_node(directional_light)
        self.render.set_light(dlnp)
        
        self.elapsedTimeText = OnscreenText(
            text="Years survived: 0", pos=(1.3, 0.9), scale=0.07, fg=(1, 1, 1, 1), align=TextNode.ARight, mayChange=True
        )
        self.check_for_collisions = False
        
        # Load conditions based on mode
        self.positions, self.velocities = self.set_positions_velocities()
        self.accelerations = self.compute_accelerations(self.positions, self.masses)
        
        self.trailMaxLength = 1000
        self.trails = []
        self.trailPoints = []
        
        # Improved colors for aesthetics
        # Realistic Star Spectral Colors (Red, Blue, Yellow-White, Planet-Grey)
        self.trailColors = [(1.0, 0.2, 0.2, 1), (0.2, 0.5, 1.0, 1), (1.0, 0.9, 0.5, 1), (0.5, 0.5, 0.5, 1)]
        self.init_bodies()
        self.init_trails()
        self.update_camera()
        self.is_paused = False
        
        if self.with_planet:
            self.set_angular_velocity_for_body(3, 1)
            
        if not self.taskMgr.hasTaskNamed("updatePhysicsTask"):
            self.simStartTime = globalClock.getRealTime()
            self.taskMgr.add(self.update_physics_task, "updatePhysicsTask")
            self.taskMgr.add(self.rotate_models_task, "RotateModelsTask")
            self.taskMgr.add(self.update_trail_task, "updateTrailTask")
            self.taskMgr.add(self.update_camera_task, "updateCameraTask")
            self.taskMgr.add(self.update_elapsed_time, "updateElapsedTimeTask")

    def restart_simulation(self):
        """restart simulation"""
        self.taskMgr.remove("updatePhysicsTask")
        self.taskMgr.remove("RotateModelsTask")
        self.taskMgr.remove("updateTrailTask")
        self.taskMgr.remove("updateCameraTask")
        self.taskMgr.remove("updateElapsedTimeTask")
        if hasattr(self, "collisionMessage"):
            self.collisionMessage.destroy()
        if hasattr(self, "elapsedTimeText"):
            self.elapsedTimeText.destroy()
        if hasattr(self, 'backgroundMusic') and self.backgroundMusic:
            self.backgroundMusic.stop()
            
        self.is_paused = False
        self.pauseButton["text"] = "Pause"
        self.clear_bodies()
        self.clear_trails()
        self.start_simulation()

    def add_trisolaris(self):
        """Adds a planet to the simulation"""
        self.with_planet = True
        self.restart_simulation()

    def toggle_pause(self):
        """toggle pause resume"""
        if self.is_paused:
            self.is_paused = False
            self.pauseButton["text"] = "Pause"
            self.paused_duration += globalClock.getRealTime() - self.pause_start_time
            self.taskMgr.add(self.rotate_models_task, "RotateModelsTask")
            self.taskMgr.add(self.update_physics_task, "updatePhysicsTask")
            self.taskMgr.add(self.update_trail_task, "updateTrailTask")
            self.taskMgr.add(self.update_camera_task, "updateCameraTask")
            self.taskMgr.add(self.update_elapsed_time, "updateElapsedTimeTask")
            if hasattr(self, "collisionMessage"):
                self.collisionMessage.destroy()
            self.disable_camera_movement()
        else:
            self.is_paused = True
            self.pauseButton["text"] = "Resume"
            self.pause_start_time = globalClock.getRealTime()
            self.taskMgr.remove("RotateModelsTask")
            self.taskMgr.remove("updatePhysicsTask")
            self.taskMgr.remove("updateTrailTask")
            self.taskMgr.remove("updateCameraTask")
            self.taskMgr.remove("updateElapsedTimeTask")
            self.enable_camera_movement()

    def set_positions_velocities(self):
        """set positions and velocities of the bodies"""
        mode_lower = self.mode.lower()
        if mode_lower == "random":
            num_bodies = 4 if self.with_planet else 3
            self.G = 1.0
            self.time_scale = 0.05
            self.masses = np.array([1.0] * 3)
            if self.with_planet:
                self.masses = np.append(self.masses, 1e-4)
            velocities = (np.random.rand(num_bodies, 3) - 0.5) * 1.5
            positions = (np.random.rand(num_bodies, 3) - 0.5) * 10.0
            
            # Center of mass correction
            total_mass = np.sum(self.masses)
            center_of_mass_velocity = np.sum(velocities * self.masses[:, None], axis=0) / total_mass
            velocities -= center_of_mass_velocity
            if self.with_planet:
                positions[3, :] += 10
                
        elif self.mode in self.orbits:
            vx, vy = self.orbits[self.mode]
            self.G = 1.0
            self.time_scale = 0.05
            self.masses = np.array([1.0, 1.0, 1.0])
            
            positions = np.array([
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ])
            velocities = np.array([
                [vx, vy, 0.0],
                [vx, vy, 0.0],
                [-2*vx, -2*vy, 0.0]
            ])
            
            positions *= 2.0
            velocities /= np.sqrt(2.0)
            
            if self.with_planet:
                self.masses = np.append(self.masses, 1e-4)
                positions = np.vstack([positions, [5.0, 5.0, 5.0]])
                velocities = np.vstack([velocities, [0.5, 0, 0]])

        elif mode_lower == "lagrange":
            self.G = 1.0
            self.time_scale = 0.05
            self.masses = np.array([1.0, 1.0, 1.0])
            r = 1.0
            positions = np.array([
                [r, 0.0, 0.0],
                [-r/2, r*np.sqrt(3)/2, 0.0],
                [-r/2, -r*np.sqrt(3)/2, 0.0]
            ])
            v_mag = np.sqrt(1/np.sqrt(3))
            velocities = np.array([
                [0.0, v_mag, 0.0],
                [-v_mag*np.sqrt(3)/2, -v_mag/2, 0.0],
                [v_mag*np.sqrt(3)/2, -v_mag/2, 0.0]
            ])
            positions *= 2.0
            velocities /= np.sqrt(2.0)
            if self.with_planet:
                self.masses = np.append(self.masses, 1e-4)
                positions = np.vstack([positions, [5.0, 5.0, 5.0]])
                velocities = np.vstack([velocities, [0.5, 0, 0]])

        return positions, velocities

    def set_angular_velocity_for_body(self, body_index, angular_velocity):
        """Set angular velocity for the planet"""
        barycenter = self.calculate_barycenter()
        body_pos = self.positions[body_index]
        radius_vector = body_pos - barycenter
        radius = np.linalg.norm(radius_vector)
        if radius == 0: return
        tangential_velocity_magnitude = radius * angular_velocity
        tangential_velocity_direction = np.array([-radius_vector[1], radius_vector[0], 0])
        norm = np.linalg.norm(tangential_velocity_direction)
        if norm == 0:
            tangential_velocity_direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        tangential_velocity_direction /= norm
        tangential_velocity = tangential_velocity_direction * tangential_velocity_magnitude
        self.velocities[body_index] = tangential_velocity

    def enable_camera_movement(self):
        self.enableMouse()

    def disable_camera_movement(self):
        self.disableMouse()

    def init_bodies(self):
        """intialise stars"""
        self.bodies = []
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tex_path = os.path.join(base_dir, "textures", "star_texture.png")
        
        # Realistic star colors: Red, Blue, Yellow-White
        star_colors = [(1.0, 0.2, 0.2, 1), (0.2, 0.5, 1.0, 1), (1.0, 0.9, 0.5, 1)]
        
        for i in range(3):
            body = self.loader.loadModel("models/misc/sphere")
            
            try:
                texture = self.loader.loadTexture(tex_path)
                body.setTexture(texture, 1)
            except:
                pass

            body.setLightOff()
            c = star_colors[i]
            body.setColorScale(c[0]*3, c[1]*3, c[2]*3, 1)
            body.reparent_to(self.render)
            self.bodies.append(body)

        if self.with_planet:
            trisolaris = self.loader.loadModel("models/misc/sphere")
            material = Material()
            material.setEmission((0.2, 0.2, 1, 1))
            material.setDiffuse((0, 0, 1, 1))
            trisolaris.setMaterial(material, 1)
            trisolaris.reparent_to(self.render)
            self.bodies.append(trisolaris)

        for i, body in enumerate(self.bodies):
            body.setPos(self.positions[i][0], self.positions[i][1], self.positions[i][2])
            body.setHpr(np.random.uniform(0, 360), np.random.uniform(-90, 90), np.random.uniform(0, 360))
            if i == 3:
                body.setScale(0.08 if self.mode.lower() == "random" else 0.1)
            else:
                body.setScale(0.15 if self.mode.lower() == "random" else 0.15)

    def init_trails(self):
        """initialise trails"""
        for _, color in zip(self.bodies, self.trailColors):
            self.trailPoints.append([])
            trailVisual = NodePath(LineSegs().create())
            trailVisual.setColor(*color)
            trailVisual.reparentTo(self.render)
            self.trails.append(trailVisual)

    def clear_trails(self):
        if hasattr(self, 'trails'):
            for trail in self.trails:
                trail.node().removeAllGeoms()
        if hasattr(self, 'bodies'):
            self.trailPoints = [[] for _ in self.bodies]
        else:
            self.trailPoints = []

    def clear_bodies(self):
        if hasattr(self, 'bodies'):
            for body in self.bodies:
                body.removeNode()
        self.bodies = []

    def display_collision_message(self, message):
        self.collisionMessage = OnscreenText(
            text=message, pos=(0, 0), scale=0.1, fg=(1, 0, 0, 1), align=TextNode.ACenter, mayChange=False
        )

    def calculate_barycenter(self):
        total_mass = np.sum(self.masses)
        if total_mass > 0:
            return np.sum(self.positions * self.masses[:, None], axis=0) / total_mass
        return np.zeros(3)

    def update_camera(self):
        if not self.is_paused:
            barycenter = self.calculate_barycenter()
            max_distance = np.max(np.linalg.norm(self.positions - barycenter, axis=1))
            camera_distance = max(15, max_distance * 2.5)
            self.camera.setPos(barycenter[0], barycenter[1] - camera_distance, barycenter[2] + camera_distance*0.3)
            self.camera.lookAt(barycenter[0], barycenter[1], barycenter[2])

    def rotate_models_task(self, task):
        dt = globalClock.getDt()
        for body in self.bodies:
            body.setH(body.getH() + 60 * dt)
        return task.cont

    def compute_accelerations(self, positions, masses):
        """Vectorized computation of gravitational accelerations"""
        # Calculate p_j - p_i (vector pointing from body i to body j)
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :] # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1) # (N, N)
        dist = np.sqrt(dist_sq) # (N, N)
        
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Check for collisions with a threshold based on mode
        col_thresh = 0.05  # Much smaller threshold so they slingshot instead of crashing
        if np.any(dist < col_thresh):
            pass # allow them to fly! self.check_for_collisions = True
            
        force_mag = self.G * masses[np.newaxis, :] / (dist_sq * dist) # (N, N)
        acc = force_mag[..., np.newaxis] * diff # (N, N, 3)
        return np.sum(acc, axis=1) # (N, 3)

    def update_physics(self, dt):
        """Symplectic Velocity Verlet integrator for stable orbits"""
        dt_scaled = dt * self.time_scale * 10.0  # Speed multiplier
        
        # Update positions
        self.positions += self.velocities * dt_scaled + 0.5 * self.accelerations * dt_scaled**2
        
        # Compute new accelerations
        new_accelerations = self.compute_accelerations(self.positions, self.masses)
        
        # Update velocities
        self.velocities += 0.5 * (self.accelerations + new_accelerations) * dt_scaled
        self.accelerations = new_accelerations

    def update_physics_task(self, task):
        dt = globalClock.getDt()
        
        # Sub-stepping for higher precision integration
        sub_steps = 10
        sub_dt = dt / sub_steps
        for _ in range(sub_steps):
            self.update_physics(sub_dt)

        for i, body in enumerate(self.bodies):
            body.setPos(self.positions[i][0], self.positions[i][1], self.positions[i][2])
            
        if self.check_for_collisions:
            self.display_collision_message("Collision Detected!")
            self.enableMouse()
            if hasattr(self, 'backgroundMusic') and self.backgroundMusic:
                self.backgroundMusic.stop()
            return task.done
        return task.cont

    def update_camera_task(self, task):
        self.update_camera()
        return task.cont

    def update_trail_visual(self, trail_points, body_index):
        trailVisual = LineSegs()
        color = self.trailColors[body_index]
        trailVisual.setColor(*color)
        trailVisual.setThickness(2.0)
        
        for i, point in enumerate(trail_points):
            if i == 0:
                trailVisual.moveTo(point)
            else:
                trailVisual.drawTo(point)
                
        trailGeom = trailVisual.create(False)
        self.trails[body_index].node().removeAllGeoms()
        self.trails[body_index].node().addGeomsFrom(trailGeom)

    def update_trail_task(self, task):
        for i, body in enumerate(self.bodies):
            pos = body.getPos()
            trail_points = self.trailPoints[i]

            if len(trail_points) >= self.trailMaxLength:
                trail_points.pop(0)
            trail_points.append((pos.x, pos.y, pos.z))

            if len(trail_points) > 1:
                self.update_trail_visual(trail_points, i)

        return task.cont

    def update_elapsed_time(self, task):
        elapsed_time = int(globalClock.getRealTime() - self.simStartTime - self.paused_duration)
        self.elapsedTimeText.setText(f"Years survived: {elapsed_time}")
        if self.check_for_collisions:
            return task.done
        return task.cont


if __name__ == "__main__":
    app = ThreeBodyApp()
    app.run()
