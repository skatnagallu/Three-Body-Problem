"""three body problem"""

from panda3d.core import NodePath, AmbientLight, DirectionalLight, LineSegs
from panda3d.core import TextNode, Material
from direct.gui.DirectGui import DGG
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectButton import DirectButton
from direct.gui.OnscreenText import OnscreenText
import numpy as np


class ThreeBodyApp(ShowBase):
    """A simple simulation of 3 body problem"""

    def __init__(self):
        super().__init__()
        self.win.setClearColor((0, 0, 0, 1))
        self.is_paused = False
        self.pause_start_time = 0
        self.paused_duration = 0
        self.positions = None
        self.velocities = None
        self.with_planet = False
        self.special = False
        self.masses = np.array([2e22, 2e22, 2e22])
        self.setup_ui()

    def setup_ui(self):
        """setup ui with buttons to start an restart"""
        self.startButton = DirectButton(
            text=("Start", "Start", "Start", "disabled"),
            text_fg=(1, 1, 1, 1),  # White text
            text_bg=(0, 0, 0, 0.5),  # Semi-transparent black background behind text
            frameColor=(
                (0.2, 0.2, 0.8, 1),
                (0.3, 0.3, 0.9, 1),
                (0.1, 0.1, 0.7, 1),
                (0.5, 0.5, 0.5, 1),
            ),  # Different colors for states
            scale=0.08,
            command=self.start_simulation,
            pos=(-0.9, 0, -0.9),
        )
        self.restartButton = DirectButton(
            text=("Restart", "Restart", "Restart", "disabled"),
            scale=0.08,
            command=self.restart_simulation,
            pos=(0.9, 0, -0.9),
        )
        self.pauseButton = DirectButton(
            text=("Pause", "Resume", "Pause", "disabled"),
            scale=0.08,
            command=self.toggle_pause,
            pos=(-0.25, 0, -0.9),
        )
        self.addTrisolarisButton = DirectButton(
            text=("Add Trisolaris", "Add Trisolaris", "Add Trisolaris", "disabled"),
            scale=0.08,
            command=self.add_trisolaris,
            pos=(0.25, 0.0, -0.9),
        )

    def start_simulation(self):
        """start simulation"""
        self.backgroundMusic = self.loader.loadMusic("./music/space.mp3")
        self.backgroundMusic.setLoop(True)
        self.backgroundMusic.setVolume(0.5)
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
            text="Years survived: 0",
            pos=(1.3, 0.9),
            scale=0.07,
            fg=(1, 1, 1, 1),
            align=TextNode.ARight,
            mayChange=True,
        )
        self.check_for_collisions = False
        self.positions, self.velocities = self.set_positions_velocities()
        self.trailMaxLength = 1e4  # Max number of points in the trail
        self.trails = []  # List to hold trails for each body
        self.trailPoints = []
        self.G = 6.67430e-11  # Gravitational constant
        self.trailColors = [
            (1, 0, 0, 1),  # Red
            (0, 1, 0, 1),  # Green
            (0, 0, 1, 1),  # Blue
            (1, 1, 0, 1),  # yellow
        ]
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
            self.backgroundMusic.play()

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
        if self.with_planet:
            self.with_planet = True
            if len(self.masses) == 4:
                self.masses = np.delete(self.masses, -1)
                self.positions = np.delete(self.positions, -1)
                self.velocities = np.delete(self.velocities, -1)
        self.is_paused = False
        self.pauseButton["text"] = "Pause"
        self.clear_bodies()
        self.clear_trails()
        self.start_simulation()

    def add_trisolaris(self):
        """Adds a planet to the simulation"""
        self.with_planet = True
        if hasattr(self, "elapsedTimeText"):
            self.restart_simulation()
        elif hasattr(self, "collisionMessage"):
            self.restart_simulation()
        else:
            self.start_simulation()

    def toggle_pause(self):
        """toggle pause resume"""
        if self.is_paused:
            # If currently paused, resume the simulation
            self.is_paused = False
            self.pauseButton["text"] = "Pause"  # Update button text
            self.paused_duration += globalClock.getRealTime() - self.pause_start_time
            # Resume tasks
            self.taskMgr.add(self.rotate_models_task, "RotateModelsTask")
            self.taskMgr.add(self.update_physics_task, "updatePhysicsTask")
            self.taskMgr.add(self.update_trail_task, "updateTrailTask")
            self.taskMgr.add(self.update_camera_task, "updateCameraTask")
            self.taskMgr.add(self.update_elapsed_time, "updateElapsedTimeTask")
            if hasattr(self, "collisionMessage"):
                self.collisionMessage.destroy()
            self.disable_camera_movement()  # Optionally disable camera movement
        else:
            # If currently running, pause the simulation
            self.is_paused = True
            self.pauseButton["text"] = "Resume"  # Update button text
            self.pause_start_time = globalClock.getRealTime()
            self.taskMgr.remove("RotateModelsTask")
            self.taskMgr.remove("updatePhysicsTask")
            self.taskMgr.remove("updateTrailTask")
            self.taskMgr.remove("updateCameraTask")
            self.taskMgr.remove(
                "updateElapsedTimeTask"
            )  # Method to remove simulation tasks
            self.enable_camera_movement()  # Allow camera movement

    def set_positions_velocities(self):
        """set positions and velocities of the bodies"""
        num_bodies = 3
        if self.with_planet:
            num_bodies = 4
            self.masses = np.append(self.masses, 1e10)
        velocities = np.random.rand(num_bodies, 3) - 0.5
        positions = np.random.rand(num_bodies, 3) * 12
        total_mass = np.sum(self.masses)
        center_of_mass_velocity = (
            np.sum(velocities * self.masses[:, None], axis=0) / total_mass
        )
        velocities = velocities - center_of_mass_velocity
        if self.with_planet:
            positions[3, :] += 10

        if self.special:
            positions, velocities = self.read_positions_and_velocities(
                "./initial_conditions/special.csv"
            )

        return positions, velocities

    def read_positions_and_velocities(self, filename):
        '''Load special initial conditions'''
        # Load the CSV file. Assuming the file has headers and skip them
        data = np.genfromtxt(
            filename, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )

        # Prepare dictionaries to hold positions and velocities for each solution type
        positions = {"Lagrange": [], "FigureEight": []}
        velocities = {"Lagrange": [], "FigureEight": []}

        # Loop through each row of the CSV file
        for row in data:
            # Determine the solution type from the 'Solution' column
            solution_type = row["Solution"]
            if solution_type == "Lagrange":
                positions["Lagrange"].append((row["PosX"], row["PosY"], row["PosZ"]))
                velocities["Lagrange"].append((row["VelX"], row["VelY"], row["VelZ"]))
            elif solution_type == "FigureEight":
                positions["FigureEight"].append((row["PosX"], row["PosY"], row["PosZ"]))
                velocities["FigureEight"].append(
                    (row["VelX"], row["VelY"], row["VelZ"])
                )
        print(positions)
        # Convert lists to numpy arrays for easier manipulation later
        for key in positions.keys():
            positions[key] = np.array(positions[key])
            velocities[key] = np.array(velocities[key])

        return positions, velocities

    def set_angular_velocity_for_body(self, body_index, angular_velocity):
        """Set angular velocity"""
        barycenter = self.calculate_barycenter()
        body_pos = np.array(self.bodies[body_index].getPos())

        # Calculate the radial distance from the body to the barycenter
        radius_vector = body_pos - barycenter
        radius = np.linalg.norm(radius_vector)

        # Calculate the tangential velocity
        tangential_velocity_magnitude = radius * angular_velocity

        # Calculate the direction of the tangential velocity (perpendicular to the radius vector)
        # For simplicity, assume a circular orbit in the x-y plane
        tangential_velocity_direction = np.array(
            [-radius_vector[1], radius_vector[0], 0]
        )
        tangential_velocity_direction = tangential_velocity_direction / np.linalg.norm(
            tangential_velocity_direction
        )

        # Apply the magnitude to the direction
        tangential_velocity = (
            tangential_velocity_direction * tangential_velocity_magnitude
        )

        # Update the velocity of the body
        self.velocities[body_index] = tangential_velocity

    def enable_camera_movement(self):
        """enable camera movement"""
        self.enableMouse()

    def disable_camera_movement(self):
        """disable camera"""
        self.disableMouse()

    def init_bodies(self):
        """intialise stars"""
        self.bodies = []
        for _ in range(3):
            # body = self.loader.loadModel("models/misc/sphere")
            body = self.loader.loadModel("../models/Sun.glb")
            # Load and apply texture
            texture = self.loader.loadTexture("./textures/star_texture.png")
            body.setTexture(texture, 1)

            # Set up and apply material
            material = Material()
            material.setEmission((1, 1, 0, 1))  # Example: glowing yellow
            body.setMaterial(material, 1)
            body.reparent_to(self.render)
            self.bodies.append(body)

        if self.with_planet:
            trisolaris = self.loader.loadModel("../models/Earth.glb")
            material = Material()
            material.setEmission((1, 1, 0, 1))
            material.setDiffuse((0, 0, 1, 1))
            trisolaris.setMaterial(material, 1)
            trisolaris.reparent_to(self.render)
            self.bodies.append(trisolaris)

        for i, body in enumerate(self.bodies):
            body.setPos(
                self.positions[i][0], self.positions[i][1], self.positions[i][2]
            )
            initial_h = np.random.uniform(0, 360)  # Random heading
            initial_p = np.random.uniform(-90, 90)  # Random pitch
            initial_r = np.random.uniform(0, 360)  # Random roll
            size = 0.8
            body.setHpr(initial_h, initial_p, initial_r)
            if i == 3:
                body.setScale(0.5 * 1e-3)
            else:
                body.setScale(size * 1e-3)

    def init_trails(self):
        """initialise trails"""
        for _, color in zip(self.bodies, self.trailColors):
            self.trailPoints.append(
                []
            )  # Initialize an empty list for each body's trail points
            trailVisual = NodePath(
                LineSegs().create()
            )  # Create an empty NodePath for the trail visualization
            trailVisual.setColor(color)
            trailVisual.reparentTo(self.render)
            self.trails.append(trailVisual)

    def clear_trails(self):
        """function to clear trails"""
        for trail in self.trails:
            trail.node().removeAllGeoms()  # Clear the geometry from each trail NodePath
        self.trailPoints = [[] for _ in self.bodies]  # Reset the trail points data

    def clear_bodies(self):
        """a function to clear bodies"""
        for body in self.bodies:
            body.removeNode()  # This removes the body from the scene
        self.bodies = []  # Reset the list of bodies

    def display_collision_message(self, message):
        """collison message"""
        self.collisionMessage = OnscreenText(
            text=message,
            pos=(0, 0),
            scale=0.07,
            fg=(1, 0, 0, 1),
            align=TextNode.ACenter,
            mayChange=False,
        )

    def calculate_barycenter(self):
        """compute center of mass"""
        weighted_positions = np.zeros(3)
        total_mass = 0

        for body, mass in zip(self.bodies, self.masses):
            pos = body.getPos()
            weighted_positions += np.array([pos.x, pos.y, pos.z]) * mass
            total_mass += mass

        if total_mass > 0:
            return weighted_positions / total_mass
        return np.zeros(3)  # Default to origin if no mass

    def update_camera(self):
        """update camer apositions according to average position of the bodies"""
        if not self.is_paused:
            barycenter = self.calculate_barycenter()
            max_distance = np.max(np.linalg.norm(self.positions - barycenter, axis=1))
            camera_distance = max(30, max_distance * 1.5)
            self.camera.setPos(
                barycenter[0], barycenter[1] - camera_distance, barycenter[2]
            )
            self.camera.lookAt(barycenter[0], barycenter[1], barycenter[2])

    def rotate_models_task(self, task):
        """rotate stars"""
        dt = globalClock.getDt()  # Get the time since the last frame

        for body in self.bodies:
            body.setH(
                body.getH() + 60 * dt
            )  # Rotate 60 degrees per second around the vertical axis

        return task.cont  # Continue the task indefinitely

    def compute_accelerations(self, positions, masses):
        """Computes acceleration due to gravitational forces"""
        n_bodies = len(masses)
        # Initialize accelerations array with float type to avoid UFuncTypeError
        accelerations = np.zeros_like(positions, dtype=float)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    # Vector from body i to body j
                    r_ij = positions[j] - positions[i]
                    # Distance between bodies i and j
                    distance_ij = np.linalg.norm(r_ij)
                    if distance_ij < 0.1:
                        self.check_for_collisions = True
                    # Calculate force magnitude
                    force_magnitude = self.G * masses[i] * masses[j] / distance_ij**2
                    # Calculate acceleration contribution from body j to body i
                    acc_contribution = force_magnitude / masses[i] * r_ij / distance_ij
                    accelerations[i] += acc_contribution
        return accelerations

    def update_physics(self, dt):
        """Need to put the gravitational solver here"""
        accelerations = self.compute_accelerations(
            self.positions, self.masses
        )  # This would be calculated based on gravitational forces

        # Update velocities
        self.velocities += accelerations * dt

        # Update positions
        self.positions += self.velocities * dt * 1e-12

    def update_physics_task(self, task):
        """Update positions based on update physics"""
        dt = globalClock.getDt()  # Time since last frame in seconds

        self.update_physics(dt)

        # Update model positions based on the updated positions array
        for i, body in enumerate(self.bodies):
            body.setPos(
                self.positions[i][0], self.positions[i][1], self.positions[i][2]
            )
        if self.check_for_collisions:
            self.display_collision_message("Stars Collided!")
            self.enableMouse()
            self.backgroundMusic.stop()
            return task.done  # Stops this task
        return task.cont  # Continue the task indefinitely

    def update_camera_task(self, task):
        """update camera positions"""
        dt = globalClock.getDt()
        if not self.is_paused:
            barycenter = self.calculate_barycenter()
            max_distance = np.max(np.linalg.norm(self.positions - barycenter, axis=1))
            camera_distance = max(30, max_distance * 1.5)
            self.camera.setPos(
                np.linalg.norm(self.positions, axis=0)[0],
                np.linalg.norm(self.positions, axis=0)[1] - camera_distance,
                np.linalg.norm(self.positions, axis=0)[2],
            )

            self.camera.lookAt(barycenter[0], barycenter[1], barycenter[2])
            # self.camera.lookAt(np.linalg.norm(self.positions,axis=0)[0], np.linalg.norm(self.positions,axis=0)[1],np.linalg.norm(self.positions,axis=0)[2])
        return task.cont

    def update_trail_visual(self, trail_points, body_index):
        """adds trajectory trails"""
        trailVisual = LineSegs()
        color = self.trailColors[body_index]
        trailVisual.setColor(*color)  # Example color: Yellow
        trailVisual.setThickness(2.0)
        for point in trail_points:
            (
                trailVisual.moveTo(point)
                if point == trail_points[0]
                else trailVisual.drawTo(point)
            )
        trailGeom = trailVisual.create(False)

        self.trails[body_index].node().removeAllGeoms()  # Clear the previous geometry
        self.trails[body_index].node().addGeomsFrom(trailGeom)  # Add the new geometry

    def update_trail_task(self, task):
        """Updates trails"""
        dt = globalClock.getDt()
        for i, body in enumerate(self.bodies):
            pos = body.getPos()
            trail_points = self.trailPoints[i]

            # Update the trail points
            if len(trail_points) >= self.trailMaxLength:
                trail_points.pop(0)  # Remove the oldest point
            trail_points.append((pos.x, pos.y, pos.z))  # Add the new point

            # Recreate the visual trail
            self.update_trail_visual(trail_points, i)

        return task.cont

    def update_elapsed_time(self, task):
        """print elapsed time"""
        elapsed_time = int(
            globalClock.getRealTime() - self.simStartTime - self.paused_duration
        )
        self.elapsedTimeText.setText(f"Years survived: {elapsed_time}")
        if self.check_for_collisions:
            return task.done
        return task.cont


if __name__ == "__main__":
    app = ThreeBodyApp()
    app.run()
