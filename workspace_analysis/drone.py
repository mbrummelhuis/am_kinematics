import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DroneVisualModel():
    def __init__(self, body = (0.2, 0.08, 0.05), rotor = (0.2286, 0.3, 0.3), alpha = 0.75):
        """Visual model for drawing the drone in the 3D matplotlib rendering

        Args:
            body (tuple, float): Body dimensions (length, width, height). Defaults to (0.2, 0.08, 0.05).
            rotor (tuple, float): Rotor dimensions (diameter, offset in body x, offset in body y). Defaults to (0.2286, 0.3, 0.3).
            alpha (float): Opacity factor of the rendered faces. Defaults to 0.75.
        """
        # Body
        self.body_length = body[0]
        self.body_width = body[1]
        self.body_height = body[2]

        # Rotor
        self.rotor_diameter = rotor[0] # 7 inch in meters, 7 inch props
        # rotor_diameter = 0.2286 # 9 inch in meters, 9 inch props
        self.rotor_offset = np.array([(rotor[1], rotor[2]),
                                      (rotor[1], -rotor[2]),
                                      (-rotor[1], rotor[2]),
                                      (-rotor[1], -rotor[2])])

        self.alpha = alpha

    def drawDroneModel(self, ax):
        # Plotting body
        ax.add_collection3d(self.__plotBox())

        # Plot rotors
        for rotor in self.rotor_offset:
            ax.add_collection3d(self.__plotCircle(rotor))

        # Plot arms
        ax.plot([self.body_length/2., self.rotor_offset[0][0]], [self.body_width/2., self.rotor_offset[0][1]], 0., color='k', linewidth=3)
        ax.plot([self.body_length/2., self.rotor_offset[1][0]], [-self.body_width/2., self.rotor_offset[1][1]], 0., color='k', linewidth=3)
        ax.plot([-self.body_length/2., self.rotor_offset[2][0]], [self.body_width/2., self.rotor_offset[2][1]], 0., color='k', linewidth=3)
        ax.plot([-self.body_length/2., self.rotor_offset[3][0]], [-self.body_width/2., self.rotor_offset[3][1]], 0., color='k', linewidth=3)


    def __plotCircle(self, rotor_center=(0., 0.)):
        # Define parameters for the circle
        resolution = 10  # Resolution of the circle (number of points along the circumference)
        theta = np.linspace(0, 2*np.pi, resolution)
        x = self.rotor_diameter/2. * np.cos(theta) + rotor_center[0]
        y = self.rotor_diameter/2. * np.sin(theta) + rotor_center[1]
        z = np.zeros_like(x)  # Z-coordinate is zero for all points, making it a flat circle

        # Create a polygon representing the circle
        polygon = [[x[i], y[i], z[i]] for i in range(resolution)]

        # Create a Poly3DCollection object with a single polygon
        circle = Poly3DCollection([polygon], alpha=self.alpha, facecolors='yellow', linewidths=1, edgecolors='k')
        return circle

    def __plotBox(self):
        vertices = np.array([
            [-self.body_length/2., -self.body_width/2., -self.body_height/2.], [self.body_length/2., -self.body_width/2., -self.body_height/2.],
            [self.body_length/2., self.body_width/2., -self.body_height/2.], [-self.body_length/2., self.body_width/2., -self.body_height/2.],  # Bottom vertices
            [-self.body_length/2., -self.body_width/2., self.body_height/2.], [self.body_length/2., -self.body_width/2., self.body_height/2.],
            [self.body_length/2., self.body_width/2., self.body_height/2.], [-self.body_length/2., self.body_width/2., self.body_height/2.]  # Top vertices
        ])

        # Define faces of the box using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face 1
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face 2
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face 3
            [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side face 4
        ]

        box = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=self.alpha)
        return box