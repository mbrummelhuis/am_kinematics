import sympy as sp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg

from system import SimpleAerialManipulator
from controller import Controller


class Simulator:
    def __init__(self, system, controller, sim_params):
        self.model = system
        self.controller = controller
        self.sim_params = sim_params
        self.state_data = []
    
    ''' Simulate the system for a given time with Euler integration scheme'''
    def simulate(self, x0, ref):
        x = x0
        t = np.linspace(0, self.sim_params['t'], self.sim_params['dt'])

        for i in range(t):
            # Get control input
            u = self.controller.get_control_input(x, ref)
            # Euler integration of system
            x = self.model.dynamics(x, u)*self.sim_params['dt'] + x
            self.state_data.append(x)

    def animate_system(self):
        def animate(i):
            # Manipulator
            x_b = self.state_data[i][0]
            y_b = self.state_data[i][1]
            x_e = self.state_data[i][0] + self.system.system_params['l']*sp.sin(self.state_data[i][2]+self.state_data[i][3])
            y_e = self.state_data[i][1] - self.system.system_params['l']*sp.cos(self.state_data[i][2]+self.state_data[i][3])
            ln1.set_data([x_b, x_e], [y_b, y_e])

            # Drone
            trans_data = Affine2D().rotate_around(x_b, y_b, self.state_data[i][2]).translate(x_b, y_b)
            drone_img.set_transform(trans_data)

        fig, ax = plt.subplots(1,1)
        ax.set_facecolor('k')
        ax.get_xaxis().set_ticks([]) # Hides the x-axis ticks
        ax.get_yaxis().set_ticks([]) # Hides the y-axis ticks
        # Drone image
        img = mpimg.imread('drone.png')
        drone_img = ax.imshow(img, extent=[-0.2, 0.2, -0.2, 0.2], alpha=0.8)


        # manipulator
        ln1, = plt.plot([], [], 'ro-', animated=True)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ani = animation.FuncAnimation(fig, animate, frames=len(self.state_data), interval=50, blit=True)

        plt.show()

if __name__=="__main__":
    dt = 0.01

    # Initialize the model
    model_name = '2d_simple_aerial_manipulator'
    state_vector = sp.symbols('xb yb pb q xb_dot yb_dot pb_dot qb_dot')
    input_vector = sp.symbols('T_1 T_2 tau')
    model_parameters = {
        sp.symbols('m_b'): 1.0,  # Mass of the base [kg]
        'm_m': 0.1,  # Mass of the manipulator [kg]
        'I_b': 1.0*0.4/12,  # Moment of inertia of the base [kg.m^2]
        'I_m': 0.1*0.5/12,  # Moment of inertia of the manipulator [kg.m^2]
        'l': 0.5,   # Length of the manipulator [m]
        'r': 0.2,   # Distance of the rotors from the center of mass [m]
        'g': 9.81}  # Acceleration due to gravity [m/s^2]

    # Initialize the model
    am_model = SimpleAerialManipulator(state_vector, input_vector, model_parameters)
    am_model.load_model(model_name)

    # Initialize the controller
    jacobian_name = '2d_simple_aerial_manipulator_end_effector_jacobian'
    Kp = sp.Matrix([[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    mixer = sp.Matrix([[1, 1, 0], [0.1, 0.1, 0], [0, 0, 1]])
    controller_parameters = {
            'dt': dt,               # Time step [s]
            'alpha': 0.3,           # EWMA smoothing factor
            'min_thrust': 25,       # Minimum rotor thrust (N)
            'max_thrust': 5,        # Maximum rotor thrust (N)
            'Kp': Kp,               # Proportional gain matrix
            'Ki': Kp,               # Integral gain matrix
            'Kd': Kp,               # Derivative gain matrix
            'mixing_matrix': mixer  # Mixing matrix
        }
    controller = Controller(state_vector, input_vector, controller_parameters)

    # Initialize the simulator
    sim_params = {
        'dt': dt,  # Time step [s]
        't': 100  # Simulation time [s]
    }
    x0 = [0, 0, 0, 0]
    sim = Simulator(am_model, controller, sim_params)
    sim.simulate()