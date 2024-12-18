import sympy as sp
import numpy as np

from workspace_analysis.kinematics import KinematicsSolver
from workspace_analysis.model import Model
from visualizer import visualise

# Settings
L1_length = 0.110
L2_length = 0.311
L3_length = 0.273

steps = 10 # Iteration steps per DOF

# Joint limits
q1_MIN = float(-sp.pi)
q1_MAX = float(sp.pi)
q2_MIN = float(-1/8*sp.pi)
q2_MAX = float(1/8*sp.pi)
q3_MIN = float(-3/4*sp.pi)
q3_MAX = float(3/4*sp.pi)

# Cartesian limits
x_MIN = -1.
x_MAX = 1.
y_MIN = -1.
y_MAX = 1.
z_MIN = -1.
z_MAX = 1.

# Drone dimensions
drone_body=(0.35, 0.15, 0.045)
drone_rotor=(0.2286, 0.245, 0.195)

# Mode
mode = "cartesianposition"

if __name__ == "__main__":
    # Joint states
    q1, q2, q3 = sp.symbols('q_1 q_2 q_3', real=True)
    #x, y, z, yaw, pitch, roll, q1, q2, q3 = sp.symbols('x_b y_b z_b yaw pitch roll q_1 q_2 q_3', real=True)

    # Parameters
    L1, L2, L3 = sp.symbols('L_1 L_2 L_3', real=True, positive=True)

    # --- FORWARD KINEMATICS DERIVATION ---
    #T_Ib = sp.Matrix([[sp.cos(yaw)*sp.cos(pitch), sp.cos(yaw)*sp.sin(pitch)*sp.sin(roll)-sp.sin(yaw)*sp.cos(roll), sp.cos(yaw)*sp.sin(pitch)*sp.cos(roll)+sp.sin(yaw)*sp.sin(roll), x],
    #                  [sp.sin(yaw)*sp.cos(pitch), sp.sin(yaw)*sp.sin(pitch)*sp.sin(roll)+sp.cos(yaw)*sp.cos(roll), sp.sin(yaw)*sp.sin(pitch)*sp.cos(roll)-sp.cos(yaw)*sp.sin(roll), y],
    #                  [-sp.sin(pitch), sp.cos(pitch)*sp.sin(roll), sp.cos(pitch)*sp.cos(roll), z],
    #                  [0, 0, 0, 1]])
    
    T_b0 = sp.Matrix([[0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])

    T_01 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0],
                    [sp.sin(q1),  sp.cos(q1), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    T_12 = sp.Matrix([[-sp.cos(q2), sp.sin(q2), 0, -L1],
                    [0, 0, 1, 0],
                    [sp.sin(q2), sp.cos(q2), 0, 0],
                    [0, 0, 0, 1]])

    T_23 = sp.Matrix([[-sp.cos(q3), sp.sin(q3), 0, L2],
                    [0, 0, 1, 0],
                    [sp.sin(q3), sp.cos(q3), 0, 0],
                    [0, 0, 0, 1]])

    T_3e = sp.Matrix([[0, 0, -1, -L3],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

    T_be = sp.trigsimp(sp.expand(T_b0*T_01*T_12*T_23*T_3e))
     
    # Set parameters
    T_be = T_be.subs([(L1, L1_length), (L2, L2_length), (L3, L3_length)])
    
    AMparameters = dict({L1:L1_length,
                         L2:L2_length,
                         L3:L3_length})
    
    AerialManipulator = Model(transformation=T_be, variables = (q1, q2, q3), parameters=AMparameters)
    AerialManipulator.getAnalyticalJacobian()
    
    solver = KinematicsSolver(AerialManipulator.transformation, (q1, q2, q3), AerialManipulator.AnalyticalJacobian)
    
    # Set up workspace analysis -  joint limits [lower, upper]
    joint_limits=np.array([[q1_MIN, q1_MAX], # q1
                           [q2_MIN, q2_MAX], # q2
                           [q3_MIN, q3_MAX],]) # q3
    
    ws_limits=np.array([[x_MIN, x_MAX],
                        [y_MIN, y_MAX],
                        [z_MIN, z_MAX]])
    
    #data = solver.solve(limits = ws_limits, steps=steps, space=mode)
    joint_positions = solver.solveInverseKinematics(target_pos=np.array([0.0, 0.5, 0.0]))
    print(joint)
    
    visualise(data, body=drone_body, rotor=drone_rotor, space=mode)
    
    