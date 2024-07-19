import sympy as sp

from drone import DroneVisualModel
from kinematics import KinematicsSolver

def deriveTransformation():
    # Joint states
    x, y, z, yaw, pitch, roll, q1, q2, q3 = sp.symbols('x_b y_b z_b yaw pitch roll q_1 q_2 q_3', real=True)

    # Parameters
    L1, L2, L3 = sp.symbols('L_1 L_2 L_3', real=True, positive=True)

    # --- FORWARD KINEMATICS DERIVATION ---
    T_Ib = sp.Matrix([[sp.cos(yaw)*sp.cos(pitch), sp.cos(yaw)*sp.sin(pitch)*sp.sin(roll)-sp.sin(yaw)*sp.cos(roll), sp.cos(yaw)*sp.sin(pitch)*sp.cos(roll)+sp.sin(yaw)*sp.sin(roll), x],
                      [sp.sin(yaw)*sp.cos(pitch), sp.sin(yaw)*sp.sin(pitch)*sp.sin(roll)+sp.cos(yaw)*sp.cos(roll), sp.sin(yaw)*sp.sin(pitch)*sp.cos(roll)-sp.cos(yaw)*sp.sin(roll), y],
                      [-sp.sin(pitch), sp.cos(pitch)*sp.sin(roll), sp.cos(pitch)*sp.cos(roll), z],
                      [0, 0, 0, 1]])
    
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

    T_Ie = sp.trigsimp(sp.expand(T_Ib*T_b0*T_01*T_12*T_23*T_3e))
    
    # Set parameters
    L1_length = 0.11
    L2_length = 0.25
    L3_length = 0.25
    T_Ie = T_Ie.subs([(L1, L1_length), (L2, L2_length), (L3, L3_length)])

    return T_Ie