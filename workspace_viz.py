#---------------------------
# Workspace analysis for the proposed RRR manipulator on the omnidrone
# redesign. 
#--------------------------- 

# Preamble
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from drone import DroneVisualModel
#from kinematics import KinematicsSolver

sp.init_printing(use_unicode=True)

def workspace_analysis_jointspace():
    # Joint states
    q1, q2, q3 = sp.symbols('q_1 q_2 q_3', real=True)

    # Joint limits
    q1_MIN = float(-sp.pi)
    q1_MAX = float(sp.pi)
    q2_MIN = float(-1/8*sp.pi)
    q2_MAX = float(1/8*sp.pi)
    q3_MIN = float(-3/4*sp.pi)
    q3_MAX = float(3/4*sp.pi)

    # Parameters
    L1, L2, L3 = sp.symbols('L_1 L_2 L_3', real=True, positive=True)

    # --- FORWARD KINEMATICS DERIVATION ---
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

    # Test
    test = False
    if test:
        q1set = 0
        q2set = 0
        q3set = 1/2*sp.pi
        x = T_be[0,3].subs([(q1, q1set), (q2, q2set), (q3, q3set)])
        y = T_be[1,3].subs([(q1, q1set), (q2, q2set), (q3, q3set)])
        z = T_be[2,3].subs([(q1, q1set), (q2, q2set), (q3, q3set)])
        print([x,y,z])

    # Set parameters
    set_params = True
    if set_params:
        L1_length = 0.11
        L2_length = 0.25
        L3_length = 0.25
        T_be = T_be.subs([(L1, L1_length), (L2, L2_length), (L3, L3_length)])
        print("Parameters set")

    # Set start time 
    start = time.time()

    # --- MAIN LOOP ---
    steps = 10
    data = np.zeros((6,steps**3))
    counter = 0

    print("Starting loop")
    for q1step in tqdm(np.linspace(q1_MIN, q1_MAX, steps)):
        for q2step in np.linspace(q2_MIN, q2_MAX, steps):
            for q3step in np.linspace(q3_MIN, q3_MAX, steps):
                # Linear
                data[0,counter] = T_be[0,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                data[1,counter] = T_be[1,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                data[2,counter] = T_be[2,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                
                # Angular
                data[3,counter] = sp.atan2(T_be[2,1].subs([(q1, q1step), (q2, q2step), (q3, q3step)]), T_be[2,2].subs([(q1, q1step), (q2, q2step), (q3, q3step)]))
                data[4,counter] = sp.asin(T_be[2,0].subs([(q1, q1step), (q2, q2step), (q3, q3step)]))
                data[5,counter] = sp.atan2(T_be[1,0].subs([(q1, q1step), (q2, q2step), (q3, q3step)]), T_be[0,0].subs([(q1, q1step), (q2, q2step), (q3, q3step)]))
                counter += 1

    # Set end time
    end = time.time()

    # Display total elapsed time message
    time_elapsed = end-start
    print("Workspace finished calculating in ", time_elapsed, " seconds")

    # --- VISUALIZATION---
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, projection='3d')
    #ax2 = fig.add_subplot(1,2,2, projection='3d')
    
    ax1.scatter(data[0], data[1], data[2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-1, 1])  # set range for X axis
    ax1.set_ylim([-1, 1])  # set range for Y axis
    ax1.set_zlim([-1, 1])  # set range for Z axis

    #ax2.scatter(data[3], data[4], data[5])
    #ax2.set_xlabel('X')
    #ax2.set_ylabel('Y')
    #ax2.set_zlabel('Z')
    #ax2.set_xlim([float(q2_MIN), float(q2_MAX)])  # set range for X axis
    #ax2.set_ylim([float(q2_MIN), float(q2_MAX)])  # set range for Y axis
    #ax2.set_zlim([float(q2_MIN), float(q2_MAX)])  # set range for Z axis

    dronemodel = DroneVisualModel(body=(0.35, 0.15, 0.045), rotor=(0.2286, 0.245, 0.195))
    dronemodel.drawDroneModel(ax1)

    plt.show()

if __name__ == "__main__":
    workspace_analysis_jointspace()