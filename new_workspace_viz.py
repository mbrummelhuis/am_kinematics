#---------------------------
# Workspace analysis for the proposed RRR manipulator on the omnidrone
# redesign. 
#--------------------------- 

# Preamble
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.spatial import ConvexHull

from tqdm import tqdm

from workspace_analysis.drone import DroneVisualModel
#from kinematics import KinematicsSolver

sp.init_printing(use_unicode=True)

def forward_kinematics_2d(q, L, plane="XZ"):
    """
    2D forward kinematics projection for schematic drawing.
    q = [q1, q2, q3] joint angles (rad)
    L = [L1, L2, L3] link lengths
    plane = "XZ" (side view) or "YZ" (front view)
    
    Returns list of (u, v) joint positions in the chosen plane.
    """
    q1, q2, q3 = q
    L1, L2, L3 = L

    joints = [(0,0)]  # pivot at origin

    # Local cumulative angle
    theta1 = q1
    theta2 = q1 + q2
    theta3 = q1 + q2 + q3

    if plane == "XZ":  # side view projection
        # Z is vertical, X horizontal
        x1 = joints[-1][0] # x doesn't change with changing q1
        z1 = joints[-1][1] - L1 * np.cos(theta1)
        joints.append((x1, z1))

        x2 = joints[-1][0] + L2 * np.sin(theta2)
        z2 = joints[-1][1] - L2 * np.cos(theta2)
        joints.append((x2, z2))

        x3 = joints[-1][0] + L3 * np.sin(theta3)
        z3 = joints[-1][1] - L3 * np.cos(theta3)
        joints.append((x3, z3))

    elif plane == "YZ":  # front view projection
        # Z is vertical, Y horizontal
        y1 = joints[-1][0] + L1 * np.sin(theta1)
        z1 = joints[-1][1] - L1 * np.cos(theta1)
        joints.append((y1, z1))

        y2 = joints[-1][0] + L2 * np.sin(theta2)
        z2 = joints[-1][1] - L2 * np.cos(theta2)
        joints.append((y2, z2))

        y3 = joints[-1][0] + L3 * np.sin(theta3)
        z3 = joints[-1][1] - L3 * np.cos(theta3)
        joints.append((y3, z3))

    return joints

def draw_drone_and_manipulator(ax, view="side", q=[0,0,0], L=[0.11, 0.33, 0.273]):
    """
    Draws a 2D schematic of the drone (body + rotors) and manipulator.
    - view: "front" (YZ plane) or "side" (XZ plane)
    - q: [q1,q2,q3] joint configuration (rad) for manipulator (only used in side view)
    - L: [L1,L2,L3] link lengths
    """
    # --- Drone dimensions ---
    body_length = 0.35     # length of fuselage bar (m)
    body_width = 0.4        # width of the fuselage bar (m)
    rotor_radius = 0.2286/2.   # radius of rotor disc (m)
    bar_length = 0.05      # short vertical bar connecting body to rotor plane (m)
    rotor_thickness = 0.02 # ellipse "thinness" for side/front rotor view (m)

    font = {"fontname": "Times New Roman"}  # consistent style

    # =============================
    #  DRONE BODY + ROTORS
    # =============================
    if view == "front":
        # Body drawn along Y axis at Z=0
        ax.plot([-body_length/2, body_length/2], [0,0], 'k-', linewidth=3)

        # Rotors with vertical bars pointing UP
        for y in [-body_length/2, body_length/2]:
            ax.plot([y,y], [0, bar_length], 'k-', linewidth=3)
            rotor = patches.Ellipse((y, bar_length),
                                    width=2*rotor_radius, height=rotor_thickness,
                                    fill=False, color='k')
            ax.add_patch(rotor)

    elif view == "side":
        # Body drawn along X axis at Z=0
        ax.plot([-body_length/2, body_length/2], [0,0], 'k-', linewidth=3)

        # Rotors with vertical bars pointing UP
        for x in [-body_length/2, body_length/2]:
            ax.plot([x,x], [0, bar_length], 'k-', linewidth=3)
            rotor = patches.Ellipse((x, bar_length),
                                    width=2*rotor_radius, height=rotor_thickness,
                                    fill=False, color='k')
            ax.add_patch(rotor)

    # =============================
    #  PIVOT JOINT
    # =============================
    pivot = plt.Circle((0,0), 0.02, color='red', zorder=15)
    ax.add_patch(pivot)

    # =============================
    #  MANIPULATOR
    # =============================
    plane = "XZ" if view=="side" else "YZ"
    joints = forward_kinematics_2d(q, L, plane=plane)

    for i in range(len(joints)-1):
        p1, p2 = joints[i], joints[i+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)

    for (u,v) in joints:
        circ = plt.Circle((u,v), 0.015, color='blue', zorder=16)
        ax.add_patch(circ)

    # =============================
    #  COORDINATE FRAME (arrows)
    # =============================
    arrow_len = 0.1
    text_fontsize = 14
    arrow_offset = -0.038
    if view == "front":  
        # Y → right, Z → down
        ax.arrow(0,0, arrow_len,0, head_width=0.03, head_length=0.03, linewidth=2.5, fc='g', ec='g',zorder=10)
        ax.text(arrow_len-0.01, arrow_offset, "Y", color='g', va='center', **font, fontsize=text_fontsize)

        ax.arrow(0,0, 0,-arrow_len, head_width=0.03, head_length=0.03, linewidth=2.5, fc='b', ec='b',zorder=10)
        ax.text(arrow_offset, -arrow_len+0.01, "Z", color='b', ha='center', **font, fontsize=text_fontsize)

    elif view == "side":
        # X → right, Z → down
        ax.arrow(0,0, arrow_len,0, head_width=0.03, head_length=0.03, linewidth=2.5,fc='r', ec='r',zorder=10)
        ax.text(arrow_len-0.01, arrow_offset, "X", color='r', va='center', **font, fontsize=text_fontsize)

        ax.arrow(0,0, 0,-arrow_len, head_width=0.03, head_length=0.03, linewidth=2.5,fc='b', ec='b',zorder=10)
        ax.text(arrow_offset, -arrow_len+0.01, "Z", color='b', ha='center', **font,fontsize=text_fontsize)


def workspace_analysis_jointspace():
    # Joint states
    q1, q2, q3 = sp.symbols('q_1 q_2 q_3', real=True)

    # Joint limits
    q1_MIN = float(-sp.pi)
    q1_MAX = float(sp.pi)
    q2_MIN = float(-1/6*sp.pi)
    q2_MAX = float(1/6*sp.pi)
    q3_MIN = float(-1.85)
    q3_MAX = float(1.85)

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
        L1_length = 0.110
        L2_length = 0.330
        L3_length = 0.273
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
    # --- Extract end-effector positions ---
    points = data[:3, :].T   # shape (N, 3), columns = [x, y, z]

    plt.figure(figsize=(10,5))

    # --- FRONT VIEW (Y-Z) ---
    points_front = points[:, [1, 2]]  # take Y and Z
    hull_front = ConvexHull(points_front)

    plt.subplot(1,2,1)
    # ... plot hull + points ...
    draw_drone_and_manipulator(plt.gca(), view="front")
    plt.axis('equal'); plt.grid(True); plt.title("Front View (YZ)")
    for simplex in hull_front.simplices:
        plt.plot(points_front[simplex,0], points_front[simplex,1], color='cyan', linewidth=2)
    plt.scatter(points_front[:,0], points_front[:,1], s=5, alpha=0.2, color='gray')
    plt.xlabel('Y [m]')
    plt.ylabel('Z [m]')
    plt.title('Workspace Front View')
    plt.axis('equal')
    plt.grid(True)

    # --- SIDE VIEW (X-Z) ---
    points_side = points[:, [0, 2]]  # take X and Z
    hull_side = ConvexHull(points_side)

    plt.subplot(1,2,2)
    # ... plot hull + points ...
    draw_drone_and_manipulator(plt.gca(), view="side")
    plt.axis('equal'); plt.grid(True); plt.title("Side View (XZ)")
    for simplex in hull_side.simplices:
        plt.plot(points_side[simplex,0], points_side[simplex,1], color='cyan', linewidth=2)
    plt.scatter(points_side[:,0], points_side[:,1], s=5, alpha=0.2, color='gray')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Workspace Side View')
    plt.axis('equal')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    workspace_analysis_jointspace()