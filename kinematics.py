import sympy as sp
import numpy as np
import tqdm
import time


class Manipulator:
    def __init__(self, qmin = None, qmax = None, qtypes = None):
        self.q = None # Generate joints from num_joints
        
        # Joint types
        # If none given, assume all Revolute 'R'
        # Options: Free 'F' (in 6DoF), Revolute 'R' (1 DoF), Universal 'U' (2 DoF), Prismatic 'P' (1 linear DoF)
        if qtypes == None:
            self.qtypes = ['R' for q in self.q]
        else:
            self.qtypes = qtypes
        
        # Joint limits
        self.qmin = qmin
        self.qmax = qmax
        self.params = None
        self.kinematics_transformation = None
        
    
    def kinematics(self):
        # Assign variables
        # Joint states
        q1, q2, q3 = sp.symbols('q_1 q_2 q_3', real=True)
        
        # Parameters
        self.params[0], self.params[1], self.params[2] = sp.symbols('L_1 L_2 L_3', real=True, positive=True)
        
        # --- FORWARD KINEMATICS DERIVATION ---
        T_b0 = sp.Matrix([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

        T_01 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0],
                        [sp.sin(q1),  sp.cos(q1), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        T_12 = sp.Matrix([[-sp.cos(q2), sp.sin(q2), 0, -self.params[0]],
                        [0, 0, 1, 0],
                        [sp.sin(q2), sp.cos(q2), 0, 0],
                        [0, 0, 0, 1]])

        T_23 = sp.Matrix([[-sp.cos(q3), sp.sin(q3), 0, self.params[1]],
                        [0, 0, 1, 0],
                        [sp.sin(q3), sp.cos(q3), 0, 0],
                        [0, 0, 0, 1]])

        T_3e = sp.Matrix([[0, 0, -1, -self.params[2]],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]])

        self.kinematics_transformation = sp.trigsimp(sp.expand(T_b0*T_01*T_12*T_23*T_3e))
    
    def analyseWorkspace(self):
        # Set start time 
        start = time.time()

        # --- MAIN LOOP ---
        steps = 20
        data = np.zeros((3,steps**3))
        counter = 0

        print("Starting loop")
        for q1step in tqdm(np.linspace(q1_MIN, q1_MAX, steps)):
            for q2step in np.linspace(q2_MIN, q2_MAX, steps):
                for q3step in np.linspace(q3_MIN, q3_MAX, steps):
                    data[0,counter] = T_be[0,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                    data[1,counter] = T_be[1,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                    data[2,counter] = T_be[2,3].subs([(q1, q1step), (q2, q2step), (q3, q3step)])
                    counter += 1

        # Set end time
        end = time.time()

        # Display total elapsed time message
        time_elapsed = end-start
        print("Workspace finished calculating in ", time_elapsed, " seconds")