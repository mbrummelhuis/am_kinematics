import sympy as sp
import numpy as np
import tqdm
import time

class KinematicsSolver:
    """KinematicsSolver class
    Class that contains several solving methods for manipulator kinematics.
    """
    def __init__(self, transformation: sp.Expr, jacobian: sp.Expr=None):
        """solver class for the forward and inverse kinematics

        Args:
            transformation (Sympy Expr): 4x4 homogeneous transformation matrix describing the 
                                         end-effector pose in the world frame
        """
        self.transformation_matrix = transformation

        self.analytical_jacobian = jacobian

        self.joint_variables = self.transformation_matrix.free_symbols
        self.dofs = len(self.joint_variables)

    def __forwardKinematics(self, configuration: list):
        """Forward kinematics function

        Args:
            configuration (list): List of desired joint positions

        Returns:
            list: List with full pose result
        """
        # Check if configuration has same length as joint variables:
        if len(configuration)!= self.dofs:
            print("[KINEMATICS SOLVER] Error: Given configuration not same length as number of DOFs")
            return

        joint_commands = zip(self.joint_variables, configuration)
        result = []

        #Linear
        result[0] = self.transformation_matrix[0,3].subs(joint_commands)
        result[1] = self.transformation_matrix[1,3].subs(joint_commands)
        result[2] = self.transformation_matrix[2,3].subs(joint_commands)

        # Angular
        result[3] = sp.atan2(self.transformation_matrix[2,1].subs(joint_commands), self.transformation_matrix[2,2].subs(joint_commands))
        result[4] = sp.asin(self.transformation_matrix[2,0].subs(joint_commands))
        result[5] = sp.atan2(self.transformation_matrix[1,0].subs(joint_commands), self.transformation_matrix[0,0].subs(joint_commands))
        return result

    def __inverseKinematics(self, pose: list):
        """Inverse Kinematics function

        Args:
            pose (list): Desired end-effector pose
        """
        # Check if analytical jacobian is provided
        if self.analytical_jacobian is None:
            print("[KINEMATICS SOLVER] Error: Please provide the analytical Jacobian")
            return

        # Implement the inverse kinematics using Newton-Raphson such as in Modern Robotics


    def analyseWorkspace(self, space='jointspace'):
        if space is 'jointspace':
            # Use forward kinematics
            print("Workspace analysis in joint space")

        elif space is 'cartesian':
            # Use inverse kinematics
            print("Workspace analysis in cartesian space")