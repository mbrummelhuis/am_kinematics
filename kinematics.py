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
            jacobian (Sympy Expr): 6xN jacobian matrix of the aerial manipulator with N the
                                   total degrees of freedom
        """
        self.transformation_matrix = transformation

        self.analytical_jacobian = jacobian

        self.joint_variables = self.transformation_matrix.free_symbols
        self.dofs = len(self.joint_variables)

    def __forwardKinematics(self, configuration: np.array):
        """Forward kinematics function

        Args:
            configuration (Numpy array): List of desired joint positions

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

    def __inverseKinematics(self, target_pose: np.array, initial_guess: np.array = None, tolerance: float = 1e-6):
        """Inverse Kinematics function

        Args:
            pose (Numpy array): Desired end-effector pose
        
        Returns:
            result (Numpy array): Configuration resulting in the target pose
        """
        # Check if analytical jacobian is provided
        if self.analytical_jacobian is None:
            print("[KINEMATICS SOLVER] Error: Please provide the analytical Jacobian")
            return

        # Initial guess
        if initial_guess is None:
            initial_guess = np.zeros(self.dofs)
        
        # Newton-Raphson implementation
        # Initialization
        error = target_pose - self.__forwardKinematics(initial_guess)
        evaluated_jacobian = self.analytical_jacobian.subs(list(zip(self.joint_variables, initial_guess)))
        previous_guess = initial_guess
        counter = 0

        # Running the algorithm
        while np.linalg.norm(error) > tolerance:
            pseudoinverse = evaluated_jacobian.transpose()*((evaluated_jacobian*evaluated_jacobian.transpose()).inv())
            guess = previous_guess + pseudoinverse*error

            error = target_pose - self.__forwardKinematics(guess)
            evaluated_jacobian = self.analytical_jacobian.subs(list(zip(self.joint_variables, guess)))

            previous_guess = guess
            counter+=1
        
        result = self.__forwardKinematics(guess)
        
        # Results statement
        print(f"[KINEMATICS SOLVER] Inverse kinematics converged with error {error} in {counter} iterations")
        print(f"[KINEMATICS SOLVER] Desired pose: {target_pose}")
        print(f"[KINEMATICS SOLVER] Achieved pose: {result}")
        return result

    def analyseWorkspace(self,  limits: np.array, steps: int=10, space: str='jointspace'):
        if space is 'jointspace':
            # Use forward kinematics
            print("[KINEMATICS SOLVER] Workspace analysis in joint space")

            # Build configuration space to explore
            

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
                        res = self.__forwardKinematics()
                        counter += 1

            # Set end time
            end = time.time()

            # Display total elapsed time message
            time_elapsed = end-start
            print("Workspace finished calculating in ", time_elapsed, " seconds")

        elif space is 'cartesian':
            # Use inverse kinematics
            print("[KINEMATICS SOLVER] Workspace analysis in cartesian space")