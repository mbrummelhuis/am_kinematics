import sympy as sp
import numpy as np
from tqdm import tqdm
import time
import itertools

class KinematicsSolver:
    """KinematicsSolver class
    Class that contains several solving methods for manipulator kinematics.
    """
    def __init__(self, transformation: sp.Expr, variables: tuple ,jacobian: sp.Expr=None):
        """solver class for the forward and inverse kinematics

        Args:
            transformation (Sympy Expr): 4x4 homogeneous transformation matrix describing the 
                                         end-effector pose in the world frame
            variables (tuple): Ordered tuple of joint variables
            jacobian (Sympy Expr): 6xN jacobian matrix of the aerial manipulator with N the
                                   total degrees of freedom
        """
        self.transformation_matrix = transformation

        self.analytical_jacobian = jacobian

        self.joint_variables = variables
        self.dofs = len(self.joint_variables)
        
        self.FKresult = np.empty(6) # For saving results of forward kinematics
        self.IKresult = np.empty(self.dofs) # For saving results of inverse kinematics

    def __forwardPoseKinematics(self, configuration: np.array):
        """Forward pose kinematics function

        Args:
            configuration (Numpy array): List of desired joint positions

        Returns:
            int: 0 in case of succesful termination
        """
        # Check if configuration has same length as joint variables:
        if len(configuration)!= self.dofs:
            print("[KINEMATICS SOLVER] Error: Given configuration not same length as number of DOFs")
            return

        joint_commands = list(zip(self.joint_variables, configuration))

        #Linear
        self.FKresult[0] = self.transformation_matrix[0,3].subs(joint_commands)
        self.FKresult[1] = self.transformation_matrix[1,3].subs(joint_commands)
        self.FKresult[2] = self.transformation_matrix[2,3].subs(joint_commands)

        # Angular
        self.FKresult[3] = sp.atan2(self.transformation_matrix[2,1].subs(joint_commands), self.transformation_matrix[2,2].subs(joint_commands))
        self.FKresult[4] = sp.asin(self.transformation_matrix[2,0].subs(joint_commands))
        self.FKresult[5] = sp.atan2(self.transformation_matrix[1,0].subs(joint_commands), self.transformation_matrix[0,0].subs(joint_commands))
        return 0
    
    def __forwardPositionKinematics(self, configuration: np.array):
        """Forward position kinematics function

        Args:
            configuration (Numpy array): List of desired joint positions

        Returns:
            int: 0 in case of succesful termination
        """
        # Check if configuration has same length as joint variables:
        if len(configuration)!= self.dofs:
            print("[KINEMATICS SOLVER] Error: Given configuration not same length as number of DOFs")
            return

        joint_commands = list(zip(self.joint_variables, configuration))

        #Linear
        self.FKresult[0] = self.transformation_matrix[0,3].subs(joint_commands)
        self.FKresult[1] = self.transformation_matrix[1,3].subs(joint_commands)
        self.FKresult[2] = self.transformation_matrix[2,3].subs(joint_commands)
        
        return 0

    def __inversePoseKinematics(self, target_pose: np.array, initial_guess: np.array = None, tolerance: float = 1e-6, max_iterations: int = 10000):
        """Inverse full pose kinematics function

        Args:
            target_pose (np.array): Desired end-effector pose to find joint configuration for
            initial_guess (np.array): First guess of the joint configuration, a good guess means faster convergence
            tolerance (float): Value under which the L2 norm of the pose is considered converged
            max_iterations (int): Value over which the pose is considered unreachable.
        
        Returns:
            int: 0 in case of succesfull termination, -1 in case of nonconvergence
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
        self.__forwardPoseKinematics(configuration=initial_guess)
        error = target_pose - self.FKresult
        evaluated_jacobian = self.analytical_jacobian.subs(list(zip(self.joint_variables, initial_guess)))
        previous_guess = initial_guess
        counter = 0

        # Running the algorithm
        while np.linalg.norm(error) > tolerance:
            pseudoinverse = evaluated_jacobian.transpose()*((evaluated_jacobian*evaluated_jacobian.transpose()).inv())
            guess = previous_guess + pseudoinverse*error

            self.__forwardPoseKinematics(configuration=guess)
            error = target_pose - self.FKresult
            evaluated_jacobian = self.analytical_jacobian.subs(list(zip(self.joint_variables, guess)))

            previous_guess = guess
            counter+=1
            if counter > max_iterations:
                return -1
        
        self.IKresult = guess
        
        # Results statement
        print(f"[KINEMATICS SOLVER] Inverse kinematics converged with error {error} in {counter} iterations")
        print(f"[KINEMATICS SOLVER] Desired pose: {target_pose}")
        print(f"[KINEMATICS SOLVER] Achieved pose: {self.FKresult}")
        return 0
    
    def __inversePositionKinematics(self, target_position: np.array, initial_guess: np.array = None, tolerance: float = 1e-6, max_iterations: int = 10000):
        """Inverse position kinematics function

        Args:
            target_position (np.array): Desired end-effector pose to find joint configuration for
            initial_guess (np.array): First guess of the joint configuration, a good guess means faster convergence
            tolerance (float): Value under which the L2 norm of the pose is considered converged
            max_iterations (int): Value over which the pose is considered unreachable.
        
        Returns:
            int: 0 in case of succesfull termination, -1 in case of nonconvergence
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
        self.__forwardPositionKinematics(configuration=initial_guess)
        error = target_position - self.FKresult[0:3]
        evaluated_jacobian = self.analytical_jacobian[0:3].subs(list(zip(self.joint_variables, initial_guess)))
        previous_guess = initial_guess
        counter = 0

        # Running the algorithm
        while np.linalg.norm(error) > tolerance:
            pseudoinverse = evaluated_jacobian.transpose()*((evaluated_jacobian*evaluated_jacobian.transpose()).inv())
            guess = previous_guess + pseudoinverse*error

            self.__forwardPositionKinematics(configuration=guess)
            error = target_position - self.FKresult[0:3]
            evaluated_jacobian = self.analytical_jacobian[0:3].subs(list(zip(self.joint_variables, guess)))

            previous_guess = guess
            counter+=1
            if counter > max_iterations:
                return -1
        
        self.IKresult[0:3] = guess
        
        # Results statement
        print(f"[KINEMATICS SOLVER] Inverse kinematics converged with error {error} in {counter} iterations")
        print(f"[KINEMATICS SOLVER] Desired pose: {target_position}")
        print(f"[KINEMATICS SOLVER] Achieved pose: {self.FKresult}")
        return 0        

    def analyseWorkspace(self,  limits: np.array, steps: int=10, space: str='jointspace'):
        """Run workspace analysis

        Args:
            limits (np.array): A Nx2 array specifying the [min, max] joint limits for all N joints
            steps (int, optional): Number of steps between limits. Defaults to 10.
            space (str, optional): Select joint space or cartesian space analysis. Defaults to 'jointspace'.

        Returns:
            np.array: Data array of Nx7 with for every checked pose on columns 0 to 6, and on column 6 a 0
                        for nonreachable position and 1 for reachable position.
        """
        if space == 'jointspace':
            # Use forward kinematics
            print("[KINEMATICS SOLVER] Workspace analysis in joint space using forward kinematics")

            # Build configuration space to explore
            configuration_space = np.empty((self.dofs, steps), dtype=float)

            for i in range(self.dofs):
                configuration_space[i,:]= np.linspace(limits[i,0], limits[i,1], steps)
            
            config_combinations = itertools.product(*configuration_space)
            num_combinations = self.dofs ** steps

            # Set start time 
            start = time.time()

            # --- MAIN LOOP ---
            # Initialize data array
            # Get the lengths of each row
            data = np.empty((3, num_combinations))
            counter = 0

            print("[KINEMATICS SOLVER] Starting loop")
            for configuration in tqdm(config_combinations):
                float_configuration = tuple(float(x) for x in configuration)
                self.__forwardKinematics(configuration=float_configuration)
                data[:,counter] = self.FKresult[0:3]
                counter += 1

            # Set end time
            end = time.time()

            # Display total elapsed time message
            time_elapsed = end-start
            print("[KINEMATICS SOLVER] Workspace finished calculating in ", time_elapsed, " seconds")
            return data

        elif space == 'cartesianposition':
            # Use inverse kinematics
            print("[KINEMATICS SOLVER] Workspace analysis in cartesian space using inverse position kinematics")
            
            # Build configuration space to explore
            configuration_space = np.empty((3, steps))
            
            for i in range(3):
                configuration_space[i,:]=np.linspace(limits[i,0], limits[i,1])
            
            position_combinations = itertools.product(*configuration_space)
            num_combinations = steps ** 3
            
            # --- MAIN LOOP ---
            start = time.time()
            
            # Create data array
            data = np.empty((num_combinations, 4))
            counter = 0
            
            print("[KINEMATICS SOLVER] Starting loop")
            for checked_position in tqdm(position_combinations):
                # Check if the pose gets a solution
                res = self.__inversePositionKinematics(target_position=checked_position, initial_guess=np.zeros(self.dofs))
                data[counter, 0,6] = self.IKresult
                data[counter, 6] = res
                counter+=1
                
            end = time.time()
            
            time_elapsed = end-start
            print(f"[KINEMATICS SOLVER] Workspace finished calculating in {time_elapsed} seconds")
            return data

        elif space == 'cartesianpose':
            # Use inverse kinematics
            print("[KINEMATICS SOLVER] Workspace analysis in cartesian space using inverse pose kinematics")
            
            # Build configuration space to explore
            configuration_space = np.empty((6, steps))
            
            for i in range(6):
                configuration_space[i,:]=np.linspace(limits[i,0], limits[i,1])
            
            pose_combinations = itertools.product(*configuration_space)
            num_combinations = steps ** 6
            
            # --- MAIN LOOP ---
            start = time.time()
            
            # Create data array
            data = np.empty((num_combinations, 7))
            counter = 0
            
            print("[KINEMATICS SOLVER] Starting loop")
            for checked_pose in tqdm(pose_combinations):
                # Check if the pose gets a solution
                res = self.__inversePoseKinematics(target_pose=checked_pose, initial_guess=np.zeros(self.dofs))
                data[counter, 0,6] = self.IKresult
                data[counter, 6] = res
                counter+=1
                
            end = time.time()
            
            time_elapsed = end-start
            print(f"[KINEMATICS SOLVER] Workspace finished calculating in {time_elapsed} seconds")
            return data
                