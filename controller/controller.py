import dill
import sympy as sp
import numpy as np


class Controller:
    def __init__(self, system_state_vars, input_vars, controller_params):
        self.system_state_vars = system_state_vars
        self.input_vars = input_vars
        self.controller_params = controller_params

        self.controller_params_list = [
            'dt',           # Time step [s]
            'alpha',        # EWMA smoothing factor
            'min_thrust',   # Minimum rotor thrust
            'max_thrust',   # Maximum rotor thrust
            'Kp',           # Proportional gain matrix
            'Ki',           # Integral gain matrix
            'Kd',           # Derivative gain matrix
            'mixing_matrix' # Mixing matrix
        ]
    
    ''' Calculate the control input using the current state
    '''
    def get_control_input(self, state, reference):
        xi = state[0:3]
        xi_dot = state[3:]
        # Evaluate Jacobian at state
        J_eval = self.J_e(xi)
        # Calculate pseudoinverse
        J_pinv = np.linalg.pinv(J_eval)
        # Calculate state velocity references
        xi_dot_ref = J_pinv*reference

        # Pass through PID and mixing
        xi_dot_err = xi_dot_ref - xi_dot
        u = self.PID(xi_dot_err)
        motor_inputs = self.mixing(u)

        # Return clipped rotor thrusts
        return np.clip(motor_inputs, self.controller_params['min_thrust'], self.controller_params['max_thrust'])
    
    def PID(self, error):
        # Calculate error integral
        self.error_integral += error*self.controller_params['dt']

        # Calculate error derivative with EWMA smoothing
        self.error_derivative = self.controller_params['alpha']*error + (1-self.controller_params['alpha'])*self.error_derivative

        # Calculate control input
        u = self.controller_params['Kp']*error + self.controller_params['Ki']*self.error_integral + self.controller_params['Kd']*self.error_derivative
        
        return u
        
    def mixing(self,u):
        # Calculate rotor thrusts
        motor_inputs = np.dot(self.controller_params['mixing_matrix'], u)
        return motor_inputs

    def load_lambified_jacobian(self, jacobian_name):
        self.J_e_full = dill.load(open(jacobian_name+'.pkl', 'rb'))
    
    def set_jacobian_parameters(self, jacobian_parameters):
        self.jacobian_parameters = jacobian_parameters
        J_non_parametrised = self.J_e_full.subs(self.jacobian_parameters)
        self.J_e = sp.lambdify((self.system_state_vars, self.input_vars), J_non_parametrised, 'numpy')