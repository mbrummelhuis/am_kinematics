import sympy as sp
import sympy.physics.mechanics as me
import pickle
import os

class SimpleAerialManipulator:
    ''' Simple Aerial Manipulator System
    Dynamics for a simple 2D aerial manipulator with 2 rotors and one DoF manipulator attached at the center of mass.
    '''
    def __init__(self, state : dict, inputs : dict, parameters : dict):
        self.model_parameters = parameters  # Model parameter values
        self.state = state                  # State variables
        self.inputs = inputs                # Input variables

        self.dynamics_function_full = None  # Full dynamics function
        self.dynamics_function = None       # Dynamics function with parameters

    ''' Calculate the state derivative using system dynamics
    '''
    def dynamics(self, x, u):
        self.state_derivative = self.dynamics_function(x, u)
        return self.state_derivative
    
    ''' Load the lambdified model from a file
    '''
    def load_model(self, model_name):
        model_path = os.path.join(os.path.dirname(__file__), model_name+'.pkl')
        print('Loading model from:', model_path)
        with open(model_path, 'rb') as file:
            self.dynamics_function_full = pickle.load(file)
        print('Model loaded successfully: ', self.dynamics_function_full)
        self.set_model_parameters()
    
    ''' Set model parameters and re-lambdify to obtain simulation function
    '''
    def set_model_parameters(self):
        mm = sp.symbols('m_m')
        dynamics_non_parametrised = self.dynamics_function_full[4].subs(mm, 0.5)
        print(type(list(dynamics_non_parametrised.free_symbols)[0]))
        print(type(mm))
        self.dynamics_function = sp.lambdify((self.state, self.inputs), dynamics_non_parametrised, 'numpy')

if __name__=="__main__":
    system = SimpleAerialManipulator()
    system.derive_dynamics()