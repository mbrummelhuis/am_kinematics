import sympy as sp
import sympy.physics.mechanics as me

class SimpleAerialManipulator:
    ''' Simple Aerial Manipulator System
    Dynamics for a simple 2D aerial manipulator with 2 rotors and one DoF manipulator attached at the center of mass.
    '''
    def __init__(self):
        #elf.model_parameters = {'mb': mb, 'mm': mm, 'Ib': Ib, 'Im': Im, 'l': l, 'g': g} # Model parameter values
        pass

    def dynamics(self):
        pass

    def derive_dynamics(self):
        ''' Derive the dynamics of the system
        '''
        mb, mm, Ib, Im, l, r, g = sp.symbols('mb mm Ib Im l r g', real=True, positive=True) # Model parameters (stationary)
        T1, T2, tau = sp.symbols('T1 T2 tau', real=True) # Control inputs

        xb, yb, pb, q = me.dynamicsymbols('xb yb pb q') # Generalized coordinates
        xbd, ybd, pbd, qd = me.dynamicsymbols('xb yb pb q', 1) # Generalized velocities
        xbdd, ybdd, pbdd, qdd = me.dynamicsymbols('xb yb pb q', 2) # Generalized accelerations

        # Kinetic and potential energy
        Kb = 0.5*mb*(xbd**2 + ybd**2) + 0.5*Ib*(pbd**2)
        Km = 0.5*mm*(xbd**2 + ybd**2) + 0.5*mm*(0.5*l*(qd+pbd))**2 + 0.5*Im*(qd+pbd)**2
        T = Kb + Km
        V = mb*g*yb + mm*g*(yb - 0.5*l*sp.cos(q))

        # Lagrangian
        L = T - V
        F = [sp.sin(pb)*(T1+T2), sp.cos(pb)*(T1+T2), -r*T1+r*T2, tau]
        LM = me.LagrangesMethod(L, [xb, yb, pb, q], forcelist=F)
        EoMs = LM.form_lagranges_equations()
        sp.pprint(EoMs)

        # Extracting matrices
        mass_matrix = EoMs.subs([(xbd, 0), (ybd, 0), (pbd, 0), (qd, 0)])
        sp.pprint(mass_matrix)

        coriolis_matrix = EoMs.subs([(xbdd, 0), (ybdd, 0), (pbdd, 0), (qdd, 0)])
        sp.pprint(coriolis_matrix)

        gravity_vector = EoMs.subs([(xbdd, 0), (ybdd, 0), (pbdd, 0), (qdd, 0)])
    
    def set_model_parameters(self, model_parameters):
        self.model_parameters = model_parameters
    
    def set_initial_conditions(self, initial_conditions):
        self.initial_conditions = initial_conditions

if __name__=="__main__":
    system = SimpleAerialManipulator()
    system.derive_dynamics()