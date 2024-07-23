import sympy as sp
import numpy as np

class Model:
    def __init__(self, transformation:sp.Expr, variables:tuple, parameters:dict = None):
        self.transformation = transformation
        self.states = variables
        self.params = parameters
        self.FK = np.empty(6)
    
    def getFKfunction(self):
        #Linear
        self.FK[0] = self.transformation[0,3]
        self.FK[1] = self.transformation[1,3]
        self.FK[2] = self.transformation[2,3]

        # Angular
        self.FK[3] = sp.atan2(self.transformation[2,1], self.transformation[2,2])
        self.FK[4] = sp.asin(self.transformation[2,0])
        self.FK[5] = sp.atan2(self.transformation[1,0], self.transformation[0,0])        
    
    def getAnalyticalJacobian(self):
        # Check if FK is empty (i.e. not derived yet)
        if np.any(self.FK):
            self.getFKfunction()
        
        # Derive analytical Jacobian
        self.AnalyticalJacobian = self.FK.jacobian(self.states)
        
    def getGeometricJacobian(self):
        pass