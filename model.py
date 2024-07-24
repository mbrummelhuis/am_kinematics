import sympy as sp
import numpy as np

class Model:
    def __init__(self, transformation:sp.Expr, variables:tuple, parameters:list = None):
        self.transformation = transformation
        self.states = variables
        self.params = parameters
        self.FK = None
        
        if parameters is not None:
            self.transformation = self.transformation.subs(parameters)
    
    def getFKfunction(self):
        #Linear
        self.FK = sp.Matrix([self.transformation[0,3],
                   self.transformation[1,3],
                   self.transformation[2,3],
                   sp.atan2(self.transformation[2,1], self.transformation[2,2]),
                   sp.asin(self.transformation[2,0]),
                   sp.atan2(self.transformation[1,0], self.transformation[0,0])])  
    
    def getAnalyticalJacobian(self):
        # Check if FK is empty (i.e. not derived yet)
        if self.FK is None:
            self.getFKfunction()
        
        # Derive analytical Jacobian
        self.AnalyticalJacobian = self.FK.jacobian(self.states)
        
    def getGeometricJacobian(self):
        pass