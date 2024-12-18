import matplotlib.pyplot as plt
from workspace_analysis.drone import DroneVisualModel
import numpy as np

class Visualizer:
    def __init__(self, drone_body, drone_rotors):
        self.drone_model = DroneVisualModel(body = drone_body, rotor=drone_rotors)
        self.fig = plt.figure()
        
        self.ax = self.fig.add_subplot(1,1,1, projection='3d')
        self.drone_model.drawDroneModel(self.ax)
    
    def plotConfig(self, configuration: np.array):
        pass

def visualise(data, body, rotor, space="jointspace"):
    if space == "jointspace":
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1, projection='3d')
        #ax2 = fig.add_subplot(1,2,2, projection='3d')
        
        ax1.scatter(data[0], data[1], data[2])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([-1, 1])  # set range for X axis
        ax1.set_ylim([-1, 1])  # set range for Y axis
        ax1.set_zlim([-1, 1])  # set range for Z axis

        dronemodel = DroneVisualModel(body=body, rotor=rotor)
        dronemodel.drawDroneModel(ax1)

        plt.show()  
    
    elif space == "cartesian":
        # Build data structure
        for i in range(len(data)):
            
            data[0:3,i]
        inws_points = [row[0:3] for row in data if data[6]==0]
        outws_points = [row[0:3] for row in data if not data[6]==-1]

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1, projection='3d')
        
        ax1.scatter(data[0], data[1], data[2], c='red', label="Out of workspace")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([-1, 1])  # set range for X axis
        ax1.set_ylim([-1, 1])  # set range for Y axis
        ax1.set_zlim([-1, 1])  # set range for Z axis

        dronemodel = Model(body=(0.35, 0.15, 0.045), rotor=(0.2286, 0.245, 0.195))
        dronemodel.drawDroneModel(ax1)

        plt.show()  
        
    def show(self):
        plt.show()