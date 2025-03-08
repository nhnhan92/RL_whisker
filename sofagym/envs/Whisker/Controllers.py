# -*- coding: utf-8 -*-

import Sofa.Core
import Sofa.Simulation
from splib3.animation import animate
from Sofa.constants import *
import math as m
import csv
import os
import numpy as np
import random
from ext_data import ext_data
contact_list, contact_pos, _ = ext_data()
# def savestrain_csv(name = "", ele = 0):
#     filePath_node = os.getcwd()+ name + str(ele)+".csv"
#     try:
#         os.remove(filePath_node)
#     except:
#         print("Error while deleting file ", filePath_node)
#     header = ["Sim_step", "lamda_xx", "lamda_yy", "lamda_zz","lamda_yz", "lamda_xz", "lamda_xy"]
#     with open(filePath_node, "w", newline="") as file_open:
#         writer = csv.writer(file_open, delimiter=",")
#         writer.writerow(header)
#         for j in range(idx_z_limit):
#             if i == 0 and init_curve[i][j][2]<h:
#                 writer.writerow([j,x[j],y[j],z[j]])
#             elif i == 1 and init_curve[i][j][2]<h:
#                 writer.writerow([j,x2[j],y2[j],z2[j]])
class WhiskerController(Sofa.Core.Controller):
    def moveRestPos(self, rest_pos, dx, dy, dz):
        str_out = []
        for i in range(0, len(rest_pos)):
            str_out += [
                [
                    rest_pos[i][0] + dx,
                    rest_pos[i][1] + dy,
                    rest_pos[i][2] + dz,
                    rest_pos[i][3],rest_pos[i][4],rest_pos[i][5],rest_pos[i][6]
                ]
            ]

        return str_out

    def moveRestPos_3(self, rest_pos, dx, dy, dz):
        str_out = []
        for i in range(0, len(rest_pos)):
            str_out += [
                [
                    rest_pos[i][0] + dx,
                    rest_pos[i][1] + dy,
                    rest_pos[i][2] + dz 
                ]
            ]

        return str_out
    
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.rootNode = kwargs['node']
        self.body_length = kwargs['body_length']

        self.dt = self.rootNode.findData("dt").value
        ### Whisker body node
        self.whisker = self.rootNode.Whisker_node.Whisker.getChild("MechanicalModel")
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.whisker_topo = self.whisker.getObject("loader")
        self.measured_ele = self.fem.strainmeasuringelements.value
        self.smallend_box = self.whisker.smallend_Box
        self.small_end_radi = 12 - (int(self.body_length)/m.tan(85.5*m.pi/180))
        self.chambers = self.whisker.getObject("Chamber")
        self.no_chamber = self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.no_chamber.value
        ### Articulation system node
        self.arti_sys = self.rootNode.Whisker_node.getChild("Articulation_system")
        self.servo_arti = self.arti_sys.ServoMotor.Articulation.dofs
        self.servo_wheel = self.arti_sys.ServoMotor.Articulation.ServoWheel.dofs

        ### Plane
        self.plane = self.rootNode.getChild("plane")
        self.meca_plane = self.plane.getObject("oscilated_dof")

        self.rootNode.Whisker_node.Whisker.addData(name='force', type='vector<float>', help='Reaction Force',
                             value=[0.0,0.0,0.0])
        self.force_value = self.rootNode.Whisker_node.Whisker.force.value
        self.step = 0
    def getTranslated(points, vec):
        r = []
        for v in points:
            x = v[0]+vec[0]
            y = v[1]+vec[1]
            z = v[2]+vec[2]
            r.append([x, y, z])
        return r
    def init_state(self, init_states):
        self.init_states = init_states
        print("CHECK Before = ",min(sublist[1] for sublist in self.mecawhisker.position.value))
        
        
        # print(min(sublist[1] for sublist in self.mecawhisker.position.value))

    def onAnimateBeginEvent(self, event):
        
        print("Begin Envent check")
        self.step += 1
        print(self.step)
        if self.step == 1:
            print("CHECK Before = ",min(sublist[1] for sublist in self.mecawhisker.position.value))
        if self.step == 2:
            print("In")
            self.init_states = [1.5,0,0.05,0,0]
            rot,strain,pressure1,pressure2,pressure3 = self.init_states
            for i in range(self.no_chamber):
                if i == 0:
                    with self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.pressure_input.value.writeable() as pressure_input:
                        pressure_input[0] = pressure1
                if i == 1:
                    with self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity1.pressure_input.value.writeable() as pressure_input:
                        pressure_input[0] = pressure2
                if i == 3:
                    with self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity2.pressure_input.value.writeable() as pressure_input:
                        pressure_input[0] = pressure3
            with self.arti_sys.angleIn.writeable() as arti_input:
                arti_input[1] = 0
            #     arti_input[1] = 10
            print("CHECK AFTER = ",min(sublist[1] for sublist in self.mecawhisker.position.value))
            
    def onAnimateEndEvent(self, event):
        print("Pressure = ", self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.pressure_input.value)
        # print(min(sublist[1] for sublist in self.mecawhisker.position.value))
        # self.rootNode.Whisker_node.Articulation_system.angleIn.value[1] = -10
        angleIn = self.rootNode.Whisker_node.Articulation_system.angleIn.value[0]
        # print("angleIn = ", angleIn)
        constraint = self.mecawhisker.constraint.value
        constraintMatrixInline = np.fromstring(constraint, sep='  ')
        pointId = []
        constraintId = []
        constraintDirections = []
        index = 0
        forcesNorm = self.rootNode.GCS.constraintForces.value
        constraintDirections = []
        
        while index < len(constraintMatrixInline):
            nbConstraint   = int(constraintMatrixInline[index+1])
            currConstraintID = int(constraintMatrixInline[index])
            for pts in range(nbConstraint):
                currIDX = index+2+pts*4
                pointId.append(constraintMatrixInline[currIDX])
                constraintId.append(currConstraintID)
                constraintDirections.append([constraintMatrixInline[currIDX+1],
                                            constraintMatrixInline[currIDX+2],
                                            constraintMatrixInline[currIDX+3]])
            index = index + 2 + nbConstraint*4
        contactforce_x = 0
        contactforce_y = 0
        contactforce_z = 0
        indice = []
        for i in range(len(pointId)):
            indice.append(int(pointId[i]))  
            contactforce_x += constraintDirections[i][0] * forcesNorm[constraintId[i]] 
            contactforce_y += constraintDirections[i][1] * forcesNorm[constraintId[i]]
            contactforce_z += constraintDirections[i][2] * forcesNorm[constraintId[i]]
        
        forces = [float(contactforce_x),float(contactforce_y),float(contactforce_z)]
        self.force_value = forces
    def onKeypressedEvent(self, e):
        c = e['key']

        increment = 1
        if c == Sofa.constants.Key.plus:  
            self.init_state([1.5,0,0.1,0,0])
        
        if c == Sofa.constants.Key.leftarrow:
            self.moveRestPos(self.mecawhisker.findData("rest_position").value, 0, 0, -increment)
        
        if c == Sofa.constants.Key.rightarrow:
            self.moveRestPos(self.mecawhisker.findData("rest_position").value, 0, 0, increment)

        if c == Sofa.constants.Key.uparrow:
            self.moveRestPos(self.mecawhisker.findData("rest_position").value, increment, 0, 0)
        
        if c == Sofa.constants.Key.downarrow:
            self.moveRestPos(self.mecawhisker.findData("rest_position").value, -increment, 0, 0)
        
        if (e["key"] == Sofa.constants.Key.KP_1):
            pressureValue_left = self.pressure_left.value + 0.005
            if pressureValue_left > 1:
                pressureValue_left = 1
            self.pressure_left.value = pressureValue_left

        if (e["key"] == Sofa.constants.Key.KP_2):
            pressureValue_left = self.pressure_left.value - 0.005
            if pressureValue_left < 0:
                pressureValue_left = self.pressure_left.value
            self.pressure_left.value = pressureValue_left

        if (e["key"] == Sofa.constants.Key.KP_4):
            pressureValue_right = self.pressure_right.value + 0.005
            if pressureValue_right > 1:
                pressureValue_right = 1
            self.pressure_right.value = pressureValue_right
        
        if (e["key"] == Sofa.constants.Key.KP_5):
            pressureValue_right = self.pressure_right.value - 0.005
            if pressureValue_right < 0:
                pressureValue_right = self.pressure_right.value
            self.pressure_right.value = pressureValue_right
