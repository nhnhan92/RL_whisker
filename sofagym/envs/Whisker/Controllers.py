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
        # self.inner_pressure = self.node.Cavity1.SurfacePressureConstraint.value

        self.dt = self.rootNode.findData("dt").value
        ### Whisker body node
        self.whisker = self.rootNode.Whisker_node.Whisker.getChild("MechanicalModel")
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.whisker_topo = self.whisker.getObject("loader")
        self.measured_ele = self.fem.strainmeasuringelements.value
        self.smallend_box = self.whisker.smallend_Box
        self.small_end_radi = 12 - (int(self.body_length)/m.tan(85.5*m.pi/180))
        # self.trash_roi = self.whisker.getObject("trash")
        # self.current_trash_roi = self.trash_roi.findData("box").value
        # self.chamber_node = self.whisker.getChild("Chamber")
        # self.chamber_right = self.chamber_node.getChild("cavity_right")
        # self.pressure_right = self.chamber_right.getObject('SurfacePressureConstraint')
 
        # self.chamber_left = self.chamber_node.getChild('cavity_left')
        # self.pressure_left = self.chamber_left.getObject('SurfacePressureConstraint')

        ### Articulation system node
        self.arti_sys = self.rootNode.Whisker_node.getChild("Articulation_system")
        for ele in self.measured_ele:
            filePath_node = "strain_data_scene/strain_" + str(ele)+".csv"
            try:
                os.remove(filePath_node)
            except:
                print("Error while deleting file ", filePath_node)
            self.strain_header = ["Sim_step", "lamda_xx", "lamda_yy", "lamda_zz","lamda_yz", "lamda_xz", "lamda_xy"]
            with open(filePath_node, "w", newline="") as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)
                csv_writer.writeheader()
            with open(filePath_node, 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)

                info = {
                    "Sim_step": 0,
                    "lamda_xx": 0,
                    "lamda_yy": 0,
                    "lamda_zz": 0,
                    "lamda_yz": 0,
                    "lamda_xz": 0,
                    "lamda_xy": 0
                }
                csv_writer.writerow(info)
        
        self.pole = self.rootNode.getChild("pole")
        self.pole_meca = self.pole.dofs
        self.pole_loader = self.pole.getObject("pole_loader")
        self.factor = 0
        self.count_scene = 0

    def getTranslated(points, vec):
        r = []
        for v in points:
            x = v[0]+vec[0]
            y = v[1]+vec[1]
            z = v[2]+vec[2]
            r.append([x, y, z])
        return r
    def animation(target, factor):
        rot_angle = 60
        target.angleIn.value = factor * (-rot_angle*m.pi/180)
    def onAnimateBeginEvent(self, event):
        # print(self.pole_meca.position.value)
        self.count_scene += round(self.dt,2)
        self.strain = self.fem.totalstrain.value
        random_no = random.randint(1, 2)
        angle = 60
        self.factor += self.dt/2
        rot_angle = self.rootNode.Whisker_node.Articulation_system.angleIn.value

        if self.count_scene <= 0.01:
            for i in range(len(self.smallend_box.pointsInROI.value)):
                if -0.0001<self.smallend_box.pointsInROI.value[i][0] - self.small_end_radi < 0.0001:
                    self.contact_idx = self.smallend_box.indices.value[i]
                    # print(self.contact_idx)
                    break
            new_pole_pos = self.moveRestPos(self.pole_meca.rest_position.value,
                                                1+3+self.rootNode.localmindistance.contactDistance.value+self.mecawhisker.rest_position.value[self.contact_idx][0]-self.pole_meca.rest_position.value[0][0],
                                                self.mecawhisker.rest_position.value[self.contact_idx][1]-self.pole_meca.rest_position.value[0][1],
                                                self.mecawhisker.rest_position.value[self.contact_idx][2]-self.pole_meca.rest_position.value[0][2])
            self.pole_meca.rest_position.value = new_pole_pos

                
        
        if rot_angle < (angle*m.pi/180)*2:
            # self.moveRestPos(self.mecawhisker.rest_position.value, increment, 0, 0)
            # current_angleIn = self.arti_sys.angleIn.value
            # self.arti_sys.angleIn.value = current_angleIn - 0.002
            # print(self.arti_sys.angleIn.value)
            # print(self.mecawhisker.position.value[0])
            if self.count_scene>0.02:
                for ele in range(len(self.measured_ele)):
                    with open("strain_data_scene/strain_" + str(self.measured_ele[ele])+".csv", 'a') as csv_file:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)

                        info = {
                            "Sim_step": round(self.count_scene,2),
                            "lamda_xx": round(self.strain[ele][0],4) ,
                            "lamda_yy": round(self.strain[ele][1],4),
                            "lamda_zz": round(self.strain[ele][2],4),
                            "lamda_yz": round(self.strain[ele][3],4),
                            "lamda_xz": round(self.strain[ele][4],4),
                            "lamda_xy": round(self.strain[ele][5],4),
                        }
                        csv_writer.writerow(info)

        else:
            pass
    def onKeypressedEvent(self, e):
        c = e['key']

        increment = 1
        # if c == Sofa.constants.Key.plus:  
        #     self.move_trash_roi(self.current_trash_roi,0,0,-increment)
        
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
