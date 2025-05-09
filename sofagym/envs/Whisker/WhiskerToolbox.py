# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "emenager", "ekhairallah"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import SofaRuntime
import Sofa
import Sofa.Core
import Sofa.Simulation

from splib3.animation.animate import Animation
import math as m
import numpy as np
import sys
import pathlib
import csv
import os
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

SofaRuntime.importPlugin("Sofa.Component")

class StateInitializer(Sofa.Core.Controller):
    """Initialize the states.

    Methods:
    -------
        __init__: Initialization of all arguments.
        init_state: Randomly initialize the environment state.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]

        self.dt = self.rootNode.findData("dt").value

        self.whisker = self.rootNode.Whisker_node.Whisker.MechanicalModel
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.measured_ele = self.fem.strainmeasuringelements.value
        self.chambers = self.whisker.getObject("Chamber")
        ### Articulation system node
        self.arti_sys = self.rootNode.Whisker_node.getChild("Articulation_system")
        self.servo_arti = self.arti_sys.ServoMotor.Articulation.dofs
        self.servo_wheel = self.arti_sys.ServoMotor.Articulation.ServoWheel.dofs

        ### Plane
        self.plane = self.rootNode.getChild("plane")
        self.meca_plane = self.plane.getChild("oscilated_dof")
    def initialize(self,incr):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value[1]
        new_angleIn = current_angleIn + incr
        self.arti_sys.angleIn.value[1] = new_angleIn 

    def init_state(self, init_states):
        self.init_states = init_states
        # print("init_states = ", self.init_states)
        rot,_,_,_,_,_,_,_ = self.init_states
        with self.arti_sys.angleIn.writeable() as arti_input:
            arti_input[1] = rot
        
    # def onAnimateBeginEvent(self, event):
    #     self.step += 1
    #     if self.step == 1:
    #         self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.pressure_input.value[0] = 0.0001
    #         self.original_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
    # def onAnimateEndEvent(self,event):
    #     self.initiated_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
    #     if self.step == 1:
    #         self.arti_sys.angleIn[0] = self.original_min_pos - self.initiated_min_pos
class rewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
        self.scale_factor = kwargs["scale_factor"]
        self.cost = None
        self.force_thres = kwargs["force_threshold"]
        self.dt = self.rootNode.findData("dt").value

        self.whisker = self.rootNode.Whisker_node.Whisker.getChild("MechanicalModel")
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.whisker_container = self.whisker.getObject("container")
        self.current_strain = []
        self.time = 0
        self.rootNode.Whisker_node.Whisker.addData(name='force', type='vector<float>', help='Reaction Force',
                             value=[0.0,0.0,0.0])
        self.force_value = self.rootNode.Whisker_node.Whisker.force.value
        
        ### Strain BOXROI
        self.strain_box = self.rootNode.Whisker_node.strain_measuring_Box
        self.indices_in_box = self.strain_box.indices.value
        self.tetrahedra_list = self.whisker_container.tetrahedra.value
        self.measured_elemennt = []
        for i, tetra in enumerate(self.tetrahedra_list):
            common_nodes = set(tetra).intersection(self.indices_in_box)
            if len(common_nodes) >= 3:
                self.measured_elemennt.append(i)
        self.fem.strainmeasuringelements.value = self.measured_elemennt
        # print(f"self.measured_elemennt = {len(self.measured_elemennt)}")
        self.ele_indices = [self.tetrahedra_list[i] for i in self.measured_elemennt]
        self.volume = np.array([1])
        ### Articulation system node
        self.arti_sys = self.rootNode.Whisker_node.getChild("Articulation_system")
        self.servo_arti = self.arti_sys.ServoMotor.Articulation.dofs
        self.servo_wheel = self.arti_sys.ServoMotor.Articulation.ServoWheel.dofs
        self.angleIn_prev = 0
        self.angleIn_after = 0
        self.force_by_initiated_pressure = 0
    # def onKeypressedEvent(self, e):
    #     c = e['key']
    def tetrahedron_volume(self,n1,n2,n3,n4):
        # Extract node coordinates
        x1, y1, z1 = n1
        x2, y2, z2 = n2
        x3, y3, z3 = n3
        x4, y4, z4 = n4
        # Compute the Jacobian matrix
        J = np.array([
            [x2 - x1, y2 - y1, z2 - z1],
            [x3 - x1, y3 - y1, z3 - z1],
            [x4 - x1, y4 - y1, z4 - z1]
        ])
        # Compute the determinant
        det_J = np.linalg.det(J)

        # Compute volume
        volume = abs(det_J) / 6.0
        return volume
    def compute_strain(self,alpha,node0, node1, node2, node3,strain_local):
        E_local = np.zeros((3,3), dtype=float)
        E_local[0,0] = strain_local[0]  # E_xx
        E_local[1,1] = strain_local[1]  # E_yy
        E_local[2,2] = strain_local[2]  # E_zz

        # off-diagonal: yz = E_local_voigt[3], xz = E_local_voigt[4], xy = E_local_voigt[5]
        E_local[1,2] = strain_local[3]  # yz
        E_local[2,1] = strain_local[3]
        
        E_local[0,2] = strain_local[4]  # xz
        E_local[2,0] = strain_local[4]
        
        E_local[0,1] = strain_local[5]  # xy
        E_local[1,0] = strain_local[5]
        # 1) Build local->global rotation R
        # Vectors in global coords
        x_axis = node1 - node0
        x_axis /= np.linalg.norm(x_axis)

        w = np.cross(x_axis, (node2 - node0))
        z_axis = w / np.linalg.norm(w)

        y_axis = np.cross(z_axis, x_axis)

        # local->global rotation matrix (3x3)
        R_local_to_global = np.column_stack((x_axis, y_axis, z_axis))

        # 2) Build the global direction n_global:
        #    "Z-axis rotated by alpha around X-axis"
        alpha = np.radians(alpha)
        # Rotation matrix around global X:
        R_x_alpha = np.array([
            [1,           0,            0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ], dtype=float)

        # The original global Z is (0,0,1):
        z_global = np.array([0.0, 0.0, 1.0], dtype=float)
        n_global = R_x_alpha @ z_global  # shape (3,)
        # -------------------------------------------------------------------------
        # 3) Convert n_global to local coords:
        #    n_local = R_local_to_global^T * n_global
        #    because v_local = R^T * v_global  (if R is local->global)
        # -------------------------------------------------------------------------
        n_local = R_local_to_global.T @ n_global
        length = np.linalg.norm(n_local)
        if length > 1e-12:
            n_local /= length
        # -------------------------------------------------------------------------
        # 4) The normal strain along n_local in E_local is:
        #    ε = n_local^T * E_local * n_local
        # -------------------------------------------------------------------------
        # E_local is a 3x3 matrix in local coords
        # n_local is shape (3,). So we do standard matrix multiply
        nE = E_local @ n_local         # shape (3,) -> E_local * n_local
        normal_strain = n_local.dot(nE)  # = n_local^T * (E_local * n_local)
        return normal_strain
    def onAnimateBeginEvent(self, event):
        self.angleIn_prev = self.arti_sys.angleIn.value[1]
        self.vonMises_stress = self.fem.vonMisesPerElement.value        
        self.time += 1
        if self.time <= 1:
            
                
            self.original_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
            # self.arti_sys.angleIn[1] = -10*m.pi/180
            # self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.pressure_input.value[0] = 0.05
            # self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity1.pressure_input.value[0] = 0.05 # real pressure is input_value / dt [kPa]
        self.strain_zz = np.array([])
        self.volume = np.array([])
        for idx,(i,j) in enumerate(zip(self.measured_elemennt,self.ele_indices)):
            strain_converted = self.compute_strain(0,self.mecawhisker.position.value[j[0]],
                                                        self.mecawhisker.position.value[j[1]],
                                                        self.mecawhisker.position.value[j[2]],
                                                        self.mecawhisker.position.value[j[3]],self.fem.totalstrain.value[idx])
            if strain_converted > 0:
                self.strain_zz = np.append(self.strain_zz,strain_converted)

                self.volume = np.append(self.volume,self.tetrahedron_volume(self.mecawhisker.position.value[j[0]],
                                                        self.mecawhisker.position.value[j[1]],
                                                    self.mecawhisker.position.value[j[2]],
                                                    self.mecawhisker.position.value[j[3]]))
        
        # print(f"strain_zz = {self.strain_zz}")
        # stress = [self.vonMises_stress[i] for i in self.measured_elemennt]
        # print(f"stress = {sum(stress)}")
        if np.sum(self.volume) != 0:
            average_strain = np.sum(self.strain_zz*self.volume)/np.sum(self.volume)
        else:
            average_strain = 0
        self.fem.ave_strain.value = average_strain
        # print(f"Strain max = {max(self.strain_zz)}")
        # print(f"Strain ave = {average_strain:.5f}")
    def onAnimateEndEvent(self, event):
        # self.arti_sys.angleIn[1] = 0.6
        self.initiated_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
        if self.time <= 1:
            self.arti_sys.angleIn[0] = self.original_min_pos - self.initiated_min_pos               
        ### Force calculation
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
        for i in range(len(pointId)):
            contactforce_x += constraintDirections[i][0] * forcesNorm[constraintId[i]]/self.dt*10**(-3) #Unit mN *10**(-3) => N
            contactforce_y += constraintDirections[i][1] * forcesNorm[constraintId[i]]/self.dt*10**(-3)  #Unit mN *10**(-3) => N
            contactforce_z += constraintDirections[i][2] * forcesNorm[constraintId[i]]/self.dt *10**(-3) #Unit mN *10**(-3) => N
        force_value = [float(contactforce_x),float(contactforce_y),float(contactforce_z)]
        number_of_valid_constraint = sum(1 for x in forcesNorm if x != 0)
        if self.time <=1:
            self.force_by_initiated_pressure = force_value
        if number_of_valid_constraint <= 2:
            self.force_value = [0 for _ in range(len(force_value))]
        else:
            self.force_value = [force_value[i] - self.force_by_initiated_pressure[i] for i in range(len(force_value))]
        # self.force_value = [force_value[i] - self.force_by_initiated_pressure[i] for i in range(len(force_value))]
        self.angleIn_after = self.arti_sys.angleIn.value[1]
        self.angleIn_diff = self.angleIn_after - self.angleIn_prev
        # print(f'Force = {self.force_value}')
    def getReward(self):
            
        alpha = 1
        beta = 0.5
        if abs(self.force_value[1]) <= 0.001 or self.rootNode.applyAction.new_angleIn > 0.6:
            self.reward = 0
        else:
            force_penalty = np.maximum(0, (abs(self.force_value[1]) - self.force_thres)/self.force_thres)
            base_reward_force = alpha - alpha*force_penalty
            # print(f"force_penalty = {force_penalty} => base_reward_force = {base_reward_force}")
            rot_angle_penalty = abs(self.angleIn_diff)*self.scale_factor/self.rootNode.applyAction.max_incr
            # print(f"self.rootNode.applyAction.max_incr = {self.rootNode.applyAction.max_incr}")
            base_reward_angle = beta * rot_angle_penalty
            # print(f"rot_angle_penalty = {rot_angle_penalty} => base_reward_force = {base_reward_angle}")
            self.reward = np.maximum(0,base_reward_force + base_reward_angle)

        # print("reward = ", self.reward)
        return self.reward, self.cost

    def update(self,goal=None):
        pass


def getReward(root):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """
    reward, cost = root.Reward.getReward()

    return False, reward
               

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.whisker_node = self.root.Whisker_node
        self.max_incr = 30*m.pi/180
        self.arti_sys = self.root.Whisker_node.getChild("Articulation_system")
    def _rotate(self, incr):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value
        self.new_angleIn = current_angleIn[1] + incr
        # with self.arti_sys.angleIn.writeable() as arti_input:
        if self.new_angleIn < 0.65:
            self.arti_sys.angleIn.value = [current_angleIn[0],self.new_angleIn]
            
    def _normalizedAction_to_action(self, action):
        return self.max_incr*action/2

    def compute_rot_action(self, action, nb_step):
        incr= self._normalizedAction_to_action(action)/nb_step
        return incr

    def apply_action(self, incr):
        self._rotate(incr)

            
def startCmd(root, action, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float  = self.config["dt"]*(self.config["scale_factor"]-1
            Duration of the animation.

    Returns:
    ------
        None.

    """
    incr = action_to_command(root,action[0],duration/root.dt.value + 1)
    # print(f'action = {action[0]*100}')
    startCmd_Whisker(root, root.Whisker_node,incr, duration)

def action_to_command(root,action,nb_step):
    """Link between Gym action (int) and SOFA command (rotation, translation,
    displacement).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).

    Returns:
    -------
        The command (rotation, direction, displacement).
    """
    incr = root.applyAction.compute_rot_action(action, nb_step)
    
    return incr

def startCmd_Whisker(rootNode, whisker,incr, duration):
    """Initialize the command.
    """

    # Definition of the elements of the animation
    def executeAnimation(whisker,incr, factor):
        rootNode.applyAction.apply_action(incr)
    

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"whisker": whisker,
                    "incr": incr},
            duration=duration, mode="once"))


def getState(root):

    chamber_node = root.Whisker_node.Whisker.MechanicalModel.Chamber
    body_length = root.Whisker_node.Whisker.body_length.value
    no_chamber = root.Whisker_node.Whisker.no_chamber.value
    chamber_length = root.Whisker_node.Whisker.chamber_length.value
    thickness = root.Whisker_node.Whisker.thickness.value
    pressure = []
    for i in range(2):
        if i <= no_chamber-1:
            pressure.append(chamber_node.getChild(f'cavity{i}').pressure_input.getData('value').value[0].tolist())
        else:
            pressure.append(0.0)

    rot_angle = root.Whisker_node.Articulation_system.angleIn.value[1].tolist()
    strain_zz = root.Whisker_node.Whisker.MechanicalModel.FEM.ave_strain.value
    
    state = [rot_angle] + [strain_zz] + [body_length] + [no_chamber]+ [chamber_length]+[thickness]+pressure
    # print("State = ", state)
    return state


def getPos(root):
    no_chamber = root.Whisker_node.Whisker.no_chamber.value
    whisker_pos = root.Whisker_node.Whisker.MechanicalModel.dofs.position.value.tolist()
    deformable_pos = root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value.tolist()
    rigid_pos = root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value.tolist()
    arti_pos = root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value.tolist()
    body_pos = root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value.tolist()
    
    if no_chamber == 1:
        chamber1_fiber1 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber1.dofs.position.value.tolist()
        chamber1_fiber2 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber2.dofs.position.value.tolist()     
    
        return [
                whisker_pos,
                deformable_pos,
                rigid_pos,
                chamber1_fiber1,
                chamber1_fiber2,
                arti_pos,
                body_pos
                ]
    else:
        chamber1_fiber1 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber1.dofs.position.value.tolist()
        chamber1_fiber2 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber2.dofs.position.value.tolist()     
        chamber2_fiber1 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber2_fiber1.dofs.position.value.tolist()
        chamber2_fiber2 = root.Whisker_node.Whisker.MechanicalModel.fiber.chamber2_fiber2.dofs.position.value.tolist()         
        return [
                whisker_pos,
                deformable_pos,
                rigid_pos,
                chamber1_fiber1,
                chamber1_fiber2,
                chamber2_fiber1,
                chamber2_fiber2,
                arti_pos,
                body_pos 
                ]
def setPos(root, pos):
    no_chamber = root.Whisker_node.Whisker.no_chamber.value
    if no_chamber == 1:
        [
        whisker_pos,
        deformable_pos,
        rigid_pos,
        chamber1_fiber1,
        chamber1_fiber2,
        arti_pos,
        body_pos
        ] = pos
        root.Whisker_node.Whisker.MechanicalModel.dofs.position.value = np.array(whisker_pos)
        root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value = np.array(deformable_pos)
        root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value = np.array(rigid_pos)
        root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value = np.array(arti_pos)
        root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value = np.array(body_pos)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber1.dofs.position.value = np.array(chamber1_fiber1)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber2.dofs.position.value = np.array(chamber1_fiber2)

    else:
        [
        whisker_pos,
        deformable_pos,
        rigid_pos,
        chamber1_fiber1,
        chamber1_fiber2,
        chamber2_fiber1,
        chamber2_fiber2,
        arti_pos,
        body_pos
        ] = pos
        root.Whisker_node.Whisker.MechanicalModel.dofs.position.value = np.array(whisker_pos)
        root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value = np.array(deformable_pos)
        root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value = np.array(rigid_pos)
        root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value = np.array(arti_pos)
        root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value = np.array(body_pos)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber1.dofs.position.value = np.array(chamber1_fiber1)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber1_fiber2.dofs.position.value = np.array(chamber1_fiber2)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber2_fiber1.dofs.position.value = np.array(chamber2_fiber1)
        root.Whisker_node.Whisker.MechanicalModel.fiber.chamber2_fiber2.dofs.position.value = np.array(chamber2_fiber2)