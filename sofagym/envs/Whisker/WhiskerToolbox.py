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
        self.no_chamber = self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.no_chamber.value
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
        print("init_states = ", self.init_states)
        rot,strain,pressure1,pressure2 = self.init_states
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
        self.cost = None
        self.force_thres = 0.2
        self.dt = self.rootNode.findData("dt").value

        self.whisker = self.rootNode.Whisker_node.Whisker.getChild("MechanicalModel")
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.fem.strainmeasuringelements.value = [0,0,0,0,0]
        self.whisker_topo = self.whisker.getObject("loader")
        self.current_strain = []
        self.time = 0
        self.rootNode.Whisker_node.Whisker.addData(name='force', type='vector<float>', help='Reaction Force',
                             value=[0.0,0.0,0.0])
        self.force_value = self.rootNode.Whisker_node.Whisker.force.value
        
        ### Strain BOXROI
        self.strain_box = self.rootNode.Whisker_node.strain_measuring_Box
        self.measured_elemennt = self.strain_box.tetrahedronIndices.value
        self.fem.strainmeasuringelements.value = self.measured_elemennt
        self.ele_indices = self.strain_box.tetrahedraInROI.value
        self.point_idx = self.strain_box.indices.value
        self.point_pos = self.strain_box.pointsInROI.value
        
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
    def onAnimateBeginEvent(self, event):

        self.angleIn_prev = self.arti_sys.angleIn.value[1]
        self.time += 1
        if self.time == 1:
            self.volume = np.array([])
            # self.mecawhisker.position.value[self.ele_indices[0]]
            for i,j in zip(self.measured_elemennt,self.ele_indices):
                self.volume = np.append(self.volume,self.tetrahedron_volume(self.mecawhisker.position.value[j[0]],
                                                        self.mecawhisker.position.value[j[1]],
                                                        self.mecawhisker.position.value[j[2]],
                                                        self.mecawhisker.position.value[j[3]]))
            self.original_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
            # self.arti_sys.angleIn[1] = -10*m.pi/180
            # self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.pressure_input.value[0] = 0.05
            # self.rootNode.Whisker_node.Whisker.MechanicalModel.Chamber.cavity1.pressure_input.value[0] = 0.05 # real pressure is input_value / dt [kPa]
        self.strain_zz = np.array([item[2] for item in self.fem.totalstrain.value])
        average_strain = np.sum(self.strain_zz*self.volume)/np.sum(self.volume)
        self.fem.ave_strain.value = average_strain
        
    def onAnimateEndEvent(self, event):
        
        self.initiated_min_pos = min(sublist[1] for sublist in self.mecawhisker.position.value)
        if self.time == 20:
            self.arti_sys.angleIn[0] = self.original_min_pos - self.initiated_min_pos        
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
        contactforce_x_57 = 0
        contactforce_y_57 = 0
        contactforce_z_57 = 0
        for i in range(len(pointId)):
            contactforce_x += constraintDirections[i][0] * forcesNorm[constraintId[i]]/self.dt*10**(-3) #Unit mN
            contactforce_y += constraintDirections[i][1] * forcesNorm[constraintId[i]]/self.dt*10**(-3)  #Unit mN
            contactforce_z += constraintDirections[i][2] * forcesNorm[constraintId[i]]/self.dt *10**(-3) #Unit mN
            if pointId[i] == 58:
                contactforce_x_57 += constraintDirections[i][0] * forcesNorm[constraintId[i]] /self.dt
                contactforce_y_57 += constraintDirections[i][1] * forcesNorm[constraintId[i]]/self.dt
                contactforce_z_57 += constraintDirections[i][2] * forcesNorm[constraintId[i]]/self.dt
        # print("Force node 57 = ",[contactforce_x_57,contactforce_y_57,contactforce_z_57])
        
        force_value = [float(contactforce_x),float(contactforce_y),float(contactforce_z)]
        
        if self.time <=20:
            self.force_by_initiated_pressure = force_value
        self.force_value = [force_value[i] - self.force_by_initiated_pressure[i] for i in range(len(force_value))]
        self.angleIn_after = self.arti_sys.angleIn.value[1]

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        # """ 
        alpha = 1
        beta = 0.5
        scale_factor = 10**(5)
        force_penalty = np.maximum(0, self.force_value[2] - self.force_thres) ** 2
        rot_angle_penalty = (self.angleIn_after - self.angleIn_prev)**2

        reward = (alpha * force_penalty + beta * rot_angle_penalty)*scale_factor
        return reward, self.cost

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
        self.max_incr = 50*m.pi/180
        self.arti_sys = self.root.Whisker_node.getChild("Articulation_system")
    def _rotate(self, incr):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value
        new_angleIn = current_angleIn[1]*m.pi/180 + incr
        # with self.arti_sys.angleIn.writeable() as arti_input:
        self.arti_sys.angleIn.value = [current_angleIn[0],new_angleIn]
    def _normalizedAction_to_action(self, action):
        return self.max_incr*action/2

    def compute_rot_action(self, action, nb_step):
        incr= self._normalizedAction_to_action(action)/nb_step
        # incr = (goal_angleIn - current_angleIn)/nb_step
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

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        fingers: list
            The fingers.
        rotation, pressure, displacement: float
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
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
    """Compute the state of the environment/agent.

    Note:
    ----
        The state is normalized.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    chamber_node = root.Whisker_node.Whisker.MechanicalModel.Chamber
    no_chamber = root.Whisker_node.Whisker.MechanicalModel.Chamber.no_chamber.value
    pressure = []
    for i in range(2):
        if i <= no_chamber-1:
            pressure.append(chamber_node.getChild(f'cavity{i}').pressure_input.getData('value').value[0].tolist())
        else:
            pressure.append(0.0)

    rot_angle = root.Whisker_node.Articulation_system.angleIn.value[1].tolist()
    strain_zz = root.Whisker_node.Whisker.MechanicalModel.FEM.ave_strain.value.tolist()
    
    state = [rot_angle] + [strain_zz] + pressure
    # print("State = ", state)
    return state


def getPos(root):
    """Return the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        The position(s) of the object(s) of the scene.
    """

    whisker_pos = root.Whisker_node.Whisker.MechanicalModel.dofs.position.value.tolist()
    deformable_pos = root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value.tolist()
    rigid_pos = root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value.tolist()
    
    # fiber1_right_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_right.dofs.position.value.tolist()
    # fiber1_left_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_left.dofs.position.value.tolist()
    # fiber2_right_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value.tolist()
    # fiber2_left_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value.tolist()

    # arm_pos = root.Whisker_node.Articulation_system.ServoArm.dofs.position.value.tolist()
    arti_pos = root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value.tolist()
    body_pos = root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value.tolist()
    
    return [
            whisker_pos,
            deformable_pos,
            rigid_pos,
            # fiber1_right_pos,
            # fiber1_left_pos,
            # fiber2_right_pos,
            # fiber2_left_pos,
            # arm_pos,
            arti_pos,
            body_pos, 
            ]

def setPos(root, pos):
    """Set the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.
        pos: list
            The position(s) of the object(s) of the scene.

    Returns:
    -------
        None.

    Note:
    ----
        Don't forget to init the new value of the position.

    """
    [
    whisker_pos,
     deformable_pos,
     rigid_pos,
    # fiber1_right_pos,
    # fiber1_left_pos,
    # fiber2_right_pos,
    # fiber2_left_pos,
    #  arm_pos,
     arti_pos,
     body_pos,
    ] = pos
    root.Whisker_node.Whisker.MechanicalModel.dofs.position.value = np.array(whisker_pos)
    root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value = np.array(deformable_pos)
    root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value = np.array(rigid_pos)
    
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_right.dofs.position.value = np.array(fiber1_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_left.dofs.position.value = np.array(fiber1_left_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_left_pos)

    # root.Whisker_node.Articulation_system.ServoArm.dofs.position.value = np.array(arm_pos)
    root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value = np.array(arti_pos)
    root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value = np.array(body_pos)
