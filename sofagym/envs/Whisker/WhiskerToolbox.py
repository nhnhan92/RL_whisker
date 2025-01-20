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
fieldnames = ["time","xx", "yy", "zz", "yz", "xz", "xy"]
# with open('strain_data_rl/strain_data.csv', 'w') as csv_file:
#     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     csv_writer.writeheader()

# with open('strain_data_rl/strain_data.csv', 'a') as csv_file:
#     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     info = {
#         "time": 0,
#         "xx": 0,
#         "yy": 0,
#         "zz": 0,
#         "yz": 0,
#         "xz": 0,
#         "xy": 0
#     }
#     csv_writer.writerow(info)
# contact_list = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

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
        # if kwargs["pole_length"]:
        #     self.pole_length = kwargs["pole_length"]
        # if kwargs["init_states"] is not None:
        #     self.init_states = kwargs["init_states"]
        # else:
        #     print(">> ERROR: no initial states given.")
        #     exit(1)

        # self.cart = self.rootNode.Modeling.Cart
        # self.pole = self.rootNode.Modeling.Pole

    def init_state(self, init_states):

        self.init_states = init_states

        # with self.pole.MechanicalObject.position.writeable() as position:
        #     position[0][0] = x_pos
        #     position[0][1] = y_pos


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
        # self.goal_pos = None
        if kwargs["goalPos"] is not None:
            self.goal_pos = kwargs["goalPos"]
        self.effMO = None

        self.cost = None
        
        self.dt = self.rootNode.findData("dt").value

        self.whisker = self.rootNode.Whisker_node.Whisker.getChild("MechanicalModel")
        self.mecawhisker = self.whisker.getObject("dofs")
        self.fem = self.whisker.getObject("FEM")
        self.whisker_topo = self.whisker.getObject("loader")
        self.measured_ele = self.fem.strainmeasuringelements.value
        self.current_strain = []
        self.forces = []
    # def onKeypressedEvent(self, e):
    #     c = e['key']


    def onAnimateEndEvent(self, event):
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
        
        self.forces = [float(contactforce_x),float(contactforce_y),float(contactforce_z)]
    
    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        # """ 
        self.count_scene += 0.01
        self.count += 1
        ele = 1785
        
        self.strain = self.fem.totalstrain.value
        # self.current_strain.append(self.strain[1][4])
        current_cost = 0.1
        # self.current_strain[self.count]= self.strain[1][4]
        # current_cost = np.sqrt(np.mean((self.current_strain - self.strain_baseline)**2))

        if not self.cost:
            self.cost = current_cost
            return 0, self.cost

        reward = round(abs(self.cost - current_cost),6)
        self.cost = current_cost
        # print(self.cost)

        angle = 60
        self.factor += self.dt/2
        rot_angle = self.factor * (angle*m.pi/180)
        # if rot_angle < (angle*m.pi/180)*2:   
        #     if self.count_scene>0.01:         
        #         for ele in range(len(self.measured_ele)):
        #             with open("strain_data_rl/strain_" + str(self.measured_ele[ele])+".csv", 'a') as csv_file:
        #                 csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)

        #                 info = {
        #                     "Sim_step": round(self.count_scene,2),
        #                     "lamda_xx": self.strain[ele][0],
        #                     "lamda_yy": self.strain[ele][1],
        #                     "lamda_zz": self.strain[ele][2],
        #                     "lamda_yz": self.strain[ele][3],
        #                     "lamda_xz": self.strain[ele][4],
        #                     "lamda_xy": self.strain[ele][5]
        #                 }
        #                 csv_writer.writerow(info)
        #         with open("strain_data_rl/reward_cost_record.csv", 'a') as csv_file:
        #             csv_writer = csv.DictWriter(csv_file, fieldnames=self.header)
        #             info = {
        #                     "Sim_step": round(self.count_scene,2),
        #                     "cost": self.cost,
        #                     "reward": reward
        #                 }
        #             csv_writer.writerow(info)   


        return reward, self.cost

    def update(self,goal=None):
        """Compute the distance between object and goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.count = 0
        self.count_scene = 0
        self.factor = 0
        ele = 1785
        # self.groundtruth_data = pd.read_csv("strain_groundtruth/strain_" + str(ele)+".csv")
        # self.strain_baseline = self.groundtruth_data['lamda_xz']
        # ele_err = 1392
        # self.error_data = pd.read_csv("strain_groundtruth/strain_" + str(ele_err)+".csv")
        # self.strain_error = self.error_data['lamda_xz']

        # self.cost = np.sqrt(np.mean((self.strain_error[1:] - self.strain_baseline[1:])**2))
        # self.current_strain = self.strain_error
        # # print('initial cost = ', self.cost)
        # for ele in self.measured_ele:
        #     filePath_node = "strain_data_rl/strain_" + str(ele)+".csv"
        #     try:
        #         os.remove(filePath_node)
        #     except:
        #         print("Error while deleting file ", filePath_node)
        #     self.strain_header = ["Sim_step", "lamda_xx", "lamda_yy", "lamda_zz","lamda_yz", "lamda_xz", "lamda_xy"]
        #     with open(filePath_node, "w", newline="") as csv_file:
        #         csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)
        #         csv_writer.writeheader()
        #     with open(filePath_node, 'a') as csv_file:
        #         csv_writer = csv.DictWriter(csv_file, fieldnames=self.strain_header)

        #         info = {
        #             "Sim_step": 0,
        #             "lamda_xx": 0,
        #             "lamda_yy": 0,
        #             "lamda_zz": 0,
        #             "lamda_yz": 0,
        #             "lamda_xz": 0,
        #             "lamda_xy": 0
        #         }
        #         csv_writer.writerow(info)

        # filePath_node = "strain_data_rl/reward_cost_record.csv"
        # try:
        #     os.remove(filePath_node)
        # except:
        #     print("Error while deleting file ", filePath_node)
        # self.header = ["Sim_step", "cost", "reward"]
        # with open(filePath_node, "w", newline="") as csv_file:
        #     csv_writer = csv.DictWriter(csv_file, fieldnames=self.header)
        #     csv_writer.writeheader()
        # with open(filePath_node, 'a') as csv_file:
        #     csv_writer = csv.DictWriter(csv_file, fieldnames=self.header)
        #     info = {
        #         "Sim_step": 0,
        #         "cost": 0,
        #         "reward": 0
        #     }
        #     csv_writer.writerow(info)

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

    if reward >= 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    if cost <= 0.00003:
        reward += 1
        return True, reward
    if root.Whisker_node.Articulation_system.angleIn.value >= round(60*m.pi/180,4):
        return True, reward

    return False, reward
               

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.whisker_node = self.root.Whisker_node
        self.max_incr = 5*m.pi/180
    def _rotate(self, incr):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value
        new_angleIn = current_angleIn + incr
        self.whisker_node.Articulation_system.angleIn.value = new_angleIn 

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
        duration: float
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
def translateWhisker(whisker, direction):
    """Function to translate finger.

    Parameters:
    ----------
        fingers: list
            The fingers.
        direction: [vec_x, vec_y, vec_z]
            Translation vector.

    Returns:
    -------
        None.

    """
    
    possible = True

    mecaobject = whisker.dofs
    res = getTranslated(mecaobject.rest_position.value,  direction)
    if res is None:
        possible = False

    if possible:
        mecaobject.rest_position.value = res

def getTranslated(points, vec):
    """Translate a point.

    Parameters:
    ----------
        points: list
            List of points [x, y, z]
        vec: [vec_x, vec_y, vec_z]
            Translation vector.
    """
    r = []

    for v in points:
        x = v[0]+vec[0]
        y = v[1]+vec[1]
        z = v[2]+vec[2]

        r.append([x, y, z])

    return r
    

def pressurize(whisker, pressure, cavity):
    """Change the pressure value of a specific chamber in the whisker.

    Parameters:
    ----------
        whisker:
            The whisker.
        cavity:
            Which chamber is applied
        pressure: float
            The applied pressure.

    Returns:
    -------
        None.

    """
    pass


def getState(root, no_chamber):
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
    pressure = []
    for i in range(3):
        if i <= no_chamber-1:
            pressure.append(chamber_node.getChild(f'cavity{i}').SurfacePressureConstraint.getData('value').value[0].tolist())
        else:
            pressure.append(0.0)
    print("pressure = ",pressure)

    rot_angle = root.Whisker_node.Articulation_system.angleIn.value
    strain_zz = root.Whisker_node.Whisker.MechanicalModel.FEM.totalstrain.value[0][2].tolist()

    state = [rot_angle] + [strain_zz] + pressure
    print("State = ", state)
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
    ref_pos = root.Whisker_node.Whisker.MechanicalModel.Ref_point.GoalMO.position.value.tolist()
    deformable_pos = root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value.tolist()
    rigid_pos = root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value.tolist()
    
    # fiber1_right_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_right.dofs.position.value.tolist()
    # fiber1_left_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_left.dofs.position.value.tolist()
    # fiber2_right_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value.tolist()
    # fiber2_left_pos = root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value.tolist()

    # arm_pos = root.Whisker_node.Articulation_system.ServoArm.dofs.position.value.tolist()
    arti_pos = root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value.tolist()
    body_pos = root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value.tolist()
    

    # goal = root.Goal.GoalMO.position.value.tolist()

    return [
            whisker_pos,
            ref_pos,
            deformable_pos,
            rigid_pos,
            # fiber1_right_pos,
            # fiber1_left_pos,
            # fiber2_right_pos,
            # fiber2_left_pos,
            # arm_pos,
            arti_pos,
            body_pos, 
            # goal
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
     ref_pos,
     deformable_pos,
     rigid_pos,
    # fiber1_right_pos,
    # fiber1_left_pos,
    # fiber2_right_pos,
    # fiber2_left_pos,
    #  arm_pos,
     arti_pos,
     body_pos,
    #  goal
    ] = pos
    root.Whisker_node.Whisker.MechanicalModel.dofs.position.value = np.array(whisker_pos)
    root.Whisker_node.Whisker.MechanicalModel.Ref_point.GoalMO.position.value = np.array(ref_pos)
    root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value = np.array(deformable_pos)
    root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value = np.array(rigid_pos)
    
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_right.dofs.position.value = np.array(fiber1_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_left.dofs.position.value = np.array(fiber1_left_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_left_pos)

    # root.Whisker_node.Articulation_system.ServoArm.dofs.position.value = np.array(arm_pos)
    root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value = np.array(arti_pos)
    root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value = np.array(body_pos)

    # root.Goal.GoalMO.position.value = np.array(goal)
