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
        goal_pos: coordinates
            The position of the goal.
        effMO: <MechanicalObject>
            The mechanical object of the element to move.
        cost:
            Evolution of the distance between object and goal.

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
        self.goal_pos = None
        if kwargs["goalPos"]:
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
    # def onKeypressedEvent(self, e):
    #     c = e['key']


    def onAnimateBeginEvent(self, event):
        pass

    
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
        self.current_strain.append(self.strain[1][4])
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
    # Hàm cập nhật đồ thị

    def update(self):
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
               
class goalSetter(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        update: Initialize the value of cost.

    Arguments:
    ---------
        goalMO: <MechanicalObject>
            The mechanical object of the goal.
        goalPos: coordinates
            The coordinates of the goal.

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

        self.goalMO = None
        if kwargs["goalMO"]:
            self.goalMO = kwargs["goalMO"]
        self.goalPos = None
        if kwargs["goalPos"]:
            self.goalPos = kwargs["goalPos"]
            
        self.count = 1
    def update(self):
        """Set the position of the goal.

        This function is used as an initialization function.
        """

        with self.goalMO.position.writeable() as position:
            position += self.goalPos

    def set_mo_pos(self, goal):
        pass

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.const_angle = -0.8*m.pi/180
        self.whisker_node = self.root.Whisker_node
        self.angle_limit = round(60*m.pi/180,4)
        self.max_incr = 0.2*m.pi/180

    def _rotate(self, incr):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value
        new_angleIn = current_angleIn + incr
        print("increment of action =", incr)
        if new_angleIn < round(60*m.pi/180,4):
            self.whisker_node.Articulation_system.angleIn.value = new_angleIn 
        else:
            self.whisker_node.Articulation_system.angleIn.value = round(60*m.pi/180,4)

    def _normalizedAction_to_action(self, action):
        return self.max_incr*action/2 + self.max_incr/2

    def compute_rot_action(self, action, nb_step):
        current_angleIn = self.whisker_node.Articulation_system.angleIn.value
        incr= (self.const_angle+self._normalizedAction_to_action(action))/nb_step
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
    print('start action')
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

def rotateWhisker(whisker, rot, direction,factor):
    """Function to rotate finger.

    Parameters:
    ----------
        direction: list
            1-Clockwise (CW), 2-Counter Clockwise (CCW)
        rot: float
            The rotation angle.

    Returns:
    -------
        None.

    """
    const_rot = -0.001
    current_angleIn = whisker.Articulation_system.angleIn.value
    # print(current_angleIn)
    # print(whisker.Whisker.MechanicalModel.dofs.position.value[0])

    if direction == "CW":
        whisker.Articulation_system.angleIn.value = current_angleIn + const_rot + rot
    elif direction == "CCW":
        whisker.Articulation_system.angleIn.value = current_angleIn + const_rot + rot
    elif direction == "Stand":
        whisker.Articulation_system.angleIn.value = current_angleIn + const_rot
    

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

def _getGoalPos(root):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return root.Goal.GoalMO.position.value[0]


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
    cs = 3
    right_pressure = root.Whisker_node.Whisker.MechanicalModel.Chamber.cavity0.SurfacePressureConstraint.getData('value').value.tolist()
    # right_pressure[0] /=10
    left_pressure = root.Whisker_node.Whisker.MechanicalModel.Chamber.cavity1.SurfacePressureConstraint.getData('value').value.tolist()
    # left_pressure[0] /=10

    rot_angle = [root.Whisker_node.Articulation_system.angleIn.value]

    goal_pos = _getGoalPos(root)
    goal_pos = [round(float(k), cs) for k in goal_pos]

    # whisker_tips = [round(k, cs) for k in root.Whisker_node.Whisker.MechanicalModel.Ref_point.GoalMO.position.value[0].tolist()]
    state = right_pressure + left_pressure + rot_angle

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

    arm_pos = root.Whisker_node.Articulation_system.ServoArm.dofs.position.value.tolist()
    arti_pos = root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value.tolist()
    body_pos = root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value.tolist()
    

    goal = root.Goal.GoalMO.position.value.tolist()

    return [
            whisker_pos,
            ref_pos,
            deformable_pos,
            rigid_pos,
            # fiber1_right_pos,
            # fiber1_left_pos,
            # fiber2_right_pos,
            # fiber2_left_pos,
            arm_pos,
            arti_pos,
            body_pos, 
            goal
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
    #  fiber1_right_pos,
    # fiber1_left_pos,
    # fiber2_right_pos,
    # fiber2_left_pos,
     arm_pos,
     arti_pos,
     body_pos,
     goal
    ] = pos
    root.Whisker_node.Whisker.MechanicalModel.dofs.position.value = np.array(whisker_pos)
    root.Whisker_node.Whisker.MechanicalModel.Ref_point.GoalMO.position.value = np.array(ref_pos)
    root.Whisker_node.RigidifiedBase.DeformableParts.dofs.position.value = np.array(deformable_pos)
    root.Whisker_node.RigidifiedBase.RigidParts.dofs.position.value = np.array(rigid_pos)
    
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_right.dofs.position.value = np.array(fiber1_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber1_left.dofs.position.value = np.array(fiber1_left_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_right_pos)
    # root.Whisker_node.Whisker.MechanicalModel.fiber.fiber2_right.dofs.position.value = np.array(fiber2_left_pos)

    root.Whisker_node.Articulation_system.ServoArm.dofs.position.value = np.array(arm_pos)
    root.Whisker_node.Articulation_system.ServoMotor.Articulation.dofs.rest_position.value = np.array(arti_pos)
    root.Whisker_node.Articulation_system.ServoMotor.ServoBody.dofs.position.value = np.array(body_pos)

    root.Goal.GoalMO.position.value = np.array(goal)
