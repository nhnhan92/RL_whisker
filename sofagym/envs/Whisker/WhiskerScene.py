
import os
import numpy as np
import csv
import math as m
import Sofa
from splib3.animation import AnimationManagerController
from stlib3.components import addOrientedBoxRoi
from splib3.numerics import vec3
from stlib3.physics.mixedmaterial import Rigidify

import json
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
from WhiskerToolbox import rewardShaper, applyAction,StateInitializer
from Controllers import WhiskerController
from ext_data import ext_data
from articulation_system import ServoArm, ServoMotor, ActuatedArm
from obstacles import pole,plane,oscilate_plane
from whisker_body import Whisker
contact_list, contact_pos, rest_pos = ext_data()

body_length = 100
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/length_'
USE_GUI = True

def Whisker_node(name="Whisker_node", design_params = None,design_index = None,
                translation = [0,0,0], rotation = [0,0,0]):
    
    def __rigidify(self, translation = [0,0,0], eulerRotation = [0, 0.0, 0.0],scale = [40, 40, 0.5]):
        deformableObject = self.Whisker.MechanicalModel
        self.Whisker.init()
        name = "RigidifiedBase"
        rot_box = addOrientedBoxRoi(
            self,
            position=[j for j in deformableObject.dofs.rest_position.value],
            name="FixedBox",
            translation=vec3.vadd(translation, [0.0, 0, 0.0]),
            eulerRotation=eulerRotation,
            scale=scale,
            drawBoxes=1,
        )
        rot_box.init()
        groupIndices = []
        groupIndices.append([ind for ind in rot_box.indices.value])
        rigidifiedpart = Rigidify(self, deformableObject, groupIndices=groupIndices, frames=[[0, 0, 0]],name=name)
        # groupIndices phai la list chu ko duoc la array

    def __attachToSkin(self):
        rigidParts = self.RigidifiedBase.RigidParts
        arti_system.ServoMotor.Articulation.ServoWheel.addChild(rigidParts)

        rigidParts.addObject(
            "SubsetMultiMapping",
            input=[arti_system.ServoMotor.Articulation.ServoWheel.getLinkPath()],
            output="@./",
            indexPairs=[0,2],
        )
          ## idx[0]: node của output model, do đó sẽ lần lượt 0, 1, ..., n
        ### idx[1]: node ucar input model, cái này thì tùy vào node nào muốn được map tương ứng với idx[0]

    # def __constraintfreeend(self):
    #     freeend
    self = Sofa.Core.Node(name)

    self.addObject('EulerImplicitSolver', name='odesolver')
    self.addObject('SparseLDLSolver',template='CompressedRowSparseMatrixd', name='linearSolver')
    # EigenSimplicialLDLT SparseLDLSolver
    self.addObject('GenericConstraintCorrection') 

    whiskernode = Whisker(visu = True, simu = True, name="Whisker",rotation=rotation, 
                          translation=translation, design_params = design_params,design_index = design_index)
    self.addChild(whiskernode)
    arti_system = ActuatedArm(
        name="Articulation_system", translation=translation, rotation=[0,0,0],
    )
    self.addChild(arti_system)

    __rigidify(self,translation=translation,eulerRotation=rotation)
    __attachToSkin(self)
    return self


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position=[0.0, 0.0, 100.0])

    return goal_mo

def add_rand_node(root):
    goal = root.addChild("test")
    goal.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    goal.addObject('SparseLDLSolver', name='preconditioner', template="CompressedRowSparseMatrixMat3x3d")
    goal.addObject('MechanicalObject', template="Vec3d", name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position = [20, 0, 0])
    goal.addObject('UniformMass', name="m2",totalMass='0.00012')
    # goal.addObject('OscillatorConstraint', template="Vec3d", name="OscillatingConstraint", oscillators="0 25 0 0 20 0 0 2 10")
    
precontact_distance = 1+2+3# 1:precontact plus radius of the pole, 1: contactdistance, 3: pole radius. 
pole_init_pos = contact_pos[0]
pole_simu_pos = [pole_init_pos[0]+precontact_distance,
                                            pole_init_pos[1],pole_init_pos[2]]

def createScene(root, config={"source": [0, 0, 160],
                                "target": [0, 1, 0],
                              "goalPos": [0, 0, 100],
                                "init_states": [0] * 4,
                                "zFar":4000,
                                "design_params": [100,1,0.01,0,0],
                                "design_index": 0
                              }, mode='simu_and_visu'):
    # Chose the mode: visualization or computations (or both)
    from splib3.animation import animate
    from splib3.animation import AnimationManager
    from splib3.objectmodel import setData
    from sofagym.header import addVisu
    # from sofagym.header import addHeader
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True
    # addHeader(root)
    root.addObject('RequiredPlugin', name='BeamAdapter')
    root.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    root.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Correction")
    root.addObject('RequiredPlugin', name="SofaPython3")
    root.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Solver")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('RequiredPlugin', name="STLIB")
    root.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Projective")
    root.addObject('RequiredPlugin', name="Sofa.Component.Engine.Select")
    root.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    root.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Direct")
    root.addObject('RequiredPlugin', name="Sofa.Component.Mapping.MappedMatrix")
    root.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Dynamic")
    root.addObject('RequiredPlugin', name="Sofa.Component.Mass")
    root.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.FEM.Elastic")
    root.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.Spring")
    root.addObject('RequiredPlugin', name="Sofa.Component.StateContainer")
    root.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Constant")
    root.addObject('RequiredPlugin', name="Sofa.GL.Component.Rendering3D")
    root.addObject('RequiredPlugin', name="Sofa.GUI.Component")
    root.addObject('RequiredPlugin', name="Sofa.GL.Component.Shader")
    root.addObject('RequiredPlugin', name="Sofa.Component.Mapping.Linear")
    root.addObject('RequiredPlugin', name="Sofa.Component.Mapping.NonLinear")
    root.addObject('RequiredPlugin', name="ArticulatedSystemPlugin")
    root.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm') # Needed to use components [BVHNarrowPhase,BruteForceBroadPhase,CollisionPipeline]  
    root.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection') # Needed to use components [LocalMinDistance]  
    root.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry') # Needed to use components [LineCollisionModel,PointCollisionModel,TriangleCollisionModel]  
    root.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact') # Needed to use components [RuleBasedContactManager]  
    root.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward') # Needed to use components [EulerImplicitSolver]  
    root.addObject('RequiredPlugin', name='Sofa.Component.Setting') # Needed to use components [BackgroundSetting]  
    root.addObject('RequiredPlugin', name='Sofa.Component.Visual') # Needed to use components [InteractiveCamera,VisualStyle]

    
    root.addObject('VisualStyle', displayFlags='showVisualModels hideCollisionModels')
    source = [config["source"]]
    target = [config["target"]]
    position_spot = [[0, 100, 300]]
    direction_spot = [[0.5, 1, 10.5]]
    addVisu(root, config, position_spot, direction_spot, cutoff = 250)
    root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.1), name='Response',
                           response='FrictionContactConstraint')
    root.addObject('GenericConstraintSolver', name='GCS', tolerance=1e-4, maxIterations=1000,
                       computeConstraintForces=1)
    root.addObject('LocalMinDistance', contactDistance=5, alarmDistance=8, name='localmindistance',
                    angleCone=0.1)
    # root.addObject(AnimationManagerController(root))
    root.gravity.value = [0.0, -9810, 0.0]
    
    root.dt.value = 0.01

    # _, ref_pos = Whisker(root, visu, simu, name="Whisker",
    #        rotation=[180, 0.0, 0.0], translation=[0.0, 0.0, 0.0], ref_point = [0, 0, 90])
    
    design_params = {"body_length": config["design_params"][0],
                     "no_chamber": config["design_params"][1],
                     "pressure_1": config["design_params"][2],
                    "pressure_2": config["design_params"][3],
                    "pressure_3": config["design_params"][4]
                    }
    design_index = config["design_index"]
    whisker_rot = [70,0,0]
    a = Whisker_node(design_params=design_params,
                    design_index = design_index,
                    translation=[0,0,0],
                    rotation=whisker_rot)
    whisker_model = root.addChild(a)

    ##  PLANE node
    contactDistance = root.localmindistance.contactDistance.value
    oscilater_plane_trans = [0,
                             -m.sin(70*m.pi/180)*design_params['body_length'] - contactDistance,
                             40]
    oscilater_plane_rot = [90,0,0]
    amp = [0,5,0,0,0,0]
    # oscillators = [0] + 
    plane = oscilate_plane(root,visu=visu,amp= amp,pulse=5,phase=10,
                           translation=oscilater_plane_trans, rotation=oscilater_plane_rot, sphere_r=None)

    # Add Controller and reward + goal for RL
    # root.addObject(WhiskerController(node=root, name='whisker_controller',body_length=body_length))  # Controller

    # SofaGym Env Components
    root.addObject(StateInitializer(name="StateInitializer", rootNode=root, init_states=config['init_states']))
    root.addObject(rewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos']))
    root.addObject(applyAction(name="applyAction", root=root,config = config))
    # setData(whisker_model.Articulation_system.ServoMotor.Articulation.ServoWheel.dofs, showObject=1, showObjectScale=20,
    # drawMode=2, showColor=[1., 1., 0., 1.])
    root.addObject(AnimationManagerController(root, name="AnimationManager"))
    root.addObject(AnimationManager(root))
    
    def animation(target, factor):
        rot_angle = 90
        invalue = [factor*rot_angle*m.pi/180, 0]
        # invalue = [0, factor*5]
        target.angleIn.value = invalue
        # root.Whisker_node.Articulation_system.angles = [0,10,1]
    # animate(animation, {"target": whisker_model.Articulation_system}, duration=2, mode="pingpong") 
    ### factor = dt/duration => angular velocity = rot_angle*dt/duration (rad/dt) with step of angleIn per dt
    return root
def main():
    import SofaRuntime
    import Sofa.Gui
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("CImgPlugin")
    SofaRuntime.importPlugin("SofaBaseMechanics")
    SofaRuntime.importPlugin("SofaImplicitOdeSolver")
    
    root=Sofa.Core.Node("root")
    
    createScene(root)
    Sofa.Simulation.init(root)
    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    
    print("End of simulation.")


if __name__ == '__main__':
    main()
    