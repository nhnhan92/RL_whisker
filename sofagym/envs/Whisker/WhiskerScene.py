
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

def Whisker_node(name="Whisker_node", design_params = None,design_index = None,
                translation = [0,0,0], rotation = [0,0,0], strain_gauge = [10,2]):
    
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
        cross_section_radius = 12-strain_gauge[0]/m.tan(85.5*m.pi/180)
        nominal_offset = m.sqrt(cross_section_radius**2 + strain_gauge[0]**2)
        alpha = rotation[0]*m.pi/180 - m.atan(cross_section_radius/strain_gauge[0])
        gauge_y_offset = -nominal_offset*m.cos(alpha)
        gauge_z_offset = -nominal_offset*m.sin(alpha)-1.25
        strain_measuring_box = addOrientedBoxRoi(
            self,
            position=[j for j in deformableObject.dofs.rest_position.value],
            name="strain_measuring_Box",
            translation=[0,gauge_y_offset,gauge_z_offset],
            eulerRotation=eulerRotation,
            scale=[10, 2, 10],
            drawBoxes=1,
        )
        strain_measuring_box.tetrahedra.value = self.Whisker.MechanicalModel.container.tetrahedra.value
        strain_measuring_box.drawTetrahedra.value = True
        strain_measuring_box.init()
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
                          translation=translation, design_params = design_params)
    self.addChild(whiskernode)
    arti_system = ActuatedArm(
        name="Articulation_system", translation=translation, rotation=[0,0,0],
    )
    self.addChild(arti_system)

    __rigidify(self,translation=translation,eulerRotation=rotation)
    __attachToSkin(self)
    return self

    
precontact_distance = 1+2+3# 1:precontact plus radius of the pole, 1: contactdistance, 3: pole radius. 
pole_init_pos = contact_pos[0]
pole_simu_pos = [pole_init_pos[0]+precontact_distance,
                                            pole_init_pos[1],pole_init_pos[2]]

def createScene(root, config={"source": [0, 0, 160],
                                "target": [0, 1, 0],
                              "goalPos": [0, 0, 100],
                                "init_states": [1,0,0,0],
                                "zFar":4000,
                                "design_params": [60, 2,30,2,0.1,0.1],
                                "scale_factor": 10
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

    root.addObject('RequiredPlugin', name="SofaPython3")
    root.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Solver")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('RequiredPlugin', name="STLIB")
    root.addObject('RequiredPlugin', name="ArticulatedSystemPlugin")
    root.addObject('RequiredPlugin', name="Sofa.GL.Component.Shader")
    
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
    root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.00001), name='Response',
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
                     "chamber_length": config["design_params"][2],
                     "thickness": float(config["design_params"][3]),
                     "pressure_1": float(config["design_params"][4]),
                    "pressure_2": float(config["design_params"][5])
                    }
    init_whisker_angle = 50
    whisker_rot = [init_whisker_angle,0,0]
    a = Whisker_node(design_params=design_params,
                    translation=[0,0,0],
                    rotation=whisker_rot,
                    strain_gauge = [10,2]) # 10: strain gauge pos; 2: gauge length
    whisker_model = root.addChild(a)

    ##  PLANE node
    contactDistance = root.localmindistance.contactDistance.value
    amp = [0,5,0,0,0,0]
    plane_offset = m.cos(init_whisker_angle*m.pi/180)*(12-design_params['body_length']/m.tan(85.5*m.pi/180))
    oscilater_plane_trans = [0,
                             -m.sin(init_whisker_angle*m.pi/180)*design_params['body_length'] - plane_offset,
                             0]
    
    oscilater_plane_rot = [90,90,0]
    plane = oscilate_plane(root,visu=visu,amp= amp,pulse=5,phase=-90,
                           translation=oscilater_plane_trans, rotation=oscilater_plane_rot, sphere_r=None)

    # Add Controller and reward + goal for RL
    # root.addObject(WhiskerController(node=root, name='whisker_controller',body_length=body_length))  # Controller

    # SofaGym Env Components
    root.addObject(StateInitializer(name="StateInitializer", rootNode=root, init_states=config['init_states']))
    root.addObject(rewardShaper(name="Reward", rootNode=root, scale_factor=config['scale_factor'], force_threshold = 0.1))
    root.addObject(applyAction(name="applyAction", root=root,config = config))
    # setData(whisker_model.Articulation_system.ServoMotor.Articulation.ServoWheel.dofs, showObject=1, showObjectScale=20,
    # drawMode=2, showColor=[1., 1., 0., 1.])
    root.addObject(AnimationManagerController(root, name="AnimationManager"))
    # root.addObject(AnimationManager(root))
    
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
    
    root=Sofa.Core.Node("root")
    
    createScene(root)
    Sofa.Simulation.init(root)
    USE_GUI = True
    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    # Sofa.Simulation.reset(self.root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    
    print("End of simulation.")


if __name__ == '__main__':
    main()
    