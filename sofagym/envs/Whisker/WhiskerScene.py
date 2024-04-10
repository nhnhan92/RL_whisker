
import os
import numpy as np
import csv
import math as m
import Sofa
from splib3.animation import AnimationManagerController
from stlib3.components import addOrientedBoxRoi
from splib3.numerics import vec3
from stlib3.physics.mixedmaterial import Rigidify

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
from WhiskerToolbox import rewardShaper, goalSetter, applyAction
from Controllers import WhiskerController
from ext_data import ext_data
from articulation_system import ServoArm, ServoMotor, ActuatedArm
from obstacles import pole,plane
from whisker_body import Whisker
contact_list, contact_pos, rest_pos = ext_data()


body_length = 100
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/length_'
USE_GUI = True



def Whisker_node(name="Whisker_node", body_length = 0,no_chamber = 0):
    
    def __rigidify(self, translation = [0,0,0], eulerRotation = [0, 0.0, 0.0],scale = [40, 40, 0.5]):
        deformableObject = self.Whisker.MechanicalModel
        self.Whisker.init()
        name = "RigidifiedBase"
        rot_box = addOrientedBoxRoi(
            self,
            position=[list(j) for j in deformableObject.dofs.rest_position.value],
            # position = rest_pos,
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
            indexPairs=[0,1],
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

    whiskernode = Whisker(visu = True, simu = True, name="Whisker",rotation=[0, 0.0, 0.0], 
                          translation=[0.0, 0.0, 0.0], body_length=body_length,no_chamber=no_chamber)
    self.addChild(whiskernode)
    arti_system = ActuatedArm(
        name="Articulation_system", translation=[0.0, 0.0, 0.0], rotation=[180.0, 0.0, 0.0],
    )
    self.addChild(arti_system)
    __rigidify(self)
    __attachToSkin(self)
    return self, int(body_length)


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position=[0.0, 0.0, 100.0])

    return goal_mo

precontact_distance = 1+2+3# 1:precontact plus radius of the pole, 1: contactdistance, 3: pole radius. 
pole_init_pos = contact_pos[0]
pole_simu_pos = [pole_init_pos[0]+precontact_distance,
                                            pole_init_pos[1],pole_init_pos[2]]

def createScene(root, config={"source": [-600.0, -25, 200],
                              "target": [30, -25, 100],
                              "goalPos": [0, 0, 100],
                              "body": 100,
                              "no_chamber":2}, mode='simu_and_visu'):
    # Chose the mode: visualization or computations (or both)
    from splib3.animation import animate
    from splib3.animation import AnimationManager
    from splib3.objectmodel import setData
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

    source = config["source"]
    target = config["target"]
    root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                'hideMappings hideForceFields hideWireframe')
    root.addObject("LightManager")
    root.addObject("DefaultVisualManagerLoop")
    spotLoc = [0, 0, source[2]]
    root.addObject("SpotLight", position=spotLoc, direction=[0.0, 0.0, -np.sign(source[2])])
    root.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=700)
    root.addObject('BackgroundSetting', color='white')
    root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.1), name='Response',
                           response='FrictionContactConstraint')
    root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
    root.addObject('LocalMinDistance', contactDistance=2, alarmDistance=3, name='localmindistance',
                    angleCone=0.1)
    # root.addObject(AnimationManagerController(root))
    root.gravity.value = [0.0, -9810, 0.0]

    root.dt.value = 0.01

    # _, ref_pos = Whisker(root, visu, simu, name="Whisker",
    #        rotation=[180, 0.0, 0.0], translation=[0.0, 0.0, 0.0], ref_point = [0, 0, 90])

    a,body_length = Whisker_node(body_length=config["body"],no_chamber=config["no_chamber"])
    whisker_model = root.addChild(a)

    goal_mo = add_goal_node(root)
    
    small_end_radi = 12 - (int(body_length)/m.tan(85.5*m.pi/180))
    pole_simu_pos = [1+3+root.localmindistance.contactDistance.value+small_end_radi,0,body_length]
    #### POLE
    pole(root,visu, name="pole",translation = pole_simu_pos)

    ###  PLANE node
    # init_angle = 30
    # sweep_dist = 20   #half cource => full course = sweep dist *2
    # x_translate = m.sqrt(90*90-sweep_dist*sweep_dist)*m.cos(m.pi/2-init_angle*m.pi/180)
    # z_translate  = m.sqrt(90*90-sweep_dist*sweep_dist)*m.sin(m.pi/2-init_angle*m.pi/180)
    # plane(root,visu=visu,translation=[x_translate, 0, z_translate], rotation=[0,init_angle,0], sphere_r=None)

    # Add Controller and reward + goal for RL
    # root.addObject(WhiskerController(node=root, name='whisker_controller',body_length=body_length))  # Controller

    ref_point = [0, 0, 100]
    ref = whisker_model.Whisker.MechanicalModel.addChild("Ref_point")
    ref.addObject('VisualStyle', displayFlags="showCollisionModels")
    ref_pos = ref.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position=ref_point)
    ref.addObject('BarycentricMapping', name="Mapping_ref")

    root.addObject(rewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos']))
    root.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))
    root.addObject(applyAction(name="applyAction", root=root))
    # setData(whisker_model.Articulation_system.ServoMotor.Articulation.ServoWheel.dofs, showObject=1, showObjectScale=20,
    # drawMode=2, showColor=[1., 1., 0., 1.])
    
    root.addObject(AnimationManager(root))
    # def animation(target, factor):
    #     rot_angle = 60
    #     target.angleIn.value = factor * (-rot_angle*m.pi/180)
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
    