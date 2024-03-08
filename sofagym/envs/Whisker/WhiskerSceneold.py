
import os
import numpy as np
import csv
import math as m
from splib3.animation import AnimationManagerController

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
from WhiskerToolbox import rewardShaper, goalSetter
# from Whisker import Whisker
from Controllers import WhiskerController
from ext_data import ext_data
contact_list, contact_pos = ext_data()
precontact_distance = 1+1+3+2 # 1:precontact plus radius of the pole, 1: contactdistance, 3: pole radius. 2: sphere radius
pole_init_pos = contact_pos[0]
pole_simu_pos = [pole_init_pos[0]+precontact_distance,
                                            pole_init_pos[1],pole_init_pos[2]]
print(pole_init_pos)
print(pole_simu_pos)
YoungsModulus = 1
PoissonRatio = 0.45

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

VolumetricMeshPath = MeshesPath + 'whisker_init_vtk.vtk'
SurfaceMeshPath = MeshesPath + 'whisker_stl.stl'
PlaneMesh = path + 'plane.stl'
PoleMeshPath = path + 'pole.stl'
USE_GUI = True

def fiber_construction(Ks = 1e3, Kd = 5):
    chamber = ["right", "left"]
    fiber_right_dof = []
    fiber_left_dof = []
    fiber_right_spring_info = []
    fiber_left_spring_info = []

    # Specify the path to your CSV file
    for i in chamber:
        for j in range(2):
            if i == "right":
                fiber_right_dof.append([]) 
            else:
                fiber_left_dof.append([])
            file_path = os.getcwd()+"/fiber" + str(j+1) + i +"_info.csv"

            # Open the CSV file
            with open(file_path, 'r') as file:
                # Create a CSV reader object as a dictionary reader
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if i == "right":
                        fiber_right_dof[-1].append(float(row['x (mm)'])) 
                        fiber_right_dof[-1].append(float(row['y (mm)']))
                        fiber_right_dof[-1].append(float(row['z (mm)']))

                    else:
                        fiber_left_dof[-1].append(float(row['x (mm)'])) 
                        fiber_left_dof[-1].append(float(row['y (mm)']))
                        fiber_left_dof[-1].append(float(row['z (mm)']))

    for i in range(len(fiber_right_dof)):
        fiber_right_spring_info.append([])
        fiber_left_spring_info.append([])
        for j in range(0,int(len(fiber_right_dof[i])/3)-2):
            fiber_right_spring_info[-1].append([j, j+1, Ks, Kd, m.sqrt((fiber_right_dof[i][3*j]-fiber_right_dof[i][3*(j+1)])**2+
                                                                (fiber_right_dof[i][3*j+1]-fiber_right_dof[i][3*(j+1)+1])**2+
                                                                (fiber_right_dof[i][3*j+2]-fiber_right_dof[i][3*(j+1)+2])**2)])
            
            fiber_left_spring_info[-1].append([j, j+1, Ks, Kd, m.sqrt((fiber_left_dof[i][3*j]-fiber_left_dof[i][3*(j+1)])**2+
                                                                (fiber_left_dof[i][3*j+1]-fiber_left_dof[i][3*(j+1)+1])**2+
                                                                (fiber_left_dof[i][3*j+2]-fiber_left_dof[i][3*(j+1)+2])**2)])
    fiber_dof = [fiber_right_dof, fiber_left_dof]
    spring_info = [fiber_right_spring_info, fiber_left_spring_info]
    return fiber_dof, spring_info

def pole(rootNode, visu, name="pole", translation = [0,0,0]):
    model = rootNode.addChild(name)
    model.addObject('MeshSTLLoader', name='pole_loader', filename=PoleMeshPath, translation = translation)
    model.addObject('MeshTopology', src='@pole_loader', name='topo')
    model.addObject('MechanicalObject', name='pole_mec', src = '@pole_loader')

    model.addObject("SphereCollisionModel", name="pole_colli", radius="2")

    if visu:
        model.addObject('OglModel', name='Visual', color='green',src = "@pole_mec")

def Whisker(rootNode, visu, simu, name="Whisker",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], ref_point = [0,0,0]):

    model = rootNode.addChild(name)
    if simu:
        model.addObject('EulerImplicitSolver', name='odesolver')
        model.addObject('EigenSimplicialLDLT',template='CompressedRowSparseMatrixd', name='linearSolver')
        model.addObject('GenericConstraintCorrection') 
    model.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, scale3d=[1, 1, 1],
                    translation=translation, rotation=rotation)
    model.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra")
    model.addObject('TetrahedronSetTopologyModifier')
    model.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale='4e-5',
                    translation=translation, rotation=rotation)
    model.addObject('UniformMass', totalMass='0.0000012')
    model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', 
                    poissonRatio=PoissonRatio,  youngModulus=YoungsModulus, strainmeasurementstatus = 1, 
                    strainmeasuringelements=[1785,1392,1336,1561])

    c = model.addChild("FixedBox")
    c.addObject('BoxROI', name='BoxROI', box="-20 -20 -1 20 20 1", drawBoxes=True, doUpdate=False)
    c.addObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness='1e12')

    if simu:
        model.addObject('LinearSolverConstraintCorrection', name='GCS')
        
        collisionmodel = model.addChild("CollisionMesh")
        collisionmodel.addObject("MeshSTLLoader", name="loader", filename=SurfaceMeshPath,
                                 rotation=rotation, translation=translation)
        collisionmodel.addObject('MeshTopology', src="@loader")
        collisionmodel.addObject('MechanicalObject')

        collisionmodel.addObject('PointCollisionModel')
        collisionmodel.addObject('LineCollisionModel')
        collisionmodel.addObject('TriangleCollisionModel')

        collisionmodel.addObject('BarycentricMapping')

    ##########################################
    # Chamber node                            #
    ##########################################
    # chambers = model.addChild(chamber_node(name= "chamber", parent=model, mesh_dir = MeshesPath))
    chamber_node = model.addChild('Chamber')
    chamber_name = ["right", "left"]
    for cavity_idx in range(len(chamber_name)):
        CavitySurfaceMeshPath = MeshesPath+'whisker_chamber_'+chamber_name[cavity_idx]+'.stl'
        cavity = chamber_node.addChild('cavity_'+chamber_name[cavity_idx])
        cavity.addObject('MeshSTLLoader', name='loader', filename=CavitySurfaceMeshPath,rotation=[0, 0, 0])
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject', name='cavity')
        cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0, flipNormal = 1,
                            triangles='@topo.triangles', valueType='pressure')
        cavity.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
        
    ##########################################
    # Fibers node                            #
    ##########################################
    # fiber = model.addChild(fiber_node(name="fiber", parent=model,Ks = 1e5, Kd = 5))
    self = model.addChild("fiber")
    fiber_dof, spring_info = fiber_construction(Ks = 1e5, Kd = 5)
    for chamber in range(len(chamber_name)):
        for fiber_idx in range (len(chamber_name)):
            fiber = self.addChild('fiber'+str(fiber_idx)+"_"+chamber_name[chamber])
            # fiber = parent.addChild(name+chamber[cavity_idx])
            fiber.addObject("MechanicalObject", template="Vec3", name="DOF",
                            position=fiber_dof[chamber][fiber_idx],
                            showObject=0, showObjectScale=3,translation=[0, 0, 0.1])
            fiber.addObject('MeshTopology', name='lines', lines=[[i, i + 1] for i in range(len(fiber_dof[chamber][fiber_idx])-1)]) 
            fiber.addObject('UniformMass', totalMass=0.000008)
            fiber.addObject("FixedConstraint", name="FixedConstraint", indices=[0])

            fiber.addObject("StiffSpringForceField", template="Vec3d", name="springs", showArrowSize=1, drawMode=1,spring=spring_info[chamber][fiber_idx])
            fiber.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
            model.addObject('MechanicalMatrixMapper', template="Vec3,Vec3", name="mapper"+str(fiber_idx)+"_"+chamber_name[chamber],
                                nodeToParse=fiber.getLinkPath(),  # where to find the forces to map
                                object1=model.tetras.getLinkPath(), parallelTasks = 0)  # where to map the forces)  # in case of multi-mapping, here you can give the second parent

    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        # modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader")
        # modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1], translation=translation, rotation=rotation, color="yellow")
        # modelVisu.addObject('BarycentricMapping')

        modelVisu.addObject("OglModel", name="Visual", template="Vec3d", color="yellow")
        modelVisu.addObject("IdentityMapping", template="Vec3d,Vec3d", name="visualMapping", input="@../tetras", output="@Visual")

    ref = model.addChild("Ref_point")
    ref.addObject('VisualStyle', displayFlags="showCollisionModels")
    ref_pos = ref.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position=ref_point)
    ref.addObject('BarycentricMapping', name="Mapping_ref")
    return model, ref_pos


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor="green", position=[0.0, 0.0, 100.0])
    return goal_mo


def createScene(root, config={"source": [-600.0, -25, 100],
                              "target": [30, -25, 100],
                              "goalPos": [0, 0, 100]}, mode='simu_and_visu'):
    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True
    
    root.addObject('RequiredPlugin', name='BeamAdapter')
    root.addObject('RequiredPlugin', name='SofaOpenglVisual')
    root.addObject('RequiredPlugin', name="SofaMiscCollision")
    root.addObject('RequiredPlugin', name="SofaPython3")
    root.addObject('RequiredPlugin', name="SofaPreconditioner")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('RequiredPlugin', name="SofaConstraint")
    root.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    root.addObject('RequiredPlugin', name="SofaLoader")
    root.addObject('RequiredPlugin', name="SofaSparseSolver")

    root.addObject('RequiredPlugin', name="SofaDeformable")
    root.addObject('RequiredPlugin', name="SofaEngine")
    root.addObject('RequiredPlugin', name="SofaMeshCollision")
    root.addObject('RequiredPlugin', name="SofaMiscFem")
    root.addObject('RequiredPlugin', name="SofaRigid")
    root.addObject('RequiredPlugin', name="SofaSimpleFem")

    if visu:

        source = config["source"]
        target = config["target"]
        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideMappings hideForceFields hideWireframe')
        root.addObject("LightManager")
        root.addObject("DefaultVisualManagerLoop")
        spotLoc = [0, 0, source[2]]
        root.addObject("SpotLight", position=spotLoc, direction=[0.0, 0.0, -np.sign(source[2])])
        root.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=500)
        root.addObject('BackgroundSetting', color='white')
    if simu:
        
        root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
        root.addObject('BruteForceBroadPhase')
        root.addObject('BVHNarrowPhase')
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
        root.addObject('LocalMinDistance', contactDistance=1.0, alarmDistance=3.0, name='localmindistance',
                       angleCone=0.2)
        root.addObject('RuleBasedContactManager', responseParams="mu="+str(0.3), name='Response',
                           response='FrictionContactConstraint')

        root.addObject(AnimationManagerController(root))
        root.gravity.value = [0.0, -9810, 0.0]

    root.dt.value = 0.01

    goal_mo = add_goal_node(root)

    _, ref_pos = Whisker(root, visu, simu, name="Whisker",rotation=[180, 0.0, 0.0], translation=[0.0, 0.0, 0.0], ref_point = [0, 0, 90])
    
    pole(root,visu, name="pole",translation = pole_simu_pos)
    # plane = root.addChild('Plane')

    
    # plane.addObject('MeshSTLLoader', name="loader", filename=PlaneMesh, triangulate="true", flipNormals=1, 
    #                 translation=[120, 0, 80])
    # plane.addObject('MeshTopology', src="@loader")
    # plane.addObject('MechanicalObject',src="@loader")

    # plane.addObject("SphereCollisionModel", radius="6")
    # plane.addObject('OglModel', name='Visual', src='@loader', color='green')

    # Add Controller and reward + goal for RL
    root.addObject(WhiskerController(node=root, name='whisker_controller'))

    root.addObject(rewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos'], effMO=ref_pos))
    root.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))
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
    