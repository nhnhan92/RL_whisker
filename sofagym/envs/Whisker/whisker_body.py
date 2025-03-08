import Sofa
import os
import csv
import math as m
from mesh.whisker_body import mesh_generator as body_mesh
# from mesh.whisker_chamber import mesh_generator as chamber_mesh

YoungsModulus = 0.8
PoissonRatio = 0.4

mesh_path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
# MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/length_'

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
            file_path = os.path.dirname(os.path.abspath(__file__))+"/fiber" + str(j+1) + "right" +"_info.csv"

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
                        fiber_left_dof[-1].append(-float(row['x (mm)'])) 
                        fiber_left_dof[-1].append(float(row['y (mm)']))
                        fiber_left_dof[-1].append(float(row['z (mm)']))

    for i in range(len(fiber_left_dof)):
        fiber_right_spring_info.append([])
        fiber_left_spring_info.append([])
        for j in range(0,int(len(fiber_left_dof[i])/3)-2):
            fiber_right_spring_info[-1].append([j, j+1, Ks, Kd, m.sqrt((fiber_right_dof[i][3*j]-fiber_right_dof[i][3*(j+1)])**2+
                                                                (fiber_right_dof[i][3*j+1]-fiber_right_dof[i][3*(j+1)+1])**2+
                                                                (fiber_right_dof[i][3*j+2]-fiber_right_dof[i][3*(j+1)+2])**2)])
            
            fiber_left_spring_info[-1].append([j, j+1, Ks, Kd, m.sqrt((fiber_left_dof[i][3*j]-fiber_left_dof[i][3*(j+1)])**2+
                                                                (fiber_left_dof[i][3*j+1]-fiber_left_dof[i][3*(j+1)+1])**2+
                                                                (fiber_left_dof[i][3*j+2]-fiber_left_dof[i][3*(j+1)+2])**2)])
    fiber_dof = [fiber_right_dof, fiber_left_dof]
    spring_info = [fiber_right_spring_info, fiber_left_spring_info]
    return fiber_dof, spring_info


def Whisker(visu, simu, name="Whisker",rotation=[0.0, 0.0, 0.0], 
            translation=[0.0, 0.0, 0.0], design_params = None,design_index = None):

    parent = Sofa.Core.Node(name)
    model = parent.addChild("MechanicalModel")
    body_mesh_path = mesh_path+f'body_{design_params["no_chamber"]}chamber_{design_params["body_length"]}.vtk'
    if os.path.exists(body_mesh_path):
        pass
    else:
        ### body mesh creator
        body_mesh(body_bot_radius = 12,
                    cone_angle = 85.5,
                    body_height = design_params["body_length"],
                    no_chamber = design_params["no_chamber"],
                    chamber_bot_radius = 10,
                    chamber_height = 24,
                    mesh_size = 4)
    
    model.addObject('MeshVTKLoader', name='loader', filename=body_mesh_path, 
                    scale3d=[1, 1, 1], translation=translation, rotation=rotation,createSubelements=1)
    model.addObject('TetrahedronSetTopologyContainer', name = "container", position="@loader.position", tetrahedra="@loader.tetrahedra")
    model.addObject('TetrahedronSetTopologyModifier')
    model.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

    model.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false', showIndicesScale='4e-5',
                    translation=translation)
    model.addObject('UniformMass', totalMass='0.00002')
    model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', 
                    poissonRatio=PoissonRatio,  youngModulus=YoungsModulus, strainmeasurementstatus = 1, 
                    strainmeasuringelements=[1785])
    model.addObject('BoxROI', name='smallend_Box', box=[-20, -20, design_params["body_length"]-0.5, 20, 20, design_params["body_length"]+0.5], 
                    drawBoxes=True, doUpdate=False)
    # model.addObject('OscillatorConstraint', name="OscillatingConstraint", oscillators="75 2 0 0 1 0 0 5 200")
    # model.addObject('LinearSolverConstraintCorrection', name='GCS')
    
    collisionmodel = model.addChild("CollisionMesh")
    # collisionmodel.addObject("MeshSTLLoader", name="loader", filename=mesh_path+'body_stl.stl',
    #                             rotation=[180.0, 0.0, 0.0], translation=translation, flipNormals = 0)
    collisionmodel.addObject('MeshTopology', src="@../loader",name='topology')
    collisionmodel.addObject('MechanicalObject')

    collisionmodel.addObject('PointCollisionModel')
    collisionmodel.addObject('LineCollisionModel')
    collisionmodel.addObject('TriangleCollisionModel')
    # collisionmodel.addObject('BarycentricMapping')
    collisionmodel.addObject('IdentityMapping')
    
    ##########################################
    # Chamber node                            #
    ##########################################
    chamber_node = model.addChild('Chamber')
    chamber_node.addData(name='no_chamber', type='int', help='Number of chambers',
                             value=design_params["no_chamber"])
    chamber_rot = [[-70,0,-120],[0,0,90]]
    for cavity_idx in range(design_params["no_chamber"]):
        rot_matrix = [rotation[0],rotation[1],cavity_idx*360/design_params["no_chamber"]+rotation[2]]
        if cavity_idx == 1:
            rot_matrix = [rotation[0],rotation[1],cavity_idx*360/design_params["no_chamber"]+rotation[2]]
        CavitySurfaceMeshPath = mesh_path+f'{design_params["no_chamber"]}chamber_{cavity_idx+1}.stl'
        cavity = chamber_node.addChild('cavity'+str(cavity_idx))
        cavity.addObject('MeshSTLLoader', name='loader', filename=CavitySurfaceMeshPath,
                         translation=translation,
                         rotation = rotation)
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject', name='cavity')
        cavity.addObject('SurfacePressureConstraint', name='pressure_input', template='Vec3', 
                        #  value=design_params[f"pressure_{cavity_idx+1}"],
                         value=0, 
                         flipNormal = 0,triangles='@topo.triangles', valueType='pressure')
        cavity.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
        
    ##########################################
    # Fibers node                            #
    ##########################################
    # fiber_node = model.addChild("fiber")
    # fiber_dof, spring_info = fiber_construction(Ks = 1e4, Kd = 5)
    # for chamber in range(len(chamber_name)):
    #     for fiber_idx in range (2):
    #         fiber = fiber_node.addChild('fiber'+str(fiber_idx+1)+"_"+chamber_name[chamber])
    #         # fiber = parent.addChild(name+chamber[cavity_idx])
    #         fiber.addObject("MechanicalObject", template="Vec3", name="dofs",
    #                         position=fiber_dof[chamber][fiber_idx],
    #                         showObject=1, showObjectScale=1,translation=[0, 0, 0.1])
    #         fiber.addObject('MeshTopology', name='lines', lines=[[i, i + 1] for i in range(len(fiber_dof[chamber][fiber_idx])-1)]) 
    #         fiber.addObject('UniformMass', totalMass=0.000008)
    #         # fiber.addObject("FixedConstraint", name="FixedConstraint", indices=[0])
    #         fiber.addObject("StiffSpringForceField", template="Vec3d", name="springs", showArrowSize=0.2, drawMode=1,spring=spring_info[chamber][fiber_idx])
    #         fiber.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
            
    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        # modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader")
        # modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1], translation=translation, rotation=rotation, color="yellow")
        # modelVisu.addObject('BarycentricMapping')

        modelVisu.addObject("OglModel", name="Visual", template="Vec3d", color="yellow")
        modelVisu.addObject("IdentityMapping", template="Vec3d,Vec3d", name="visualMapping", input="@../dofs", output="@Visual")


    return parent
