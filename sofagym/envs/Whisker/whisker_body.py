import Sofa
import os
import csv
import math as m
from fibers import generate_inverted_truncated_conical_double_helix
# from mesh.whisker_body import mesh_generator as body_mesh
# from mesh.whisker_chamber import mesh_generator as chamber_mesh

YoungsModulus = 500
PoissonRatio = 0.4

mesh_path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
# MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/length_'

def fiber_construction(Ks = 1e3, 
                       Kd = 5,
                       H=25.0,       # "design" cone height
                        R=11.0,       # big-end radius at z=0
                        alpha=90-85.5,   # cone half-angle in degrees
                        pitch=4.0,    # vertical distance per 2π turn
                        num_points=1000,    
                        no_chamber = 2,
                        offset = 1,
                        plot = False,
                        save_data = False):
    
    fiber_spring_info = [[],[]]
    fiber1, fiber2 = generate_inverted_truncated_conical_double_helix(H=H,       # "design" cone height
                                                                    R=R,       # big-end radius at z=0
                                                                    alpha=alpha,   # cone half-angle in degrees
                                                                    pitch=pitch,    # vertical distance per 2π turn
                                                                    num_points=num_points,    
                                                                    no_chamber = no_chamber,
                                                                    offset = offset,
                                                                    plot = plot,
                                                                    save_data = save_data)
    fiber_dof = [fiber1,fiber2]
    # # Specify the path to your CSV file
    # for j in range(2):
    #     file_path = "/home/nhnhan/Desktop/sofa/SofaGym/sofagym/envs/Whisker/fiber"+str(j+1)+"_info.csv"

    #     # Open the CSV file
    #     with open(file_path, 'r') as file:
    #         # Create a CSV reader object as a dictionary reader
    #         csv_reader = csv.DictReader(file)
    #         for row in csv_reader:
    #             fiber_dof[j].append(float(row['X'])) 
    #             fiber_dof[j].append(float(row['Y']))
    #             fiber_dof[j].append(float(row['Z']))

    for i in range(len(fiber_spring_info)):
        for j in range(len(fiber_dof[i])-1):
            fiber_spring_info[i].append([j, j+1, Ks, Kd, m.sqrt((fiber_dof[i][j][0]-fiber_dof[i][j+1][0])**2+
                                                                (fiber_dof[i][j][1]-fiber_dof[i][j+1][1])**2+
                                                                (fiber_dof[i][j][2]-fiber_dof[i][j+1][2])**2)])
            
            
    return fiber_dof, fiber_spring_info


def Whisker(visu, simu, name="Whisker",rotation=[0.0, 0.0, 0.0], 
            translation=[0.0, 0.0, 0.0], design_params = None):

    parent = Sofa.Core.Node(name)
    model = parent.addChild("MechanicalModel")
    parent.addData(name='body_length', type='float', help='body length',
                             value=design_params["body_length"])
    parent.addData(name='no_chamber', type='float', help='no chamber',
                             value=design_params["no_chamber"])
    parent.addData(name='chamber_length', type='float', help='chamber length',
                             value=design_params["chamber_length"])
    parent.addData(name='thickness', type='float', help='thickness',
                             value=design_params["thickness"])
    parent.addData(name='force', type='vector<float>', help='Reaction Force',
                             value=[0.0,0.0,0.0])
    body_mesh_path = mesh_path + f'mesh_body/body_{design_params["no_chamber"]}chamber_{design_params["body_length"]}_{design_params["chamber_length"]}_{design_params["thickness"]}.vtk'
    # if os.path.exists(body_mesh_path):
    #     pass
    # else:
    #     ### body mesh creator
    #     body_mesh(body_bot_radius = 12,
    #                 cone_angle = 85.5,
    #                 body_height = design_params["body_length"],
    #                 no_chamber = design_params["no_chamber"],
    #                 chamber_bot_radius = 12 - design_params["thickness"],
    #                 chamber_height = design_params["chamber_length"],
    #                 mesh_size = 4)
    
    model.addObject('MeshVTKLoader', name='loader', filename=body_mesh_path, 
                    scale3d=[1, 1, 1], translation=translation, rotation=rotation,createSubelements=1)
    model.addObject('TetrahedronSetTopologyContainer', name = "container", position="@loader.position", tetrahedra="@loader.tetrahedra")
    model.addObject('TetrahedronSetTopologyModifier')
    model.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

    model.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false', showIndicesScale='4e-5',
                    translation=translation)
    model.addObject('UniformMass', totalMass='0.02')
    fem = model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', 
                    poissonRatio=PoissonRatio,  youngModulus=YoungsModulus, strainmeasurementstatus = 1, 
                    strainmeasuringelements=[0],computeVonMisesStress = 2)
    fem.addData(name='ave_strain', type='float', help='average strain',
                             value=0)
    model.addObject('BoxROI', name='smallend_Box', box=[-20, -20, design_params["body_length"]-0.5, 20, 20, design_params["body_length"]+0.5], 
                    drawBoxes=False, doUpdate=False)
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
    chamber_rot = [[-70,0,-120],[0,0,90]]
    for cavity_idx in range(design_params["no_chamber"]):
        rot_matrix = [rotation[0],rotation[1],cavity_idx*360/design_params["no_chamber"]+rotation[2]]
        if cavity_idx == 1:
            rot_matrix = [rotation[0],rotation[1],cavity_idx*360/design_params["no_chamber"]+rotation[2]]
        CavitySurfaceMeshPath = mesh_path+f'mesh_chamber/{design_params["no_chamber"]}chamber_{design_params["chamber_length"]}_{float(design_params["thickness"])}_idx{cavity_idx+1}.stl'
        cavity = chamber_node.addChild('cavity'+str(cavity_idx))
        cavity.addObject('MeshSTLLoader', name='loader', filename=CavitySurfaceMeshPath,
                         translation=translation,
                         rotation = rotation)
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject', name='cavity')
        cavity.addObject('SurfacePressureConstraint', name='pressure_input', template='Vec3', 
                         value=design_params[f"pressure_{cavity_idx+1}"],
                        #  value=0, 
                         flipNormal = 1,triangles='@topo.triangles', valueType='pressure')
        cavity.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
    ##########################################
    # Fibers node                            #
    ##########################################
    fiber_node = model.addChild("fiber")
    for i in range(design_params["no_chamber"]):
        fiber_dof, spring_info = fiber_construction( Ks = 1e4, 
                                                Kd = 5,
                                                H=design_params["chamber_length"],       # "design" cone height
                                                pitch=2.0,    # vertical distance per 2π turn
                                                num_points=200,    
                                                no_chamber = design_params["no_chamber"],
                                                offset = (-1)**(i+1) * 1)
        for fiber_idx in range (2):
            fiber = fiber_node.addChild(f'chamber{i+1}_fiber{str(fiber_idx+1)}')
            # fiber = parent.addChild(name+chamber[cavity_idx])
            fiber.addObject("MechanicalObject", template="Vec3", name="dofs",
                            position=fiber_dof[fiber_idx],
                            showObject=1, showObjectScale=1,translation=[0, 0, 0.1], rotation=rotation)
            fiber.addObject('MeshTopology', name='lines', lines=[[i, i + 1] for i in range(len(fiber_dof[fiber_idx])-1)]) 
            fiber.addObject('UniformMass', totalMass=0.000008)
            # fiber.addObject("FixedConstraint", name="FixedConstraint", indices=[0])
            fiber.addObject("StiffSpringForceField", template="Vec3d", name="springs", showArrowSize=0.2, drawMode=1,spring=spring_info[fiber_idx])
            fiber.addObject('BarycentricMapping', name='mapping', input = model.getLinkPath())
            
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
