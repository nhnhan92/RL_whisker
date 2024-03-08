from fibers import fibers
import math as m

import Sofa


def fiber_parameters(Ks = 1e3, Kd = 5):

    chamber_name = ["right", "left"]
    # Create fibers:
    fiber_right = fibers(cutting_plane = 1.2)
    fiber_left = fibers(cutting_plane = -1.2)

    fiber_right_dof = []
    fiber_left_dof = []
    fiber_right_spring_info = []
    fiber_left_spring_info = []
    for k in range(len(fiber_right)):
        fiber_right_dof.append([])
        for i in range(1,len(fiber_right[k])):
            for j in range(3):
                fiber_right_dof[-1].append(float(fiber_right[k][i][j])) 
    for k in range(len(fiber_left)):
        fiber_left_dof.append([])
        for i in range(1,len(fiber_left[k])):
            for j in range(3):
                fiber_left_dof[-1].append(float(fiber_left[k][i][j])) 

    for k in range(len(fiber_right)):
        fiber_right_spring_info.append([])
        fiber_left_spring_info.append([])
        for i in range(0,len(fiber_right[k])-2):
            fiber_right_spring_info[-1].append([i, i+1, Ks, Kd, m.sqrt((fiber_right_dof[k][3*i]-fiber_right_dof[k][3*(i+1)])**2+
                                                            (fiber_right_dof[k][3*i+1]-fiber_right_dof[k][3*(i+1)+1])**2+
                                                            (fiber_right_dof[k][3*i+2]-fiber_right_dof[k][3*(i+1)+2])**2)])
        for i in range(0,len(fiber_left[k])-2):
            fiber_left_spring_info[-1].append([i, i+1, Ks, Kd, m.sqrt((fiber_left_dof[k][3*i]-fiber_left_dof[k][3*(i+1)])**2+
                                                            (fiber_left_dof[k][3*i+1]-fiber_left_dof[k][3*(i+1)+1])**2+
                                                            (fiber_left_dof[k][3*i+2]-fiber_left_dof[k][3*(i+1)+2])**2)])

    fiber_dof = [fiber_right_dof, fiber_left_dof]
    spring_info = [fiber_right_spring_info, fiber_left_spring_info]

    return fiber_dof, spring_info

def fiber_node(name, parent, Ks, Kd):

    self = parent.addChild(name)
    chamber_name = ["right", "left"]
    fiber_dof, spring_info = fiber_parameters(Ks=Ks, Kd=Kd)
    #########################################
    # Fibers                                 #
    ######################################### 
    # for chamber in range(len(chamber_name)):
    for chamber in range(len(chamber_name)):
        for fiber_idx in range (len(chamber_name)):
            fiber = self.addChild('fiber'+str(fiber_idx)+"_"+chamber_name[chamber])
            # fiber = parent.addChild(name+chamber[cavity_idx])
            fiber.addObject("MechanicalObject", template="Vec3", name="DOF",
                            position=fiber_dof[chamber][fiber_idx],
                            showObject=True, showObjectScale=3,translation=[0, 0, 0.1])
            fiber.addObject('MeshTopology', name='lines', lines=[[i, i + 1] for i in range(len(fiber_dof[chamber][fiber_idx])-1)]) 
            fiber.addObject('UniformMass', totalMass=0.000008)
            fiber.addObject("FixedConstraint", name="FixedConstraint", indices=[0])

            fiber.addObject("StiffSpringForceField", template="Vec3d", name="springs", showArrowSize=1, drawMode=1,spring=spring_info[chamber][fiber_idx])
            fiber.addObject('BarycentricMapping', name='mapping', input = parent.getLinkPath())
            parent.addObject('MechanicalMatrixMapper', template="Vec3,Vec3", name="mapper"+str(fiber_idx)+"_"+chamber_name[chamber],
                                nodeToParse=fiber.getLinkPath(),  # where to find the forces to map
                                object1=parent.tetras.getLinkPath(), parallelTasks = 0)  # where to map the forces)  # in case of multi-mapping, here you can give the second parent
    
    return self


def chamber_node (name, parent,mesh_dir):

    self = parent.addChild(name)
    ##########################################
    # Constraint							 #
    ##########################################
    chamber = ["right", "left"]
    for cavity_idx in range(len(chamber)):
        CavitySurfaceMeshPath = mesh_dir+'whisker_chamber_'+chamber[cavity_idx]+'.stl'
        cavity = self.addChild('cavity_'+chamber[cavity_idx])
        cavity.addObject('MeshSTLLoader', name='loader', filename=CavitySurfaceMeshPath,rotation=[0, 0, 0])
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject', name='cavity')
        cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0, flipNormal = 1,
                            triangles='@topo.triangles', valueType='pressure')
        cavity.addObject('BarycentricMapping', name='mapping', input = parent.getLinkPath())
        # visual_cavity = cavity.addChild('visual')
        # visual_cavity.addObject("OglModel", name="Visual", template="Vec3d", color="blue")
        # visual_cavity.addObject("IdentityMapping", template="Vec3d,Vec3d", name="visualMapping", input="@../cavity", output="@Visual")
    
    return self