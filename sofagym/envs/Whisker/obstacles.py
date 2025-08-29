import os
import math as m
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

PlaneMesh = path + 'plane.stl'
PoleMeshPath = path + 'pole.stl'
def pole(rootNode, visu, name="pole", translation = [0,0,0]):
    model = rootNode.addChild(name)
    model.addObject('EulerImplicitSolver', name='odesolver')
    model.addObject('SparseLDLSolver',template='CompressedRowSparseMatrixd', name='linearSolver')
    # EigenSimplicialLDLT SparseLDLSolver
    model.addObject('GenericConstraintCorrection') 
    
    model.addObject('MechanicalObject',name='dofs',template='Rigid3d',translation = translation)
    model.addObject('BoxROI', name='Box', box=[-20, -20, -0.5, 20, 20, 0.5], 
                    drawBoxes=True, doUpdate=False)
    model.addObject('RestShapeSpringsForceField', name = "RestShapeSpringsForceField_bottom",
                        points='@BOX.indices', stiffness=1e12,angularStiffness=1e12)
    # model.addObject("SphereCollisionModel", name="pole_colli", radius="2")
    colli =model.addChild("colli")
    colli.addObject('MeshSTLLoader', name='pole_loader', filename=PoleMeshPath,flipNormals = 1)
    colli.addObject('MeshTopology', src='@pole_loader', name='topo')
    colli.addObject('MechanicalObject')
    colli.addObject('PointCollisionModel')
    colli.addObject('LineCollisionModel')
    colli.addObject('TriangleCollisionModel')
    colli.addObject('RigidMapping')
    if visu:
        visual =model.addChild("visual")
        visual.addObject('MeshSTLLoader', name='pole_loader', filename=PoleMeshPath,flipNormals = 1)
        visual.addObject('OglModel', name='Visual', color='green',src = '@pole_loader')
        visual.addObject('RigidMapping')

def euler2quarter(pos,euler):
        out = []
        # low_bound += [pos[i]-amp[i] for i in range(len(pos))]
        cz = m.cos(euler[2]*0.5*m.pi/180.0)
        sz = m.sin(euler[2]*0.5*m.pi/180.0)
        cy = m.cos(euler[1]*0.5*m.pi/180.0)
        sy = m.sin(euler[1] * 0.5*m.pi/180.0)
        cx = m.cos(euler[0] * 0.5*m.pi/180.0)
        sx = m.sin(euler[0] * 0.5*m.pi/180.0)
        qx = sx*cy*cz-cx*sy*sz
        qy = cx*sy*cz+sx*cy*sz
        qz = cx*cy*sz-sx*sy*cz
        w = cx*cy*cz+sx*sy*sz
        out = pos+[qx,qy,qz,w]
        return out

def plane(rootNode, visu, name="plane", rotation = [0,0,0], translation = [0,0,0], sphere_r = 0):
    plane = rootNode.addChild(name)   
    plane.addObject('MeshSTLLoader', name="loader", filename=PlaneMesh, triangulate="true", flipNormals=1, 
                    translation=translation, rotation = rotation)
    plane.addObject('MeshTopology', src="@loader")
    plane.addObject('MechanicalObject',src="@loader")
    
    if sphere_r != None:
        plane.addObject("SphereCollisionModel", radius=str(sphere_r))
    else:
        plane.addObject('PointCollisionModel')
        plane.addObject('LineCollisionModel')
        plane.addObject('TriangleCollisionModel') 
    if visu:
        plane.addObject('OglModel', name='Visual', src='@loader', color='green')
    return plane

def oscilate_plane(rootNode, visu, name="plane", amp = [0,0,0,0,0,0], pulse = 1, phase = 1,
                   rotation = [0,0,0], translation = [0,0,0], sphere_r = 0):
    plane = rootNode.addChild(name)   
    plane.addObject('MechanicalObject', name = "oscilated_dof", template = "Rigid3d", 
                    translation = translation, rotation = rotation,
                    showObject = 0,showObjectScale = 5)
    dof_pos = euler2quarter(pos=translation,euler=rotation)
    oscillators = [0]+dof_pos + amp+[pulse]+[phase]
    plane.addObject('UniformMass', totalMass = 0.000001)
    plane.addObject('OscillatorConstraint',template="Rigid3", name="osc2", oscillators=oscillators)
    colli_model = plane.addChild('collision')
    if sphere_r != None:
        colli_model.addObject("SphereCollisionModel", radius=str(sphere_r))
    else:
        colli_model.addObject('MeshSTLLoader', name="loader", filename=PlaneMesh, triangulate="true", flipNormals=1)
        colli_model.addObject('MeshTopology', src='@loader', name='topology')
        colli_model.addObject('MechanicalObject', name='collisMech')
        colli_model.addObject('PointCollisionModel')
        colli_model.addObject('LineCollisionModel')
        colli_model.addObject('TriangleCollisionModel')
        colli_model.addObject('RigidMapping')

    visual_model = plane.addChild('visual')
    if visu:
        visual_model.addObject('OglModel', name='Visual', src='@../collision/loader', color='green')
        visual_model.addObject('RigidMapping', input = '@..', output = '@Visual')
    return plane