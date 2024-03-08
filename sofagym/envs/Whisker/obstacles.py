import os
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