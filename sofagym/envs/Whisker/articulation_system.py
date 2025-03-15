import os
import Sofa
from stlib3.scene import Scene

dirPath = os.path.dirname(os.path.abspath(__file__)) + '/'


class ServoMotor(Sofa.Prefab):
    """A S90 servo motor

    This prefab is implementing a S90 servo motor.
    https://servodatabase.com/servo/towerpro/sg90

    The prefab ServoMotor is composed of:
    - a visual model
    - a mechanical model composed two rigids. One rigid is for the motor body
      while the other is implementing the servo rotating wheel.

    The prefab has the following parameters:
    - translation           to change default location of the servo (default [0.0,0.0,0.0])
    - rotation              to change default rotation of the servo (default [0.0,0.0,0.0,1])
    - scale                 to change default scale of the servo (default 1)
    - showServo             to control wether a visual model of the motor is added (default True)
    - showWheel             to control wether the rotation axis of the motor is displayed (default False)

    The prefab have the following property:
    - angle         use this to specify the angle of rotation of the servo motor
    - angleLimits   use this to set a min and max value for the servo angle rotation
    - position      use this to specify the position of the servo motor

    Example of use in a Sofa scene:

    def addScene(root):
        ...
        servo = ServoMotor(root)

        ## Direct access to the components
        servo.angle.value = 1.0
    """
    prefabParameters = [
        {'name': 'rotation', 'type': 'Vec3d', 'help': 'Rotation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'translation', 'type': 'Vec3d', 'help': 'Translation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'scale3d', 'type': 'Vec3d', 'help': 'Scale 3d', 'default': [1.0, 1.0, 1.0]}]

    prefabData = [
        {'name': 'minAngle', 'help': 'min angle of rotation (in radians)', 'type': 'float', 'default': -180},
        {'name': 'maxAngle', 'help': 'max angle of rotation (in radians)', 'type': 'float', 'default': 180},
        # {'name': 'angleIn', 'help': 'angle of rotation (in radians)', 'type': 'float', 'default': 0},
        # {'name': 'angleOut', 'help': 'angle of rotation (in degree)', 'type': 'float', 'default': 0}
        {'name': 'angleIn', 'help': 'angle of rotation (in radians)', 'type': 'vector<float>', 'default': [0,0]},
        {'name': 'angleOut', 'help': 'angle of rotation (in degree)', 'type': 'vector<float>', 'default': [0,0]}
    ]

    def __init__(self, *args, **kwargs):
        Sofa.Prefab.__init__(self, *args, **kwargs)

        # Servo body
        servoBody = self.addChild('ServoBody')
        servoBody.addObject('MechanicalObject', name='dofs', template='Rigid3',
                            position=[[0., 0., 0., 0., 0., 0., 1.]],
                            translation=self.translation.value, rotation=self.rotation.value,
                            scale3d=self.scale3d.value
                            )
        servoBody.addObject('FixedConstraint', indices=0)
        servoBody.addObject('UniformMass', totalMass=0.00001)

        # visual = servoBody.addChild('VisualModel')
        # visual.addObject('MeshSTLLoader', name='loader', filename='mesh/SG90_servomotor.stl',rotation = [0,90,0])
        # visual.addObject('MeshTopology', src='@loader')
        # visual.addObject('OglModel', color=[0.15, 0.45, 0.75, 0.7], writeZTransparent=True)
        # visual.addObject('RigidMapping', index=0)

        # Servo wheel
        angle = self.addChild('Articulation')
        angle.addObject('MechanicalObject', name='dofs', template='Vec1', position=[0,0],
                        rest_position=self.angleIn.getLinkPath())
        angle.addObject('ArticulatedHierarchyContainer', printLog=False)
        angle.addObject('RestShapeSpringsForceField', points=[0,1], stiffness=1e9)
        angle.addObject('UniformMass', totalMass=0.00001)

        servoWheel = angle.addChild('ServoWheel')
        servoWheel.addObject('MechanicalObject', name='dofs', template='Rigid3',
                             position=[[0., 0., 0., 0., 0., 0., 1.],[0., 0., 0., 0., 0., 0., 1.],
                                       [0., 0., 0., 0., 0., 0., 1.]
                                    ],
                             showObject =1, 
                             showObjectScale=5,translation=self.translation.value, rotation=self.rotation.value,
                             scale3d=self.scale3d.value
                             )
        servoWheel.addObject('ArticulatedSystemMapping', input1=angle.dofs.getLinkPath(), 
                             input2 = "@../../ServoBody/dofs",
                             output=servoWheel.dofs.getLinkPath())

        articulationCenter_sys = angle.addChild('ArticulationCenter')
        translation_center = articulationCenter_sys.addChild("translation")
        translation_center.addObject('ArticulationCenter', parentIndex=0, childIndex=1, posOnParent=[0., 0., 0.],
                                     posOnChild=[0., 0., 0.], articulationProcess = 0)
        trans_articulation = translation_center.addChild('Articulations')
        trans_articulation.addObject('Articulation', translation=True, rotation=False, axis=[0, 1, 0],
                               articulationIndex=0)
        
        rotation_center = articulationCenter_sys.addChild("rotation")
        rotation_center.addObject('ArticulationCenter', parentIndex=1, childIndex=2, posOnParent=[0., 0., 0.],
                                     posOnChild=[0., 0., 0.], articulationProcess = 0)
        rot_articulation = rotation_center.addChild('Articulations')
        rot_articulation.addObject('Articulation', translation=False, rotation=True, axis=[1, 0, 0],
                               articulationIndex=1)
        

class ServoArm(Sofa.Prefab):
    """ServoArm is a reusable sofa model of a servo arm for the S90 servo motor

       Parameters:
            parent:        node where the ServoArm will be attached
            mappingInput:  the rigid mechanical object that will control the orientation of the servo arm
            indexInput: (int) index of the rigid the ServoArm should be mapped to
    """

    prefabData = [
        {'name': 'mappingInputLink', 'type': 'string',
         'help': 'the rigid mechanical object that will control the orientation of the servo arm', 'default': ''},
        {'name': 'indexInput', 'type': 'int', 'help': 'index of the rigid the ServoArm should be mapped to',
         'default': 1}]

    def __init__(self, *args, **kwargs):
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def init(self):
        self.addObject('MechanicalObject',
                       name='dofs',
                       size=1,
                       template='Rigid3',
                       showObject=True,
                       showObjectScale=5,
                       translation=[0, 0, 0])

    def setRigidMapping(self, path):
        self.addObject('RigidRigidMapping', name='mapping', input=path, index=self.indexInput.value)

        # visual = self.addChild(VisualModel(visualMeshPath='data/mesh/SG90_servoarm.stl', translation=[0., -25., 0.],
        #                                    color=[1., 1., 1., 0.75]))
        # visual.OglModel.writeZTransparent = True
        # visual.addObject('RigidMapping', name='mapping')

class ActuatedArm(Sofa.Prefab):
    """ActuatedArm is a reusable sofa model of a S90 servo motor and the tripod actuation arm.
           Parameters:
             - translation the position in space of the structure
             - eulerRotation the orientation of the structure

           Structure:
           Node : {
                name : 'ActuatedArm'
                MechanicalObject     // Rigid position of the motor
                ServoMotor           // The s90 servo motor with its actuated wheel
                ServoArm             // The actuation arm connected to ServoMotor.ServoWheel
            }
    """
    prefabParameters = [
        {'name': 'rotation', 'type': 'Vec3d', 'help': 'Rotation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'translation', 'type': 'Vec3d', 'help': 'Translation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'scale', 'type': 'Vec3d', 'help': 'Scale 3d', 'default': [1.0, 1.0, 1.0]}
    ]

    prefabData = [
        {'name': 'angleIn', 'group': 'ArmProperties', 'help': 'angle of rotation (in radians) of the arm',
         'type': 'vector<float>', 'default': [0,0]},
        {'name': 'angleOut', 'group': 'ArmProperties', 'type': 'vector<float>', 'help': 'angle of rotation (in radians) of '
                                                                                'the arm', 'default': [0,0]}
        #        {'name': 'angleIn', 'group': 'ArmProperties', 'help': 'angle of rotation (in radians) of the arm',
        #  'type': 'float', 'default':0},
        # {'name': 'angleOut', 'group': 'ArmProperties', 'type': 'float', 'help': 'angle of rotation (in radians) of '
        #                                                                         'the arm', 'default': 0}                                                                       
    ]

    def __init__(self, *args, **kwargs):
        Sofa.Prefab.__init__(self, *args, **kwargs)
        self.servoarm = None
        self.servomotor = None

    def init(self):
        self.servomotor = self.addChild(ServoMotor(name="ServoMotor", translation=self.translation.value,
                                                   rotation=self.rotation.value))
        # self.servoarm = self.addChild(ServoArm(name="ServoArm"))
        # self.servoarm.setRigidMapping(self.ServoMotor.Articulation.ServoWheel.dofs.getLinkPath())

        # add a public attribute and connect it to the private one.
        self.ServoMotor.angleIn.setParent(self.angleIn)

        # connect the public attribute to the internal one.
        self.angleOut.setParent(self.ServoMotor.angleOut)
def createScene(rootNode):
    import math
    from splib3.animation import animate

    def animation(target, factor):
        target.angleIn.value = math.cos(factor * 2 * math.pi)

    scene = Scene(rootNode, plugins=['SofaConstraint', 'SofaGeneralRigid', 'SofaOpenglVisual', 'SofaRigid',
                                     "ArticulatedSystemPlugin", "Sofa.Component.AnimationLoop",
                                     "Sofa.Component.Constraint.Lagrangian.Correction",
                                     "Sofa.Component.Constraint.Lagrangian.Solver",
                                     "Sofa.Component.Constraint.Projective", "Sofa.Component.IO.Mesh",
                                     "Sofa.Component.LinearSolver.Direct", "Sofa.Component.Mass",
                                     "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.Spring",
                                     "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Visual",
                                     "Sofa.GL.Component.Rendering3D", "Sofa.GUI.Component", ], iterative=False)
    scene.addMainHeader()
    scene.addObject('DefaultVisualManagerLoop')
    scene.addObject('FreeMotionAnimationLoop')
    scene.addObject('GenericConstraintSolver', maxIterations=1e3, tolerance=1e-5)
    scene.Simulation.addObject('GenericConstraintCorrection')

    scene.dt = 0.01
    scene.gravity = [0., -9810., 0.]

    arti_system = scene.Modelling.addChild(ActuatedArm(name="Articulation_system",translation=[0.0, 60.0, 0.0],rotation=[0.0, 0.0, 0.0]))
    animate(animation, {'target': arti_system}, duration=10., mode='loop',factor = 2)
    scene.Modelling.Articulation_system.ServoMotor.Articulation.ServoWheel.dofs.showObject = 0
    scene.Simulation.addChild(scene.Modelling)

    return scene
