from core import Robot
# from .whisker_body import whisker_body
from legs import Leg, CalibrationLeg
from disk_with_legs import DiskWithLegs, DiskWithLegsOpenLoop
from one_mesh_disk_with_legs import OneMeshDiskWithLegs

_robot_classes = list(locals().items())
a = {}
# print(_robot_classes)
for k, v in _robot_classes:
    # print(k)
    # print(v)
    try:
        if issubclass(v, Robot):
            a[k] = v
    except TypeError:
        pass

def get_robot_class(id):
    if id not in a:
        raise ValueError("Unknown robot class: %s" % id)
    else:
        return a[id]
