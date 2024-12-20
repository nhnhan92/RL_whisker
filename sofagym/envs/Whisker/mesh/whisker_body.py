import gmsh
import math as m
import os
from .whisker_mesh_manager import whisker_body_mesh

# Element size for creating mechanical model
mesh_size = 4

def mesh_generator(body_bot_radius,
                    cone_angle,
                    body_height,
                    no_chamber,
                    chamber_bot_radius,
                    chamber_height,
                    mesh_size,
                    index):
    ### Chambers
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("whisker_body")
    gmsh.logger.start()
    whisker_body_mesh.main_body(body_bot_radius = body_bot_radius,
                        cone_angle = cone_angle,
                        body_height = body_height,
                        no_chamber = no_chamber,
                        chamber_bot_radius = chamber_bot_radius,
                        chamber_height = chamber_height,
                        mesh_size = mesh_size)

    ### Mesh creation
    # Parameters

    gmsh.model.occ.mesh.setSize(gmsh.model.occ.getEntities(-1), mesh_size)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)

    ### Exporting files
    gmsh.write(f"/home/nhnhan/Desktop/sofa/SofaGym/sofagym/envs/Whisker/mesh/body{index}_vtk.vtk")

    # gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':
    height = [100,100,100,80,80,80,60,60,60]
    no_chamber = [1,2,3,1,2,3,1,2,3]
    for i in range(9):
        mesh_generator(body_bot_radius = 12,
                        cone_angle = 85.5,
                        body_height = height[i],
                        no_chamber = 2,
                        chamber_bot_radius = 10,
                        chamber_height = 24,
                        mesh_size = 4,
                        index = i)