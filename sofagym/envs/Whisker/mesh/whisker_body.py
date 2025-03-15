import gmsh
import math as m
import os
from whisker_mesh_manager import whisker_body_mesh

# Element size for creating mechanical model
mesh_size = 4

def mesh_generator(body_bot_radius,
                    cone_angle,
                    body_height,
                    no_chamber,
                    chamber_bot_radius,
                    chamber_height,
                    mesh_size
                    ):
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
    thickness = float(body_bot_radius - chamber_bot_radius)
    ### Exporting files
    gmsh.write(f"/home/nhnhan/Desktop/sofa/SofaGym/sofagym/envs/Whisker/mesh/mesh_body/body_{no_chamber}chamber_{body_height}_{chamber_height}_{thickness}.vtk")

    # gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':

    import numpy as np
    height = np.linspace(60,100,5,dtype=int)
    chamber_height = np.linspace(20,40,11,dtype=int)
    chamber_bot_radius = np.linspace(8,10,5)  #~thickness = np.linspace(2,4,5)
    no_chamber = [1,2]
    for i in height:
        for l in no_chamber:
            for j in chamber_height:
                for k in chamber_bot_radius:
                    mesh_generator(body_bot_radius = 12,
                                    cone_angle = 85.5,
                                    body_height = i,
                                    no_chamber = l,
                                    chamber_bot_radius = k,
                                    chamber_height = j,
                                    mesh_size = 4)

    # mesh_generator(body_bot_radius = 12,
    #                 cone_angle = 85.5,
    #                 body_height = 100,
    #                 no_chamber = 2,
    #                 chamber_bot_radius = 10,
    #                 chamber_height = 20,
    #                 mesh_size = 4)