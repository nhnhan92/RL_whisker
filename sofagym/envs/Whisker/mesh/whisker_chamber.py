import gmsh
import math as m
import os
from whisker_mesh_manager import chamber_mesh

# Element size for creating mechanical model
mesh_size = 4


def mesh_generator(no_chamber = 2,
                    chamber_bot_radius = 10,
                    cone_angle = 85.5,
                    chamber_height = 24,
                    mesh_size = mesh_size,
                    cavity_idx = None):
    

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("whisker_chamber")
    gmsh.logger.start()
    ### Chambers
    chamber = chamber_mesh.chamber(no_chamber = no_chamber,
                        chamber_bot_radius = chamber_bot_radius,
                        cone_angle = cone_angle,
                        chamber_height = chamber_height,
                        mesh_size = mesh_size)
    # gmsh.model.occ.remove([(3, i+2)], recursive=1)
    ### Mesh creation
    # Parameters
    gmsh.model.occ.mesh.setSize(gmsh.model.occ.getEntities(-1), mesh_size)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)

    if no_chamber > 1:
        surfaces_removed = {1: [2,3],  #idx of this dict is the idx of cavity remained from removing
                        2: [1,4]}

        for i in surfaces_removed[cavity_idx]:
            gmsh.model.removeEntities([(2, i)], recursive=True)
    thickness = float(12 - chamber_bot_radius)
    ### Exporting files
    gmsh.write(f"/home/nhnhan/Desktop/sofa/SofaGym/sofagym/envs/Whisker/mesh/mesh_chamber/{no_chamber}chamber_{chamber_height}_{thickness}_idx{cavity_idx}.stl")
    # gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':
    import numpy as np
    chamber_height = np.linspace(20,40,11,dtype=int)
    chamber_bot_radius = np.linspace(8,10,5)  #~thickness = np.linspace(2,4,5)
    no_chamber = [1,2]
    cavity_idx = [1,2]
    for i in no_chamber:
        for j in chamber_height:
            for k in chamber_bot_radius:
                for l in range(i):
                    mesh_generator(no_chamber = i,
                                    chamber_bot_radius = k,
                                    cone_angle = 85.5,
                                    chamber_height = j,
                                    mesh_size = 4,
                                    cavity_idx = l+1)
        
    # mesh_generator(no_chamber = 1,
    #                 chamber_bot_radius = 10,
    #                 cone_angle = 85.5,
    #                 chamber_height = 24,
    #                 mesh_size = 4,
    #                 cavity_idx = 1)