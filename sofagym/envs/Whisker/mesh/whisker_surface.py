import gmsh
import math as m
import os


def surface_mesh_generator(body_bot_radius,
            cone_angle,
            body_height,
            mesh_size):
    body_cone_angle = cone_angle*m.pi/180
    body_top_radius = body_bot_radius - (body_height/m.tan(body_cone_angle))
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    gmsh.model.add("body_stl")
    gmsh.logger.start()
    ## Main body
    # Body construction
    p_1 = gmsh.model.occ.addPoint(0, body_bot_radius, 0, mesh_size, -1)
    p_2 = gmsh.model.occ.addPoint(0, body_top_radius, body_height, mesh_size,-1)

    gmsh.model.occ.synchronize()

    l_1 = gmsh.model.occ.addLine(p_1, p_2, tag=-1)

    body = gmsh.model.occ.revolve([(1,l_1)], 0, 0, 0, 0, 0, 1, m.pi*2,recombine=1)
    gmsh.model.occ.remove([(1, l_1)], recursive=1)
    gmsh.model.occ.mesh.setSize(gmsh.model.occ.getEntities(-1), mesh_size)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("body_stl.stl")
    # gmsh.fltk.run()
    gmsh.finalize()
def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    surface_mesh_generator(body_bot_radius = 12,
                                    cone_angle = 85.5,
                                    body_height = 100,
                                    mesh_size = 4)

    gmsh.logger.start()
    ### Mesh creation
    # Parameters

    
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    ### Exporting files
    gmsh.write("test_stl.stl")

    # gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':
    main()