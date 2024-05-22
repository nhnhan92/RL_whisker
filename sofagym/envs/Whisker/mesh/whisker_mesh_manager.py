import gmsh
import math as m
import os

class chamber_mesh():
     # Chamber Geometry parameters

    def chamber(no_chamber,chamber_bot_radius,cone_angle,chamber_height,mesh_size):
        chamber_cone_angle = cone_angle*m.pi/180
        chamber_top_radius = chamber_bot_radius - (chamber_height/m.tan(chamber_cone_angle))
        chamber_rel_angle = 360*m.pi/(no_chamber*180)
        chamber_dist = 1.5
        if no_chamber == 2:
            rot_axis = [0, 0, 0, 0, 0, 1]
            init_p = [[0,0,0],[chamber_bot_radius, 0,0],[chamber_top_radius, 0,chamber_height],[0, 0,chamber_height]]
            p1 = gmsh.model.occ.addPoint(init_p[0][0], init_p[0][1], init_p[0][2], mesh_size, -1)
            p2 = gmsh.model.occ.addPoint(init_p[1][0], init_p[1][1], init_p[1][2], mesh_size, -1)
            p3 = gmsh.model.occ.addPoint(init_p[2][0], init_p[2][1], init_p[2][2], mesh_size, -1)
            p4 = gmsh.model.occ.addPoint(init_p[3][0], init_p[3][1], init_p[3][2], mesh_size, -1)

            l1 = gmsh.model.occ.addLine(p1, p2, tag=-1)
            l2 = gmsh.model.occ.addLine(p2, p3, tag=-1)
            l3 = gmsh.model.occ.addLine(p3, p4, tag=-1)
            l4 = gmsh.model.occ.addLine(p4, p1, tag=-1)
            gmsh.model.occ.synchronize()

            curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4], -1)
            chamber_plane = gmsh.model.occ.addPlaneSurface([curve], tag=-1)
            chamber = gmsh.model.occ.revolve([(2, chamber_plane)], rot_axis[0], rot_axis[1], 
                                            rot_axis[2], rot_axis[3], rot_axis[4], rot_axis[5], 2*m.pi)
            p1 = gmsh.model.occ.addPoint(chamber_bot_radius, chamber_dist, 0, mesh_size, -1)
            p2 = gmsh.model.occ.addPoint(-chamber_bot_radius, -chamber_dist, 0, mesh_size, -1)
            p3 = gmsh.model.occ.addPoint(chamber_bot_radius, -chamber_dist, 0, mesh_size, -1)
            p4 = gmsh.model.occ.addPoint(-chamber_bot_radius, chamber_dist, 0, mesh_size, -1)
            
            l1 = gmsh.model.occ.addLine(p1, p4, tag=-1)
            l2 = gmsh.model.occ.addLine(p4, p2, tag=-1)
            l3 = gmsh.model.occ.addLine(p2, p3, tag=-1)
            l4 = gmsh.model.occ.addLine(p3, p1, tag=-1)
            curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4], -1)
            cut_plane = gmsh.model.occ.addPlaneSurface([curve], tag=-1)
            gmsh.model.occ.synchronize()
            cut_body = gmsh.model.occ.extrude([(2,cut_plane)],0,0,chamber_height,recombine=0)
            chamber = gmsh.model.occ.cut([(3,1)], [(3,2)], tag=-1,removeObject=1, removeTool=1)
            gmsh.model.occ.remove([(2, chamber_plane)], recursive=1)
            gmsh.model.occ.synchronize()
            
        else:
            if no_chamber >= 3:
                rot_axis = []
                init_p = []
                for i in range(no_chamber):
                    rot_axis.append([m.cos(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)-chamber_dist*m.sin(chamber_rel_angle*i),
                            m.sin(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)+chamber_dist*m.cos(chamber_rel_angle*i),
                            0, 0, 0, 1])
                    init_p.append([[m.cos(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)-chamber_dist*m.sin(chamber_rel_angle*i),
                            m.sin(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)+chamber_dist*m.cos(chamber_rel_angle*i),
                            0],
                            [m.cos(chamber_rel_angle*i)*chamber_bot_radius-m.sin(chamber_rel_angle*i)*chamber_dist,
                                m.sin(chamber_rel_angle*i)*chamber_bot_radius+m.cos(chamber_rel_angle*i)*chamber_dist,
                                0],
                                [m.cos(chamber_rel_angle*i)*chamber_top_radius-m.sin(chamber_rel_angle*i)*chamber_dist,
                                m.sin(chamber_rel_angle*i)*chamber_top_radius+m.cos(chamber_rel_angle*i)*chamber_dist,
                                chamber_height],
                                [m.cos(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)-chamber_dist*m.sin(chamber_rel_angle*i),
                            m.sin(chamber_rel_angle*i)*chamber_dist/m.tan(chamber_rel_angle/2)+chamber_dist*m.cos(chamber_rel_angle*i),
                            chamber_height]])

            else:
                rot_axis = [[0, 0, 0, 0, 0, 1]]
                init_p = [[[0,0,0],[chamber_bot_radius, 0,0],[chamber_top_radius, 0,chamber_height],[0, 0,chamber_height]]]
            
            for i in range(no_chamber):
                p1 = gmsh.model.occ.addPoint(init_p[i][0][0], init_p[i][0][1], init_p[i][0][2], 5, -1)
                p2 = gmsh.model.occ.addPoint(init_p[i][1][0], init_p[i][1][1], init_p[i][1][2], 5, -1)
                p3 = gmsh.model.occ.addPoint(init_p[i][2][0], init_p[i][2][1], init_p[i][2][2], 5, -1)
                p4 = gmsh.model.occ.addPoint(init_p[i][3][0], init_p[i][3][1], init_p[i][3][2], 5, -1)

                l1 = gmsh.model.occ.addLine(p1, p2, tag=-1)
                l2 = gmsh.model.occ.addLine(p2, p3, tag=-1)
                l3 = gmsh.model.occ.addLine(p3, p4, tag=-1)
                l4 = gmsh.model.occ.addLine(p4, p1, tag=-1)
                gmsh.model.occ.synchronize()

                curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4], -1)
                chamber_plane = gmsh.model.occ.addPlaneSurface([curve], tag=-1)
                chamber = gmsh.model.occ.revolve([(2, chamber_plane)], rot_axis[i][0], rot_axis[i][1], 
                                                rot_axis[i][2], rot_axis[i][3], rot_axis[i][4], rot_axis[i][5], chamber_rel_angle)
                gmsh.model.occ.remove([(2, chamber_plane)], recursive=1)
                gmsh.model.occ.synchronize()
        return chamber
class whisker_body_mesh():

    def main_body(body_bot_radius,
                cone_angle,
                body_height,
                no_chamber,
                chamber_bot_radius,
                chamber_height,
                mesh_size):
        body_cone_angle = cone_angle*m.pi/180
        body_top_radius = body_bot_radius - (body_height/m.tan(body_cone_angle))
        ### Chambers
        chamber = chamber_mesh.chamber(no_chamber = no_chamber,
                            chamber_bot_radius = chamber_bot_radius,
                            cone_angle = cone_angle,
                            chamber_height = chamber_height,
                            mesh_size = mesh_size)
        ## Main body
        # Body construction
        p_1 = gmsh.model.occ.addPoint(0, 0, 0, 5, -1)
        p_2 = gmsh.model.occ.addPoint(0, body_bot_radius, 0, 5, -1)
        p_3 = gmsh.model.occ.addPoint(0, body_top_radius, body_height, 5,-1)
        p_4 = gmsh.model.occ.addPoint(0, 0, body_height, 5, -1)

        gmsh.model.occ.synchronize()

        l_1 = gmsh.model.occ.addLine(p_1, p_2, tag=-1)
        l_2 = gmsh.model.occ.addLine(p_2, p_3, tag=-1)
        l_3 = gmsh.model.occ.addLine(p_3, p_4, tag=-1)
        l_4 = gmsh.model.occ.addLine(p_4, p_1, tag=-1)
        curve = gmsh.model.occ.addCurveLoop([l_1, l_2, l_3, l_4], -1)
        plane = gmsh.model.occ.addPlaneSurface([curve], tag=-1)
        body = gmsh.model.occ.revolve([(2, plane)], 0, 0, 0, 0, 0, 1, m.pi*2,recombine=1)
        gmsh.model.occ.remove([(2, plane)], recursive=1)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.synchronize()           
        # Cut body
        for i in range (no_chamber):
            final_body = gmsh.model.occ.cut([(3,gmsh.model.occ.getMaxTag(3))], [(3,i+1)], tag=-1,removeObject=1, removeTool=1)

def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    for i in range (1):
        if i == 0:
            gmsh.model.add("chamber")
            chamber_mesh.chamber(no_chamber = 2,
                        chamber_bot_radius = 10,
                        cone_angle = 85.5,
                        chamber_height = 24,
                        mesh_size = 4)
        else:
            gmsh.model.add("whisker_body")
            whisker_body_mesh.main_body(body_bot_radius = 12,
                                        cone_angle = 85.5,
                                        body_height = 100,
                                        no_chamber = 2,
                                        chamber_bot_radius = 10,
                                        chamber_height = 24,
                                        mesh_size = 4)
        gmsh.logger.start()
        ### Mesh creation
        # Parameters

        # gmsh.model.occ.mesh.setSize(gmsh.model.occ.getEntities(-1), mesh_size)
        gmsh.model.occ.synchronize()
        if i == 0:
            gmsh.model.mesh.generate(2)
            ### Exporting files
            gmsh.write("test_stl.stl")
        if i == 1:
            gmsh.model.mesh.generate(3)
            ### Exporting files
            gmsh.write("test_vtk.vtk")

        

    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == '__main__':
    main()