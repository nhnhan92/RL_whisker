import numpy as np
import matplotlib.pyplot as plt
import csv


def generate_inverted_truncated_conical_double_helix(H=25.0,       # "design" cone height
                                                    R=11.0,       # big-end radius at z=0
                                                    alpha=85.5,   # cone half-angle in degrees
                                                    pitch=4.0,    # vertical distance per 2π turn
                                                    num_points=1000,    
                                                    no_chamber = 2,
                                                    offset = 1,
                                                    plot = False,
                                                    save_data = False):
    """
    Generate two strands of a double helix on an inverted truncated cone:
      - At z=0, radius=R (the big end).
      - Radius shrinks linearly: r(z)=R - z*tan(alpha_rad).
      - If H < H, we truncate the vertical extent at z=H.
      - pitch sets the vertical distance per 2π revolution of the helix.
    """

    alpha_rad = np.radians(alpha)

    # z from 0..H
    z_vals = np.linspace(0, H, num_points)
    # radius from R at z=0 down to R - H*tan(alpha) at z=H
    r_vals = R - z_vals * np.tan(alpha_rad)

    # total number of turns from z=0..H
    total_turns = (H / pitch) if pitch != 0 else 0.0

    # param t in [0..1], angle in [0..2π*total_turns]
    if H > 0:
        t = z_vals / H
    else:
        t = np.zeros_like(z_vals)
    theta = 2.0 * np.pi * total_turns * t

    # Strand 1
    x1 = r_vals * np.cos(theta)
    y1 = r_vals * np.sin(theta)
    fiber1 = np.column_stack((x1, y1, z_vals))

    # Strand 2, offset by π
    x2 = r_vals * np.cos(-theta + np.pi)
    y2 = r_vals * np.sin(-theta + np.pi)
    fiber2 = np.column_stack((x2, y2, z_vals))

    if no_chamber > 1:
        # 2) Cut each fiber by the plane y=offset + insert intersection points
        fiber1_cut = cut_fiber_and_insert_intersections(fiber1, offset)
        fiber2_cut = cut_fiber_and_insert_intersections(fiber2, offset)

        # 3) Further subdivide line segments that lie entirely in the plane y=offset
        #    so we have multiple points instead of just a single straight line.
        #    e.g. n_subdiv=5 => 4 additional points along any in-plane segment.
        fiber1 = subdivide_in_plane_segments(fiber1_cut, offset, n_subdiv=5)
        fiber2 = subdivide_in_plane_segments(fiber2_cut, offset, n_subdiv=5)
        if plot:
            # 4) Plot final geometry
            plot_3d_curves(
                fiber1, fiber2,
                title=f"Double Helix, cut y>={offset}, with plane segments subdivided"
            )
        if save_data:
            # 5) Save final data (optional)
            save_helix_points(fiber1, "fiber1.csv")
            save_helix_points(fiber2, "fiber2.csv")

    return fiber1, fiber2

def cut_fiber_and_insert_intersections(fiber, offset):
    """
    Cut 'fiber' by the plane y=offset and insert intersection points 
    so that the resulting fiber is continuous and includes boundary points.
    
    We keep:
      - If offset >= 0, keep y >= offset.
      - If offset < 0,  keep y <= offset.
      
    For each segment p1->p2 in the original fiber:
      1) If p1 is inside the cut region, add p1 to new_fiber.
      2) If the segment crosses the plane (one side in, other out), 
         compute the intersection point, add it if it lies in the keep side.
      3) If p2 is inside, add p2.
      
    This ensures that if the helix crosses y=offset, we get 
    an intersection node in the final cut geometry.
    """
    new_fiber = []
    n = len(fiber)
    if n < 2:
        return fiber  # nothing to cut or interpolate

    # Decide which side is "inside" based on offset's sign
    # offset >= 0 => keep y >= offset
    # offset < 0  => keep y <= offset
    def inside(y_val):
        return (y_val >= offset) if (offset >= 0) else (y_val <= offset)

    for i in range(n - 1):
        p1 = fiber[i]
        p2 = fiber[i + 1]
        y1, y2 = p1[1], p2[1]

        in1 = inside(y1)
        in2 = inside(y2)

        # If p1 is inside, we add it to new_fiber
        if in1:
            new_fiber.append(p1)

        # Check segment crossing: (y1 - offset)*(y2 - offset) < 0
        # => p1 and p2 lie on opposite sides of plane
        side1 = y1 - offset
        side2 = y2 - offset
        if side1 * side2 < 0.0:
            # We have a crossing -> find intersection by linear interpolation
            alpha_t = (offset - y1) / (y2 - y1)  # fraction along p1->p2
            x_int = p1[0] + alpha_t*(p2[0] - p1[0])
            z_int = p1[2] + alpha_t*(p2[2] - p1[2])
            # y_int = offset
            intersection = np.array([x_int, offset, z_int])
            
            # Now, figure out if we keep that intersection
            # If p2 is inside, that means we are crossing from out->in
            # => intersection is in. Similarly, if p1 is inside, we cross from in->out
            # => intersection might be the last point in new_fiber for that segment.
            # In typical "cut" logic, we always include the boundary intersection 
            # if p1 or p2 is inside:
            new_fiber.append(intersection)

        # If p2 is inside, add it
        if in2:
            new_fiber.append(p2)

    # Handle the very last point if not processed
    # The loop uses i in [0..n-2], so the last point is p2 at i=n-2
    # but we might not have added fiber[-1] if it wasn't inside or 
    # if the second to last segment didn't keep it. 
    # Let's do a simpler approach: check if last point is inside 
    # and not already appended
    last_pt = fiber[-1]
    if inside(last_pt[1]):
        if len(new_fiber) == 0 or not np.allclose(new_fiber[-1], last_pt):
            new_fiber.append(last_pt)

    return np.array(new_fiber)

def plot_3d_curves(fiber1, fiber2, title="Cut + Intersection Helix"):
    """
    Plot two 3D curves on a single figure.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(fiber1[:,0], fiber1[:,1], fiber1[:,2], marker = 'o', label='Fiber 1')
    ax.plot(fiber2[:,0], fiber2[:,1], fiber2[:,2], marker = '*', label='Fiber 2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.show()

def save_helix_points(fiber, filename):
    """
    Save (x, y, z) of a single helix fiber to CSV.
    """
    header = ['Index', 'X', 'Y', 'Z']
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (x, y, z) in enumerate(fiber):
            writer.writerow([i, x, y, z])

def cut_by_y_plane_offset(fiber, offset):
    """
    Keep only points on one side of y=offset:
      - If offset > 0 => keep y > offset
      - If offset < 0  => keep y <= offset
    """
    if offset >= 0:
        mask = (fiber[:,1] >= offset)
    else:
        mask = (fiber[:,1] <= offset)
    return fiber[mask]

def subdivide_in_plane_segments(fiber, offset, n_subdiv=5):
    """
    After cutting, we might find consecutive points (p1->p2) 
    both on the plane y=offset. If so, we subdivide that segment 
    into 'n_subdiv' pieces (including endpoints), inserting 
    intermediate points along the line from p1->p2.
    """
    new_fiber = []
    n = len(fiber)
    if n < 2:
        return fiber

    for i in range(n - 1):
        p1 = fiber[i]
        p2 = fiber[i+1]
        new_fiber.append(p1)  # always keep p1

        # Check if p1, p2 both exactly on the plane
        # i.e. y1==offset and y2==offset
        # (use a small tolerance if floating precision is a concern)
        eps = 1e-12
        if abs(p1[1] - offset) < eps and abs(p2[1] - offset) < eps:
            # subdivide in-plane segment
            for k in range(1, n_subdiv):
                alpha = k / n_subdiv
                x_int = p1[0] + alpha*(p2[0] - p1[0])
                y_int = offset  # exactly in plane
                z_int = p1[2] + alpha*(p2[2] - p1[2])
                new_fiber.append([x_int, y_int, z_int])

    new_fiber.append(fiber[-1])
    return np.array(new_fiber)

def main():
    # Example usage
    H = 30.0
    R = 11.0
    alpha = 90 - 85.5  # i.e. 4.5 deg
    pitch = 4
    offset = -2.0  # y=2 => keep y >= 2

    # 1) Generate the original double helix
    fiber1, fiber2 = generate_inverted_truncated_conical_double_helix(
        H=H, 
        R=R, 
        alpha=alpha, 
        pitch=pitch, 
        num_points=200,
        no_chamber=2,
        offset= 1,
        plot=True,
        save_data=False
    )
    print(fiber1.shape)

if __name__ == "__main__":
    main()
