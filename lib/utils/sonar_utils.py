import numpy as np


__SONAR_YAW_STEP__ = None
__SONAR_RHO_STEP__ = None

def set_sonar_parameters(yaw_step, rho_step):
  global __SONAR_YAW_STEP__
  global __SONAR_RHO_STEP__
  __SONAR_YAW_STEP__ = yaw_step
  __SONAR_RHO_STEP__ = rho_step

def area_sector(r1, r2, width):
    pre = (width/360) * np.pi
    a1 = pre *r1*r1
    a2 = pre *r2*r2
    return a2-a1

def angle_range(angle):
    angle = angle % 360
    if angle > 180:
        angle = angle - 360
    return angle

def to_rad(deg):
    return deg*np.pi/180


def angle_diff(a1, a2):
    return (a1 - a2 + 180) % 360 - 180


def circle_intersections(pos1, pos2, radius):
    dist = np.linalg.norm((pos1-pos2)[:2])
    summ = (pos1+pos2)[:2]
    diff = (pos1-pos2)[:2]
    diff = diff[[1,0]]
    diff[0] *= -1
    p1 = 0.5*summ
    p2 = 0.5*np.sqrt(4*radius*radius/(dist*dist) -1)*diff
    return p1+p2,p1-p2

def on_segment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False

def points_orientation(p,q,r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    else:
        # Collinear orientation
        return 0

def line_intersect(s1,e1,s2,e2):
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = points_orientation(s1, e1, s2)
    o2 = points_orientation(s1, e1, e2)
    o3 = points_orientation(s2, e2, s1)
    o4 = points_orientation(s2, e2, e1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and on_segment(s1, s2, e1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and on_segment(s1, e2, e1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and on_segment(s2, s1, e2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and on_segment(s2, e1, e2)):
        return True
  
    # If none of the cases
    return False

def inside_sector(x,y,pos,radius, yaw,awidth):
    nx = x-pos[0]
    ny = y-pos[1] 
    rhosq = nx * nx + ny * ny
    if rhosq > (radius*radius)+1e-5:
        return False
    theta = (180*np.arctan2(ny,nx))/np.pi
    theta_diff = angle_diff(angle_range(yaw), theta)
    return theta_diff >= (-awidth/2) and theta_diff <= (awidth/2)

def line_sector_intersection(sl, dl, sc, rc, yaw, awidth):
    ds = (sl-sc)[:2]
    dl_norm = dl[0]*dl[0]+dl[1]*dl[1]
    ds_norm = ds[0]*ds[0]+ds[1]*ds[1]
    delta = (np.dot(dl,ds))**2 - dl_norm*(ds_norm-rc*rc)
    if delta < -1e-5:
        return False
    elif np.abs(delta)<1e-5:
        t = np.dot(-dl,ds)/dl_norm
        intersection = sl[:2] + t*dl
        return inside_sector(intersection[0], intersection[1], sc, rc, yaw, awidth)
    else:
        t1 = (np.dot(-dl,ds) + np.sqrt(delta)) /dl_norm
        intersection1 = sl[:2] + t1*dl
        return inside_sector(intersection1[0], intersection1[1], sc, rc, yaw, awidth)



def check_sectors_collisions(pos1, pos2, yaw1, yaw2, radius, awidth):
    sa1 = to_rad(yaw1-awidth/2)
    ea1 = to_rad(yaw1+awidth/2)
    sa2 = to_rad(yaw2-awidth/2)
    ea2 = to_rad(yaw2+awidth/2)

    d1 = np.array([np.cos(sa1),np.sin(sa1)])
    d2 = np.array([np.cos(ea1),np.sin(ea1)])
    d3 = np.array([np.cos(sa2),np.sin(sa2)])
    d4 = np.array([np.cos(ea2),np.sin(ea2)])

    eps1 = pos1[:2] + radius* d1
    eps2 = pos1[:2] + radius* d2
    eps3 = pos2[:2] + radius* d3
    eps4 = pos2[:2] + radius* d4

    if line_intersect(pos1, eps1, pos2, eps3):
        return True
    if line_intersect(pos1, eps1, pos2, eps4):
        return True
    if line_intersect(pos1, eps2, pos2, eps3):
        return True
    if line_intersect(pos1, eps2, pos2, eps4):
        return True

    cints = circle_intersections(pos1, pos2, radius)
    if inside_sector(cints[0][0],cints[0][1], pos1,radius,yaw1,awidth) and inside_sector(cints[0][0],cints[0][1], pos2,radius,yaw2,awidth):
        return True
    if inside_sector(cints[1][0],cints[1][1], pos1,radius,yaw1,awidth) and inside_sector(cints[1][0],cints[1][1], pos2,radius,yaw2,awidth):
        return True
    

    if line_sector_intersection(pos1, d1, pos2, radius, yaw2, awidth):
        return True
    if line_sector_intersection(pos1, d2, pos2, radius, yaw2, awidth):
        return True
        
    if line_sector_intersection(pos2, d3, pos1, radius, yaw1, awidth):
        return True
    if line_sector_intersection(pos2, d4, pos1, radius, yaw1, awidth):
        return True

    return False


def do_intersect(pos1, pos2, yaw1, yaw2, radius, awidth, max_angle_diff):
    
    if np.abs(angle_diff(yaw1, yaw2)) > max_angle_diff:
        return False
    
    dist = np.linalg.norm(pos1 - pos2)
    # too far apart to intersect in anything more than just a point
    
    if dist >= 2*radius:
        return False
    return check_sectors_collisions(pos1, pos2, yaw1, yaw2, radius, awidth)

def explicit_integral(pos1, pos2, yaw1, yaw2, radius, awidth):
    yaw1 = angle_range(yaw1)
    yaw2 = angle_range(yaw2)

    rsize = 1
    tsize = 3

    dist_2_1 = np.linalg.norm(pos2-pos1) + 1e-5
    dir_2_1 = (pos2-pos1)/dist_2_1
    p1f = np.array([np.cos(yaw1*np.pi/180), np.sin(yaw1*np.pi/180)])
    cosd1f = np.dot(dir_2_1[:2], p1f)
    if cosd1f > 0:
        rho_min = max(0, dist_2_1*cosd1f)
        pos_inner = pos2
        yaw_inner = yaw2
        pos_outer = pos1
        yaw_outer = yaw1
    else:
        rho_min = max(0, -1*dist_2_1*cosd1f)
        pos_inner = pos1
        yaw_inner = yaw1
        pos_outer = pos2
        yaw_outer = yaw2

    rstart = int(rho_min/rsize)
    n_steps_rho = int(radius/rsize)
    rend = n_steps_rho + 1
    n_steps_theta = int(awidth/tsize)

    # sectors starting in the same point (50 cm diff)
    if np.linalg.norm(pos1-pos2) < 0.5:
        # we need to find the yaw overlap
        yaw_diff = np.abs(angle_diff(yaw1, yaw2))
        # if the difference is more than 120 degrees than they do not overlap at all
        # if it is 60 degrees the overlap is half the area, if 0 degrees it is the whole area
        if yaw_diff > 120:
            return 0
        else:
            yaw_overlap = 120 - yaw_diff
            return np.pi*radius*radius*yaw_overlap/360


    # collinear sectors
    if np.abs(angle_diff(yaw1, yaw2)) <= 5:
        area = np.pi*(radius-dist_2_1)*(radius-dist_2_1)*awidth/360
        r2 = radius-dist_2_1
        rho_min = r2
        rstart = int(rho_min/rsize)+1
        pos_outer, pos_inner = (pos_inner,pos_outer)
        yaw_inner, yaw_outer = (yaw_outer, yaw_inner)
        rho_max = (radius+r2)/2
        rend = int(rho_max/rsize)+1
    else:
        area = 0

    for rstep in range(rstart,n_steps_rho+1):
        rho = rstep * rsize
        carea = area_sector(rho-rsize, rho, tsize)
        for tstep in range(0,n_steps_theta):
            theta = yaw_outer - awidth/2 + tsize * tstep
            theta_rad = (theta*np.pi)/180
            if inside_sector(pos_outer[0] + rho*np.cos(theta_rad), pos_outer[1] + rho*np.sin(theta_rad), pos_inner, radius, yaw_inner, awidth):
                area = area + carea
    return area

def sonar_overlap_score(pos1, pos2, yaw1, yaw2, radius, awidth, max_angle_diff=360):
    # too far apart to intersect in anything more than just a point
    if not do_intersect(pos1, pos2, yaw1, yaw2, radius, awidth, max_angle_diff) :
        return 0
    int_area = explicit_integral(pos1, pos2, yaw1, yaw2, radius, awidth)
    area_tot = np.pi*radius*radius*awidth/360
    return int_area / area_tot
