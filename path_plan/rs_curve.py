'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-08
FilePath: /Automated Valet Parking/path_planner/rs_curve.py
Description: rs curve for hybrid a star

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import math
import numpy as np
import matplotlib.pyplot as plt


'''
This file is surrported by this repo:https://github.com/zhm-real/CurvesGenerator
'''

# parameters initiation
STEP_SIZE = 0.5
MAX_LENGTH = 1000.0
PI = math.pi


class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.5 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + PI - angle
        theta_hat_R = theta + PI + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


class Car:
    def __init__(self, x, y, yaw, w, L):
        theta_B = PI + yaw

        xB = x + L / 4 * np.cos(theta_B)
        yB = y + L / 4 * np.sin(theta_B)

        theta_BL = theta_B + PI / 2
        theta_BR = theta_B - PI / 2

        x_BL = xB + w / 2 * np.cos(theta_BL)        # Bottom-Left vertex
        y_BL = yB + w / 2 * np.sin(theta_BL)
        x_BR = xB + w / 2 * np.cos(theta_BR)        # Bottom-Right vertex
        y_BR = yB + w / 2 * np.sin(theta_BR)

        x_FL = x_BL + L * np.cos(yaw)               # Front-Left vertex
        y_FL = y_BL + L * np.sin(yaw)
        x_FR = x_BR + L * np.cos(yaw)               # Front-Right vertex
        y_FR = y_BR + L * np.sin(yaw)

        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color='black')

        Arrow(x, y, yaw, L / 2, 'black')
        # plt.axis("equal")
        # plt.show()


# class for PATH element
class PATH:
    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        # lengths of each part of path (+: forward, -: backward) [float]
        self.lengths = lengths
        self.ctypes = ctypes  # type of each part of the path [string]
        self.L = L  # total path length [float]
        self.x = x  # final x positions [m]
        self.y = y  # final y positions [m]
        self.yaw = yaw  # final yaw angles [rad]
        self.directions = directions  # forward: 1, backward:-1


def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw,
                           maxc, step_size=step_size)

    minL = paths[0].L
    mini = 0

    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL, mini = paths[i].L, i

    return paths[mini]


def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]

    paths = generate_path(q0, q1, maxc)

    for path in paths:
        x, y, yaw, directions = \
            generate_local_course(path.L, path.lengths,
                                  path.ctypes, maxc, step_size * maxc)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2])
                  * iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2])
                  * iy + q0[1] for (ix, iy) in zip(x, y)]
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
        path.directions = directions
        path.lengths = [l / maxc for l in path.lengths]
        path.L = path.L / maxc

    return paths


def set_path(paths, lengths, ctypes):
    path = PATH([], [], 0.0, [], [], [], [])
    path.ctypes = ctypes
    path.lengths = lengths

    # check same path exist
    for path_e in paths:
        if path_e.ctypes == path.ctypes:
            if sum([x - y for x, y in zip(path_e.lengths, path.lengths)]) <= 0.01:
                return paths  # not insert path

    path.L = sum([abs(i) for i in lengths])

    if path.L >= MAX_LENGTH:
        return paths

    assert path.L >= 0.01
    paths.append(path)

    return paths


def LSL(x, y, phi):
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if t >= 0.0:
        v = M(phi - t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LSR(x, y, phi):
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2

    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)

        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRL(x, y, phi):
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))

    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = M(t1 + 0.5 * u + PI)
        v = M(phi - t + u)

        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def SCS(x, y, phi, paths):
    flag, t, u, v = SLS(x, y, phi)

    if flag:
        paths = set_path(paths, [t, u, v], ["S", "L", "S"])

    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])

    return paths


def SLS(x, y, phi):
    phi = M(phi)

    if y > 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CSC(x, y, phi, paths):
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "L"])

    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "L"])

    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])

    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])

    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "S", "R"])

    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "S", "R"])

    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "L"])

    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "L"])

    return paths


def CCC(x, y, phi, paths):
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["L", "R", "L"])

    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["L", "R", "L"])

    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "L", "R"])

    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "L", "R"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["L", "R", "L"])

    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["L", "R", "L"])

    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "L", "R"])

    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "L", "R"])

    return paths


def calc_tauOmega(u, v, xi, eta, phi):
    delta = M(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0

    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0

    if t2 < 0:
        tau = M(t1 + PI)
    else:
        tau = M(t1)

    omega = M(tau - u + v - phi)

    return tau, omega


def LRLRn(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))

    if rho <= 1.0:
        u = math.acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRLRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0

    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -0.5 * PI:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCCC(x, y, phi, paths):
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["L", "R", "L", "R"])

    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["R", "L", "R", "L"])

    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["R", "L", "R", "L"])

    return paths


def LRSR(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(-eta, xi)

    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = M(t + 0.5 * PI - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def LRSL(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = M(theta + math.atan2(r, -2.0))
        v = M(phi - 0.5 * PI - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSC(x, y, phi, paths):
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "L"])

    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "R"])

    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["L", "R", "S", "R"])

    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "L", "S", "L"])

    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "L", "S", "L"])

    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "R", "L"])

    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "L", "R"])

    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "R", "L"])

    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["L", "S", "L", "R"])

    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["L", "S", "L", "R"])

    return paths


def LRSLR(x, y, phi):
    # formula 8.11 *** TYPO IN PAPER ***
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(xi, eta)

    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= 0.0:
            t = M(math.atan2((4.0 - u) * xi - 2.0 *
                  eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)

            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0


def CCSCC(x, y, phi, paths):
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(
            paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["L", "R", "S", "L", "R"])

    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["L", "R", "S", "L", "R"])

    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path(
            paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["R", "L", "S", "R", "L"])

    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path(
            paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["R", "L", "S", "R", "L"])

    return paths


def generate_local_course(L, lengths, mode, maxc, step_size):
    point_num = int(L / step_size) + len(lengths) + 3

    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    pd = d
    ll = 0.0

    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = \
                interpolate(ind, pd, m, maxc, ox, oy,
                            oyaw, px, py, pyaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        ind += 1
        px, py, pyaw, directions = \
            interpolate(ind, l, m, maxc, ox, oy, oyaw,
                        px, py, pyaw, directions)

    # remove unused data
    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions


def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):
    if m == "S":
        px[ind] = ox + l / maxc * math.cos(oyaw)
        py[ind] = oy + l / maxc * math.sin(oyaw)
        pyaw[ind] = oyaw
    else:
        ldx = math.sin(l) / maxc
        if m == "L":
            ldy = (1.0 - math.cos(l)) / maxc
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)

        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy
        px[ind] = ox + gdx
        py[ind] = oy + gdy

    if m == "L":
        pyaw[ind] = oyaw + l
    elif m == "R":
        pyaw[ind] = oyaw - l

    if l > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return px, py, pyaw, directions


def generate_path(q0, q1, maxc):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)

    return paths

# utils


def pi_2_pi(theta):
    while theta > PI:
        theta -= 2.0 * PI

    while theta < -PI:
        theta += 2.0 * PI

    return theta


def R(x, y):
    """
    Return the polar coordinates (r, theta) of the point (x, y)
    """
    r = math.hypot(x, y)
    theta = math.atan2(y, x)

    return r, theta


def M(theta):
    """
    Regulate theta to -pi <= theta < pi
    """
    phi = theta % (2.0 * PI)

    if phi < -PI:
        phi += 2.0 * PI
    if phi > PI:
        phi -= 2.0 * PI

    return phi


def get_label(path):
    label = ""

    for m, l in zip(path.ctypes, path.lengths):
        label = label + m
        if l > 0.0:
            label = label + "+"
        else:
            label = label + "-"

    return label


def calc_curvature(x, y, yaw, directions):
    c, ds = [], []

    for i in range(1, len(x) - 1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = math.hypot(dxn, dyn)
        dp = math.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        d = (dn + dp) / 2.0

        if np.isnan(curvature):
            curvature = 0.0

        if directions[i] <= 0.0:
            curvature = -curvature

        if len(c) == 0:
            ds.append(d)
            c.append(curvature)

        ds.append(d)
        c.append(curvature)

    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds


def check_path(sx, sy, syaw, gx, gy, gyaw, maxc):
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc)

    assert len(paths) >= 1

    for path in paths:
        assert abs(path.x[0] - sx) <= 0.01
        assert abs(path.y[0] - sy) <= 0.01
        assert abs(path.yaw[0] - syaw) <= 0.01
        assert abs(path.x[-1] - gx) <= 0.01
        assert abs(path.y[-1] - gy) <= 0.01
        assert abs(path.yaw[-1] - gyaw) <= 0.01

        # course distance check
        d = [math.hypot(dx, dy)
             for dx, dy in zip(np.diff(path.x[0:len(path.x) - 1]),
                               np.diff(path.y[0:len(path.y) - 1]))]

        for i in range(len(d)):
            assert abs(d[i] - STEP_SIZE) <= 0.001


# def main(map:_map):
#     # choose states pairs: (x, y, yaw)

#     # simulation-2
#     start_x = map.case.x0
#     start_y = map.case.y0
#     start_theta = map.case.theta0
#     final_x = map.case.xf
#     final_y = map.case.yf
#     final_theta = map.case.thetaf
#     states = [(start_x, start_y, start_theta), (final_x, final_y, final_theta)]

#     max_c = 0.1  # max curvature
#     path_x, path_y, yaw = [], [], []

#     for i in range(len(states) - 1):
#         s_x = states[i][0]
#         s_y = states[i][1]
#         s_yaw = states[i][2]
#         g_x = states[i + 1][0]
#         g_y = states[i + 1][1]
#         g_yaw = states[i + 1][2]

#         path_i = calc_optimal_path(s_x, s_y, s_yaw,
#                                    g_x, g_y, g_yaw, max_c)

#         path_x += path_i.x
#         path_y += path_i.y
#         yaw += path_i.yaw

    # animation
    # plt.ion()
    # plt.figure(1)

    # for i in range(len(path_x)):
    #     plt.clf()
    #     plt.plot(path_x, path_y, linewidth=1, color='gray')

    #     for x, y, theta in states:
    #         Arrow(x, y, theta, 2, 'blueviolet')

    #     Car(path_x[i], path_y[i], yaw[i], 1.942, 4.689)

    #     for j in range(0, map.case.obs_num):
    #         plt.fill(map.case.obs[j][:, 0], map.case.obs[j][:, 1], facecolor = 'k', alpha = 0.5)

    #     plt.axis("equal")
    #     plt.title("Simulation of Reeds-Shepp Curves")
    #     plt.axis([map.boundary[0], map.boundary[1], map.boundary[2], map.boundary[3]])
    #     plt.draw()
    #     plt.pause(0.001)

    # plt.pause(1)
