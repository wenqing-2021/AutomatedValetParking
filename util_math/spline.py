'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-10-31
FilePath: /Automated Valet Parking/util_math/spline.py
Description: generate the spine function and compute its length

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''



from util_math.coordinate_transform import coordinate_transform
import numpy as np
import math
from scipy.linalg import solve
from scipy import integrate


class spine:
    def __init__(self) -> None:
        pass

    @staticmethod
    def cubic_spline(start, end):
        '''
        description: note that after rotation tranformation, the new_end point is only in the quadrant 1 or 4
        param {*} start
        param {*} end
        return {*} target cubic function,include [a,b,c,d]
        '''
        rotation_matrix, new_end = coordinate_transform.twodim_transform(
            start=start, end=end)
        x0, y0, theta0 = 0, 0, 0
        x1, y1, theta1 = new_end[0], new_end[1], new_end[2]
        A = np.array([
            [x0**3, x0**2, x0, 1],
            [x1**3, x1**2, x1, 1],
            [3*x0**2, 2*x0, 1, 0],
            [3*x1**2, 2*x1, 1, 0]
        ])
        b = np.array([y0, y1, math.tan(theta0), math.tan(theta1)])
        result = solve(A, b)

        def cubic_func(x):
            a = result[0]
            b = result[1]
            c = result[2]
            d = result[3]
            y = a*x**3+b*x**2+c*x+d
            y_pie = 3*a*x**2 + 2*b*x + c
            slope_angle = math.atan(y_pie)  # [-pi/2, pi/2]

            return y, y_pie, slope_angle

        return cubic_func, rotation_matrix, new_end

    @staticmethod
    def Simpson_integral(cubic_func,
                         start_point: list,
                         end_point: list) -> np.float64:
        '''
        description: use simpson_integral to get the arc length of the optimized path
        param: {function}: cubic function
        param: {list} : the start point and the end point
        return {*} the arc lenth
        '''
        y = []
        x_points = np.linspace(
            start=start_point[0], stop=end_point[0], num=100)
        for x in x_points:
            _, y_pie, _ = cubic_func(x)
            y_i = np.sqrt(1 + (y_pie)**2)
            y.append(y_i)

        y = np.array(y)
        arc_length = integrate.simpson(y=y, x=x_points)

        return abs(arc_length)
