from util_math.coordinate_transform import coordinate_transform
import numpy as np
import math
from scipy.linalg import solve


class spine:
    def __init__(self) -> None:
        pass

    @staticmethod
    def cubic_spline(start, end):
        '''
        :param start: start node
        :param end: end node
        :return: target cubic function,include [a,b,c,d]
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
            k = 3*a*x**2 + 2*b*x + c
            theta_angle = math.atan(k)

            return y, theta_angle

        return cubic_func, rotation_matrix, new_end
