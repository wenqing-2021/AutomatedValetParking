'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-06
FilePath: /Automated Valet Parking/util_math/coordinate_transform.py
Description: provide coordinate transformation

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import numpy as np


class coordinate_transform:
    def __init__(self) -> None:
        pass

    @staticmethod
    def twodim_transform(start, end):
        '''
        intro: this function is used to build a new coordinate based on
        the start point.
        input: 
            start point (x,y,theta): list
            end point (x,y,theta): list
        return: 
            rotation_matrix: np.array
            new_end position: list
        '''
        theta = start[2]
        R = np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])
        trans_end_x_y = np.dot(R, np.array(
            [[end[0] - start[0]], [end[1] - start[1]]]))
        new_end = [trans_end_x_y[0, 0], trans_end_x_y[1, 0]]
        new_end.append(end[2] - theta)

        return R, new_end

    @staticmethod
    def inverse_trans(trans_path: list = None,
                      rotation_matrix: np.array = None,
                      start: list = None) -> list:
        '''
        intro: this function is used to inverse transform the points into 
        the original coordinates
        input: 
            path: the interpolated points list
            rotation_matrix: the two dim transform matrix
            start(x,y,theta): the start point of this path
        return:
            inversed_path
        '''

        start_array = np.ones((len(trans_path), 2))
        start_array[:, 0] = start_array[:, 0] * start[0]
        start_array[:, 1] = start_array[:, 1] * start[1]
        trans_path = np.array(trans_path)
        trans_path_position = trans_path[:, :2]
        # compute the x,y position
        inversed_path_position = (rotation_matrix.transpose().dot(trans_path_position.transpose())).transpose() + \
            start_array
        trans_path[:, :2] = inversed_path_position
        # compute theta
        trans_path[:, 2:3] += start[2]
        inversed_path = trans_path.tolist()

        return inversed_path
