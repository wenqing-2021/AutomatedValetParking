'''
Author: wenqing-hnu
Date: 2022-11-02
LastEditors: wenqing-hnu
LastEditTime: 2022-11-04
FilePath: /HybridAstar/animation/record_solution.py
Description: record trajectory into a file

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from typing import List
import pandas as pd
import os


class DataRecorder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def record(save_path: str,
               save_name: str,
               trajectory: List[List]):
        '''
        description: save the traj into a csv file in the solution folder and
                     the sep is '\t'.
        param {str} save_path
        param {str} case_name
        param {List} trajectory: x,y,theta,v,a,sigma,omega,t
        return {*}
        '''

        assert len(trajectory[0]) == 8, 'the trajectory size should be 8'
        trajectory_data = pd.DataFrame(trajectory)
        column_name = ['x',
                       'y',
                       'theta',
                       'v',
                       'a',
                       'sigma',
                       'omega',
                       't']
        trajectory_data.columns = column_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = 'Solution_' + save_name
        file_name = os.path.join(save_path, save_name)
        trajectory_data.to_csv(file_name, index='True', sep='\t')

    @staticmethod
    def save_gif():
        pass
