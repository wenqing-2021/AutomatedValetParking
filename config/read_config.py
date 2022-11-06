'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-06
FilePath: /Automated Valet Parking/config/read_config.py
Description: read config and return a dict

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import yaml
import os


def read_config(config_name) -> dict:
    name = config_name + '.yaml'
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, name)
    f = open(yamlPath, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config
