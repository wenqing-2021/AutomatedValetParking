import yaml
import os


def read_config(config_name) -> dict:
    name = config_name + '.yaml'
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, name)
    f = open(yamlPath, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config
