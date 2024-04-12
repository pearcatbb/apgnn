import argparse
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict


################################################################################
# Main #
################################################################################
from torch import nn

from models.handler import ModelHandler
import warnings


warnings.filterwarnings("ignore")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['seed'])
    model = ModelHandler(config)
    model.train()



################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', default='./config/tfinance.yml', type=str, help='path to the config file')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")



################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':

    cfg = get_args()
    config = get_config(cfg['config'])

    main(config)
    # process_data()


# https://pypi.tuna.tsinghua.edu.cn/simple/
