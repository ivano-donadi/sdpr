import torch
from lib.networks.DescNet import get_desc_net
from lib.utils.net_utils import load_model
from lib.datasets.triplet_loader import make_data_loader
from lib.evaluators.particle_evaluator import ParticleEvaluator
import os
import argparse
import lib.utils.file_utils as file_utils
import lib.utils.image_utils as image_utils
import tqdm 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import lib.utils.visualize_utils as visualize_utils
import lib.utils.sonar_utils as sonar_utils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Descriptor database pose visualization tool')
    parser.add_argument('-d', '--database', 
                        help='input descriptor database file', 
                        required=True)
    parser.add_argument('--cfg_file', help='Configuration file', required=True)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg_file, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(exc)

    network_cfg = cfg['network']
    train_cfg = cfg['train']
    eval_cfg = cfg['eval']
    data_cfg = cfg['data']

    image_utils.set_detector(data_cfg["cfar_algorithm"])
    image_utils.set_train_img_width(data_cfg['train_width'])
    image_utils.set_test_img_size(data_cfg['test_width'],data_cfg['test_height'])
    visualize_utils.set_sonar_parameters(data_cfg['sonar_range'], data_cfg['sonar_width'], data_cfg['max_sonar_angle'], data_cfg['min_sonar_similarity'])
    sonar_utils.set_sonar_parameters(data_cfg['sonar_yaw_step'], data_cfg['sonar_rho_step'])
    
    evaluator = ParticleEvaluator(True,args.database, data_cfg, eval_cfg['batch_size'], eval_cfg['scan_stride'],-1)
    evaluator.build_dataset(None)
    position_descs = evaluator.position_descs

    for pd in position_descs:
        pos = pd.position
        x,y = (pos[0],pos[1])
        for yaw,_ in pd.yaw_descriptors:
            yaw = yaw*np.pi/180
            dx = np.cos(yaw)
            dy = np.sin(yaw)
            plt.arrow(x,y,dx,dy, linewidth=0.25, head_width=0.05)
    plt.show()

    