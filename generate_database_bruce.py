import torch
from lib.networks.DescNet import get_desc_net
from lib.networks.SonarNet import get_desc_net as get_sonar_net
from lib.utils.net_utils import load_model
from lib.datasets.bruce_loader import make_data_loader
from lib.evaluators.matrix_evaluator_bruce import Evaluator
import os
import argparse
import lib.utils.file_utils as file_utils
import lib.utils.image_utils as image_utils
import tqdm 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import lib.utils.sonar_utils as sonar_utils

def parse_args():
    parser = argparse.ArgumentParser(description='DescNet database generation tool')
    parser.add_argument('-d', '--dataset_dir', 
                        help='Input directory containing the training dataset', 
                        required=True) 
    parser.add_argument('-m', '--model_dir',  
                        help='Input directory where the trained model is stored',
                        required=True)
    parser.add_argument('-o', '--output_fn',  
                        help='Output file in which to store the generated datbase',
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
    sonar_utils.set_sonar_parameters(data_cfg['sonar_yaw_step'], data_cfg['sonar_rho_step'])

    if network_cfg['use_resnet']:
        get_net = get_desc_net
    else:
        get_net = get_sonar_net

    record_dir = os.path.join(args.model_dir, 'record')
    eval_stats_fn, best_model_dir = file_utils.make_directories(args.model_dir, True)
    
    print('Resize ratio:', data_cfg['resize_ratio'])
    network = get_net(network_cfg['use_vlad'], train_cfg['use_arcface'], network_cfg['descriptor_size'], network_cfg['n_clusters'])
    begin_epoch = load_model(network, None, None, None, best_model_dir, True)
    network = network.cuda()
    network.eval()

    evaluator = Evaluator(False, args.dataset_dir, 'test', data_cfg, eval_cfg, train_cfg['batch_size'], database_fn=args.output_fn, graph=True)
    evaluator.save_database(network)