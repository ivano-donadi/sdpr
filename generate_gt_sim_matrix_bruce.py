import os
import argparse
import yaml
import lib.utils.image_utils as image_utils
import numpy as np
from lib.datasets.bruce_loader import make_data_loader
import lib.utils.visualize_utils as visualize_utils
from lib.evaluators.matrix_evaluator_bruce import Evaluator
import lib.utils.sonar_utils as sonar_utils

def parse_args():
    parser = argparse.ArgumentParser(description='Ground truth similarity matrix generation tool for path datasets.')
    parser.add_argument('-d', '--database', 
                        help='Input descriptor database computed on the path dataset. Only the position and orientation associated with each descriptor will be used so it can be created with an untrained network.', 
                        required=True)    
    parser.add_argument('-t', '--test_dir', 
                        help='Input directory containing the same dataset used for generating the descriptors database.', 
                        required=True)
    parser.add_argument('-o', '--output', 
                        help='Output similarity matrix file (.npy extension)', 
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
    data_cfg = cfg['data']
    eval_cfg = cfg['eval']

    image_utils.set_detector(data_cfg["cfar_algorithm"])
    image_utils.set_train_img_width(data_cfg['train_width'])
    image_utils.set_test_img_size(data_cfg['test_width'],data_cfg['test_height'])
    visualize_utils.set_sonar_parameters(data_cfg['sonar_range'], data_cfg['sonar_width'], data_cfg['max_sonar_angle'], data_cfg['min_sonar_similarity'])
    sonar_utils.set_sonar_parameters(data_cfg['sonar_yaw_step'], data_cfg['sonar_rho_step'])

    test_loader = make_data_loader(args.test_dir, 1, 'full', data_cfg)
    evaluator = Evaluator(True, args.database, 'test', data_cfg, eval_cfg, train_cfg['batch_size'])
    evaluator.construct_dataset(None)
    GTMAP = evaluator.generate_gt_similarity_matrix(None, test_loader)

    np.save(args.output,GTMAP)
