import torch
from lib.networks.DescNet import get_desc_net
from lib.networks.SonarNet import get_desc_net as get_sonar_net
from lib.utils.net_utils import load_model
from lib.datasets.bruce_loader import make_data_loader
from lib.evaluators.matrix_evaluator_bruce import Evaluator
import os
import argparse
import lib.utils.image_utils as image_utils
import lib.utils.file_utils as file_utils
import tqdm 
import yaml
import lib.utils.visualize_utils as visualize_utils
import numpy as np
import matplotlib.pyplot as plt
import lib.utils.sonar_utils as sonar_utils

def get_lost_indices():
    banned_indices = []
    banned_indices.append(3)
    banned_indices.extend(range(38, 52))
    banned_indices.extend(range(62,64))
    banned_indices.extend(range(65,72))
    banned_indices.extend(range(74,93))
    banned_indices.extend(range(106,107))
    banned_indices.extend(range(113,129))
    banned_indices.extend(range(130,132))
    banned_indices.extend(range(133,136))
    banned_indices.extend(range(148,174))
    banned_indices.extend(range(176,177))
    banned_indices.extend(range(179,185))
    return banned_indices


def parse_args():
    parser = argparse.ArgumentParser(description='DescNet eval tool for path datasets')
    parser.add_argument('--from_database', action='store_true',
                        help='If this option is specified the database of descriptor is loaded from file, otherwise it is computed online.')
    parser.add_argument('-d', '--database', 
                        help='if from_database is true this is the input descriptor database file, otherwise it is the folder containing the path dataset.', 
                        required=True)    
    parser.add_argument('-t', '--test_dataset_dir', 
                        help='Input directory containing the test set (same as the one used to compute the descriptors database)', 
                        required=True)    
    parser.add_argument('-g', '--gt_sim', 
                        help='Input ground truth similarity file. If it does not exist it will be computed and saved at the given path. (.npy exteension)', 
                        required=True)  
    parser.add_argument('-m', '--model_dir',  
                        help='Input directory where the trained model is stored',
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

    if network_cfg['use_resnet']:
        get_net = get_desc_net
    else:
        get_net = get_sonar_net

    record_dir = os.path.join(args.model_dir, 'record')
    eval_stats_fn, best_model_dir = file_utils.make_directories(args.model_dir, True)
    
    print('Resize ratio:', data_cfg['resize_ratio'])

    network = get_net(network_cfg['use_vlad'],train_cfg['use_arcface'], network_cfg['descriptor_size'], network_cfg['n_clusters'])

    begin_epoch = load_model(network, None, None, None, best_model_dir, True)
    test_loader = make_data_loader(args.test_dataset_dir, 1, 'full', data_cfg)
    evaluator = Evaluator(args.from_database,args.database, 'test', data_cfg, eval_cfg, train_cfg['batch_size'], graph=True)

    network = network.cuda()
    network.eval()

    torch.cuda.empty_cache()
    with torch.no_grad():
        evaluator.construct_dataset(network)

    if os.path.exists(args.gt_sim):
        print("Loading GT similarity matrix ...")
        GT_MAP = np.load(args.gt_sim)
        print("Done")
    else:
        GT_MAP = evaluator.generate_gt_similarity_matrix(network, test_loader)
        np.save(args.gt_sim, GT_MAP)
    
    lis = []

    print('Processing the test dataset ...')
    for batch in tqdm.tqdm(test_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['anchor'])
            ###for i in range(output.shape[0]):
            ###    fig = plt.figure(1)
            ###    plt.clf()
            ###    plt.imshow(batch['anchor'].detach().cpu().numpy()[i].transpose(1,2,0), cmap = "gray")
            ###    index = batch['anchor_index'][i].item()
            ###    if index in lis:
            ###        title = str(index) + " ("+str(batch['anchor_yaw'][i].item()) + ")  >> lost"
            ###        fig.set_edgecolor('red')
            ###        fig.set_linewidth(15)
            ###    else:
            ###        fig.set_edgecolor('green')
            ###        fig.set_linewidth(15)
            ###        title = str(index) + " ("+str(batch['anchor_yaw'][i].item()) + ") "
            ###    plt.title(title)
            ###    plt.pause(0.01)
            
            #maps = network.get_feature_map(batch['anchor'])
            #visualize_utils.show_maps(batch['anchor'][0].detach().cpu().numpy(), maps.detach().cpu().numpy(), max_nmaps=16)
            if evaluator is not None:
                evaluator.evaluate(output, batch)

    ###plt.close()
    result = evaluator.summarize(GT_MAP,True)