import torch
from lib.trainers.trainer import Trainer
from lib.trainers.DescNet import NetworkWrapper
from lib.networks.DescNet import get_desc_net
from lib.networks.SonarNet import get_desc_net as get_sonar_net
from lib.trainers.recorders import Recorder
from lib.optimizers.scheduler import MultiStepLR
from lib.utils.net_utils import load_model, save_model, load_model_arcface, save_model_arcface
from lib.datasets.triplet_loader import make_data_loader
from lib.evaluators.matrix_evaluator import Evaluator
import lib.utils.file_utils
import os
import argparse
import json
import yaml
import lib.utils.image_utils as image_utils
import lib.utils.visualize_utils as visualize_utils
import numpy as np
import lib.utils.sonar_utils as sonar_utils

def parse_args():
    parser = argparse.ArgumentParser(description='DescNet training tool')
    parser.add_argument('-t', '--train_dir', 
                        help='Input directory containing the training dataset', 
                        required=True)    
    parser.add_argument('-v', '--val_dir', 
                        help='Input directory containing the validation dataset', 
                        required=True)   
    parser.add_argument('-g', '--gt_sim', 
                        help='Input ground truth similarity file. If it does not exist it will be computed and saved at the given path. (.npy extension)', 
                        required=True)     
    parser.add_argument('-m', '--model_dir',  
                        help='Output directory where the trained models will be stored',
                        required=True)
    parser.add_argument('--cfg_file', help='Configuration file', required=True)
    parser.add_argument('--epochs',  
                        help='Number of training epochs',
                        required=True)
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

    if network_cfg['use_resnet']:
        get_net = get_desc_net
    else:
        get_net = get_sonar_net

    record_dir = os.path.join(args.model_dir, 'record')
    eval_stats_fn, best_model_dir = lib.utils.file_utils.make_directories(args.model_dir, train_cfg['resume'])
    network = get_net(network_cfg['use_vlad'], train_cfg['use_arcface'], network_cfg['descriptor_size'], network_cfg['n_clusters'])
    if train_cfg['use_arcface']:
        tcp_free = False
        i = 0
        while not tcp_free and i < 10:
            try:
                port = 23450 + i
                torch.distributed.init_process_group(backend='nccl', init_method="tcp://localhost:{0}".format(port), world_size=1, rank=0)
                tcp_free = True
            except:
                i = i+1
    trainer = NetworkWrapper(network, train_cfg['margin'], network_cfg['descriptor_size'],network_cfg['n_clusters'],train_cfg['use_ratio'], train_cfg['use_arcface'])
    trainer = Trainer(trainer)

    params = []
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": train_cfg['learning_rate'], "weight_decay": train_cfg['weight_decay']}]
    
    optimizer = torch.optim.Adam(params, train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=train_cfg['milestones'], gamma=train_cfg['gamma'], reset_threshold=0)
    recorder = Recorder(record_dir, train_cfg['resume'], 'desc_net')

    if train_cfg['use_arcface']: 
        begin_epoch = load_model_arcface(network, trainer.network.module.arc_layer, optimizer, scheduler, recorder, args.model_dir, resume=train_cfg['resume'])
    else:
        begin_epoch = load_model(network, optimizer, scheduler, recorder, args.model_dir, resume=train_cfg['resume'])

    train_loader = make_data_loader(args.train_dir, train_cfg['batch_size'], 'train', data_cfg, network_cfg['n_clusters'],train_cfg['show_triplets'])
    if train_cfg['use_arcface']: 
        train_labels, train_centroids = train_loader.dataset.get_labels()
    validation_loader = make_data_loader(args.val_dir, train_cfg['batch_size'], 'validation', data_cfg, network_cfg['n_clusters'])
    if train_cfg['use_arcface']: 
        validation_loader.dataset.set_labels(train_labels, train_centroids)

    evaluator = Evaluator(False, args.val_dir, 'validation', data_cfg, eval_cfg, train_cfg['batch_size'])

    if not os.path.exists(args.gt_sim):
        GT_MAP = evaluator.generate_gt_similarity_matrix(network, validation_loader)
        print("GT_MAP shape:", GT_MAP.shape)
        np.save(args.gt_sim, GT_MAP)

    end_epoch = int(args.epochs)

    print('Training: start epoch =', begin_epoch, ', last epoch =', end_epoch)

    for epoch in range(begin_epoch, int(args.epochs)):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % train_cfg['save_ep'] == 0:
            if train_cfg['use_arcface']:
                save_model_arcface(network, trainer.network.module.arc_layer, optimizer, scheduler, recorder, epoch, args.model_dir)
            else:
                save_model(network, optimizer, scheduler, recorder, epoch, args.model_dir)

        if (epoch + 1) % train_cfg['eval_ep'] == 0:
            new_stats, loss_val = trainer.val(epoch, validation_loader, args.gt_sim, evaluator = evaluator, recorder=recorder)
            if os.path.exists(eval_stats_fn):
                with open(eval_stats_fn, 'r') as eval_stats:
                    old_stats = json.load(eval_stats)
            else:
                old_stats = {'precision': 0.0, 'auc':0.0} 

            new_score = new_stats['precision'] + new_stats['auc']
            old_score = old_stats['precision'] + old_stats['auc']

            if new_score > old_score :
                filetype = '.pth'
                pths = [int(pth.split('.')[0]) for pth in os.listdir(best_model_dir)  if filetype in pth]
                if pths :
                    os.system('rm {}'.format(os.path.join(best_model_dir, ('{}' + filetype).format(min(pths)))))
                if train_cfg['use_arcface']:
                    save_model_arcface(network, trainer.network.module.arc_layer, optimizer, scheduler, recorder, epoch, best_model_dir)
                else:
                    save_model(network, optimizer, scheduler, recorder, epoch, best_model_dir)

                # do not overwrite stats from other net
                new_acc = new_stats['precision']
                new_auc = new_stats['auc']
                new_stats = old_stats
                new_stats['precision'] = new_acc
                new_stats['auc'] = new_auc
                with open(eval_stats_fn, 'w') as eval_stats:
                    json.dump(new_stats, eval_stats)
