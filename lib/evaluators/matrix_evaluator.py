import numpy as np
import os.path as path
import time
import matplotlib.pyplot as plt
import cv2 as cv
from lib.utils.file_utils import ImagePoseIndexer, IndicesFilter, load_database, PoseIndexConverter
from lib.utils.image_utils import yaw_cropping, denormalize_image
import lib.utils.eval_utils as eval_utils
import torch
import lib.utils.visualize_utils as visualize_utils
from lib.utils.sonar_utils import sonar_overlap_score


def write_formatted_float_array(array):
    output = '['
    for i, element in enumerate(array):
        output = output + "{:.4f}".format(element)
        if i < len(array) - 1:
            output = output + ', '
    output = output + ']'
    return output

class Evaluator:

    def __init__(self, from_database, data_dir, split, data_cfg, eval_cfg, batch_size,database_fn=path.join("/tmp","sonar_desc_tmp.txt"),graph=False):
        self.split=split
        self.batch_size = batch_size
        self.graph=graph
        self.resize_ratio = data_cfg['resize_ratio']
        self.k_neighbors = eval_cfg['knn']
        self.scan_stride = eval_cfg['scan_stride']
        self.display = eval_cfg['display']
        self.from_database = from_database
        self.min_angle = data_cfg['augmentation']['anchor_yaw_min']
        self.max_angle = data_cfg['augmentation']['anchor_yaw_max']

        self.max_sonar_angle = data_cfg['max_sonar_angle']
        self.sonar_range = data_cfg['sonar_range']
        self.sonar_width = data_cfg['sonar_width']

        if not self.from_database:
            self.pose_dir = path.join(data_dir,'poses')
            self.img_dir = path.join(data_dir, 'imgs')
            if self.split == 'test':
                indices_filter=IndicesFilter(data_cfg['filter'],'test', True)
            else:
                indices_filter=IndicesFilter(data_cfg['filter'],'train', True)
            self.train_data_indexer = ImagePoseIndexer(self.img_dir, self.pose_dir, visible_only=True, indices_filter=indices_filter, resize_ratio=data_cfg['resize_ratio'])
            self.dataset_fn = database_fn
        else:
            self.dataset_fn = data_dir

        self.anchor_descs = []
        self.anchor_indices = []
        self.anchor_poses = []
        self.anchor_yaws = []

    def reset(self):
        self.anchor_descs = []
        self.anchor_indices = []
        self.anchor_poses = []
        self.anchor_yaws = []

        if not self.from_database:
            self.train_data_indexer.reset()
        self.desc_tree = None
        self.net = None
        self.position_descs = None
        self.desired_angle = None
        self.pose_idx_converter = None


    def construct_dataset(self, net, override = False):
        print('Generating evaluation dataset...')
        st = time.time()
        if self.from_database and not override:
            stride, width_desired, desired_angle, _, desc_tree, yaws, position_descs, _ = load_database(self.dataset_fn)
            self.desc_tree = desc_tree
            self.position_descs = position_descs
            self.pose_idx_converter = PoseIndexConverter(stride, width_desired, desired_angle, True)
        else:
            self.desired_angle = self.max_angle - self.min_angle
            self.desc_tree = self.train_data_indexer.tree_from_descriptors(net, self.batch_size, self.scan_stride, self.min_angle, self.max_angle)
            self.pose_idx_converter = self.train_data_indexer.pose_idx_converter
            
        self.net = net
        et = time.time()
        print('Done (', '{:.2f}'.format(et-st), 'seconds )')

    def save_database(self, net):
        self.reset()
        self.train_data_indexer.sort()
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.construct_dataset(net,override=True)
        descriptors = self.desc_tree.data
        print('Writing database - n poses:',len(self.train_data_indexer))
        print('Writing database - n descriptors:',len(descriptors))
        output_file =  open(self.dataset_fn,'w')
        header = str(self.train_data_indexer.stride) + ' ' + str(int(self.train_data_indexer.conv_width_desired)) + ' ' + str(int(self.train_data_indexer.desired_angle)) + ' \n'
        output_file.write(header)
        n_steps = int(np.ceil(self.train_data_indexer.conv_width_desired/self.train_data_indexer.stride))
        for i,sample in enumerate(self.train_data_indexer):
            position = sample.pose
            center = sample.yaw
            for j in range(n_steps):
                yaw = self.train_data_indexer.desc_index_to_angle(j) 
                descriptor = descriptors[n_steps*i + j]
                line = write_formatted_float_array(descriptor)
                line = line + ' '
                line = line + write_formatted_float_array(position)
                line = line + ' '
                line = line + str(int((center - yaw + self.train_data_indexer.desired_angle//2)%360))
                line = line + ' \n'
                output_file.write(line)

    def generate_gt_similarity_matrix(self, net, gt_data_loader):
        if not self.from_database:
            self.save_database(net)
            self.from_database = True
            self.construct_dataset(net)
            self.from_database = False

        db_descs = np.array([[yd[1] for yd in pos.yaw_descriptors] for pos in self.position_descs])
        db_descs = db_descs.reshape(db_descs.shape[0] * db_descs.shape[1], db_descs.shape[2])

        db_yaws = np.array([[yd[0] for yd in pos.yaw_descriptors] for pos in self.position_descs])
        db_yaws = db_yaws.reshape(db_yaws.shape[0] * db_yaws.shape[1])
        
        db_poses = np.array([[pos.position for _ in pos.yaw_descriptors] for pos in self.position_descs])
        db_poses = db_poses.reshape(db_poses.shape[0] * db_poses.shape[1], db_poses.shape[2])

        GTMAP = visualize_utils.generate_gt_distance_image_db_query(gt_data_loader.dataset.anchor_indexer, db_poses, db_yaws)
        return GTMAP

    def evaluate(self, output, batch):
        anchor_desc = output.detach().cpu().numpy()
        anchor_index = batch['anchor_index'].detach().cpu().numpy()
        anchor_pose = batch['anchor_pose'].detach().cpu().numpy()
        anchor_yaw = batch['anchor_yaw'].detach().cpu().numpy()
        #if (83 * 6) in anchor_index:
        #    print(anchor_index)
        #    print(anchor_pose)
        #    print(batch['anchor_yaw'].detach().cpu().numpy())
        self.anchor_descs.extend(anchor_desc)
        self.anchor_indices.extend(anchor_index)
        self.anchor_poses.extend(anchor_pose)
        self.anchor_yaws.extend(anchor_yaw)

    def summarize(self, GT_MAP, show_tsne = False):
        self.anchor_indices = np.array(self.anchor_indices)
        self.anchor_descs = np.array(self.anchor_descs)
        self.anchor_poses = np.array(self.anchor_poses)
        self.anchor_yaws = np.array(self.anchor_yaws)

        idxs = np.argsort(self.anchor_indices)
        self.anchor_descs = self.anchor_descs[idxs,:]
        self.anchor_poses = self.anchor_poses[idxs,:]
        self.anchor_yaws = self.anchor_yaws[idxs]
        self.anchor_indices = self.anchor_indices[idxs]

        db_descs = np.array([d for d in self.desc_tree.data])
        SIM_MAP = visualize_utils.generate_distance_image_db_query(self.anchor_indices, self.anchor_descs, db_descs, None)
        precision, auc, optimal_threshold, all_thresholds = eval_utils.prcurve_from_similarity_matrix(GT_MAP, SIM_MAP, 100, self.graph)
        all_stats = {"precision": precision, "auc": auc, "optimal_threshold":optimal_threshold}  


        sim_img = visualize_utils.generate_distance_image_big(self.anchor_indices , self.anchor_descs, 0)
        for i in range(sim_img.shape[0]):
            sim_img[i,i] = -1
        nn1_neigbors = np.argmax(sim_img,axis = 1)
        overlaps = []
        for anchor_i, neighbor_i in enumerate(nn1_neigbors):
            p1 = self.anchor_poses[anchor_i]
            p2 = self.anchor_poses[neighbor_i]
            y1 = self.anchor_yaws[anchor_i]
            y2 = self.anchor_yaws[neighbor_i]
            overlaps.append(sonar_overlap_score(p1, p2, y1, y2, self.sonar_range, self.sonar_width))
        
        overlaps = np.array(overlaps)
        thresholds = [0.1 * (i+1) for i in range(9)]
        ys = [np.sum(overlaps >= t)/self.anchor_indices.shape[0] for t in thresholds]

        if self.graph:
            plt.title("1-NN retrieval")
            plt.axvline(0.7, color='r')
            plt.plot(thresholds, ys, marker = 'o')
            plt.xlabel("Overlap threshold [%]")
            plt.xticks(thresholds+[1.])
            plt.yticks([0.1*i for i in range(11)])
            plt.ylabel("Accuracy")
            plt.grid(color='k', linestyle='--', linewidth=0.5)
            plt.show()

            threshold = optimal_threshold
            thresholded_map = SIM_MAP >= threshold
            true_positives = (thresholded_map * GT_MAP)*255
            false_positives = (thresholded_map * (1-GT_MAP))*255
            plt.title("Threshold: "+str(threshold)+", TP: "+str(np.sum(true_positives/255))+", FP: "+str(np.sum(false_positives/255)))
            false_negatives = ((1-thresholded_map)*GT_MAP)*255
            output_img = np.stack([false_positives, true_positives, false_negatives], axis =2)
            plt.imshow(output_img)
            plt.show()

            _, axs =plt.subplots(1,2)
            axs[0].imshow(sim_img)
            axs[1].imshow(sim_img > (optimal_threshold))
            plt.show()
            if show_tsne:
                visualize_utils.tsne_clustering(self.anchor_indices, self.anchor_descs, self.anchor_poses)
        self.reset()
        return all_stats
