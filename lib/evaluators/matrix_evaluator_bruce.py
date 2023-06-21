import numpy as np
import os.path as path
import time
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import cv2 as cv
from lib.utils.bruce_utils import ImagePoseIndexer, IndicesFilter, load_database, PoseIndexConverter
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

def show_path_similarities(anchor_poses, nns):
    print(nns.shape, nns[0], str(nns[0]))
    sc = plt.scatter(anchor_poses[:,0], anchor_poses[:,1], c = nns, cmap = 'viridis')
    plt.colorbar(sc)
    plt.show()

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

        self.max_sonar_angle = data_cfg['max_sonar_angle']
        self.sonar_range = data_cfg['sonar_range']
        self.sonar_width = data_cfg['sonar_width']

        if not self.from_database:
            self.pose_dir = path.join(data_dir,'poses')
            self.img_dir = path.join(data_dir, 'imgs')
            if self.split == 'test':
                indices_filter=IndicesFilter(data_cfg['filter'],'train', True)
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
        self.pose_idx_converter = None


    def construct_dataset(self, net, override = False):
        print('Generating evaluation dataset...')
        st = time.time()
        if self.from_database and not override:
            stride, width_desired, _, _, desc_tree, yaws, position_descs, _ = load_database(self.dataset_fn)
            self.desc_tree = desc_tree
            self.position_descs = position_descs
            self.pose_idx_converter = PoseIndexConverter(True)
        else:
            self.desc_tree = self.train_data_indexer.tree_from_descriptors(net, self.batch_size)
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
        header = str(0) + ' ' + str(0) + ' ' + str(0) + ' \n'
        output_file.write(header)
        for i,sample in enumerate(self.train_data_indexer):
            position = sample.pose
            center = sample.yaw
            descriptor = descriptors[i]
            line = write_formatted_float_array(descriptor)
            line = line + ' '
            line = line + write_formatted_float_array(position)
            line = line + ' '
            line = line + str(center)
            line = line + ' \n'
            output_file.write(line)

    def generate_gt_similarity_matrix(self, net, gt_data_loader):
        if not self.from_database:
            self.save_database(net)
            self.from_database = True
            self.construct_dataset(net)
            self.from_database = False
        
        db_descs = []
        db_yaws = []
        db_poses = []
        for pos in self.position_descs:
            for yd in pos.yaw_descriptors:
                db_yaws.append(yd[0])
                db_descs.append(yd[1]) 
                db_poses.append(pos.position)

        db_descs = np.array(db_descs)
        db_yaws = np.array(db_yaws)
        db_poses = np.array(db_poses)

        GTMAP = visualize_utils.generate_gt_distance_image_db_query_bruce(db_poses, db_yaws)

        return GTMAP

    def evaluate(self, output, batch):
        anchor_desc = output.detach().cpu().numpy()
        anchor_index = batch['anchor_index'].detach().cpu().numpy()
        anchor_pose = batch['anchor_pose'].detach().cpu().numpy()
        anchor_yaw = batch['anchor_yaw'].detach().cpu().numpy()
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

        

        ## remove diagonal to avoid matching every sample with itself
        s = 0
        for i in range(SIM_MAP.shape[1]):
            min_i = max(0, i-s)
            max_i = min(SIM_MAP.shape[0], i+s+1)
            for j in range(min_i, max_i):
                GT_MAP[i,j] = 0
                SIM_MAP[i,j] = -1
            GT_MAP[i,i] = 0
            SIM_MAP[i,i] = -1
        precision, auc, optimal_threshold, _ = eval_utils.prcurve_from_similarity_matrix(GT_MAP, SIM_MAP, 100, self.graph)
        all_stats = {"precision": precision, "auc": auc, "optimal_threshold":optimal_threshold}    

        # get closest sample in database
        nn1_neigbors = np.argmax(SIM_MAP,axis = 1)
        nn1_distances = np.max(SIM_MAP, axis=1)

        show_path_similarities(self.anchor_poses, nn1_distances)


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
            print(ys)

            thresholded_map = SIM_MAP >= optimal_threshold
            
            true_positives = (thresholded_map * GT_MAP)*255
            false_positives = (thresholded_map * (1-GT_MAP))*255
            false_negatives = ((1-thresholded_map)*GT_MAP)*255
            output_img = np.stack([false_positives, true_positives, false_negatives], axis =2)
            _,axs = plt.subplots(2,2)
            axs[0][0].imshow(SIM_MAP, vmin = -1, vmax = 1)
            axs[0][1].imshow(output_img)
            axs[1][0].imshow(GT_MAP)
            thresholded_map_2 = SIM_MAP.copy()
            thresholded_map_2[thresholded_map_2 < optimal_threshold] = -1
            axs[1][1].imshow(thresholded_map_2, vmin=-1, vmax=1)
            plt.show()

            if show_tsne:
                visualize_utils.tsne_clustering(self.anchor_indices, self.anchor_descs, self.anchor_poses)
        self.reset()
        return all_stats
