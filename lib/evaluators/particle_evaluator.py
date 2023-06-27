import os.path as path
import lib.utils.file_utils as file_utils
import lib.utils.image_utils as image_utils
import time
import torch
import torch.nn as nn
import numpy as np
import cv2

class ParticleEvaluator:

    def __init__(self, from_database:bool, data_dir: str, data_cfg: dict, batch_size: int, scan_stride: int, distance_threshold: float):
        '''
        ## Parameters:

        data_dir: str
            path to the dataset folder containing the "poses" and "imgs" subfolders
        data_cfg: dict
            "data" configuration dictionary
        batch_size: int
            batch size to use when building the descriptor dataset
        scan_stride: int
            stride to applyto sliding window over dataset descriptors
        distance_threshold: float
            maximum threshold for valid descriptor distance
        '''
        self.from_database = from_database
        self.resize_ratio = data_cfg['resize_ratio']
        self.img_width = data_cfg['test_width']
        self.img_height = data_cfg['test_height']
        self.batch_size = batch_size
        self.scan_stride = scan_stride
        self.distance_threshold = distance_threshold
        if not self.from_database:
            self.pose_dir = path.join(data_dir,'poses')
            self.img_dir = path.join(data_dir, 'imgs')
            indices_filter=file_utils.IndicesFilter(data_cfg['filter'],'full', True)
            self.data_indexer = file_utils.ImagePoseIndexer(self.img_dir, self.pose_dir, visible_only=True, indices_filter=indices_filter, resize_ratio=self.resize_ratio)
        else:
            self.dataset_fn = data_dir

    def build_dataset(self, net: nn.Module):
        '''
        ## Parameters:

        net: torch.nn.Module
            trained descriptor generator network

        ## Effects:

        Builds a pose KD tree from the dataset poses and another from the corresponding descriptors
        '''
        print('Building descriptor dataset ...')
        st = time.time()
        if not self.from_database:
            self.pose_tree = self.data_indexer.tree_from_poses()
            self.desc_db = self.data_indexer.compute_descriptor_database(net,self.batch_size, self.scan_stride)
            self.pose_idx_converter = self.train_data_indexer.pose_idx_converter
            self.db_yaws = None
        else:
            stride, width_desired, desired_angle, _, _, _, position_descs, positions_tree = file_utils.load_database(self.dataset_fn)
            self.position_descs = position_descs
            self.positions_tree = positions_tree
            self.pose_idx_converter = file_utils.PoseIndexConverter(stride, width_desired, desired_angle, True)
        self.net = net
        et = time.time()
        print('Done (', '{:.2f}'.format(et-st), 'seconds )')

    def evaluate_particles(self, particle_poses: np.ndarray, particle_yaws: np.ndarray, current_scan: np.ndarray):
        '''
        ## Parameters:

        particle_poses: numpy.ndarray of shape (n_particles,3)
            array of particles' 3D (x,y,z) locations in space
        particle_yaws: numpy.ndarray of shape (n_particles)
            array of particles' yaw orientation in degrees
        current_scan: numpy.ndarray of shape (height, width, 3) and dtype int
            raw RGB image of the current sonar scan (120°)

        ## Returns:

        a numpy.ndarray of shape (n_particles) where each element contains the distance of
        the 'current_scan' descriptor from the descriptor associated with the corresponding pose in 
        the dataset. If the distance is over the confidence threshold, the distance returned will be negative

        '''
        # resize scan, convert to float and normalize
        if current_scan.shape[2] > 1:
            current_scan = cv2.cvtColor(current_scan, cv2.COLOR_BGR2GRAY)[:,:,None]
        
        current_scan = np.asarray(current_scan).astype(np.float32).transpose(2,0,1) / 255
        current_scan = image_utils.normalize_image(current_scan)
        #print(current_scan.shape)
        #current_scan = cv2.resize(current_scan,(self.img_width, self.img_height))
        
        current_desc = self.net(torch.from_numpy(current_scan).unsqueeze(0).cuda())[0].detach().cpu().numpy()
        particle_yaws = particle_yaws % 360

        # retrieve closest poses in dataset
        if not self.from_database:
            print("Error: not yet implemented")
            quit()


        # retrieve closest pose and associated descriptor
        position_dists, position_indices = self.positions_tree.query(particle_poses, 1)
        descriptors = np.zeros((len(particle_poses), current_desc.shape[0]), dtype=np.float32)
        failed_matches = []
        for j,pi in enumerate(position_indices):
            desc, _, yaw_match = self.position_descs[pi].match_yaws(particle_yaws[j])
            if desc is not None:
                descriptors[j] = desc
            failed_matches.append(not yaw_match)
        # compute descriptor distance between query and retrieved
        distances = np.dot(current_desc, descriptors.transpose(1,0))
        # threshold on descriptor distance, position distance and angle distance
        distances[distances < self.distance_threshold] = -1.
        distances[position_dists > 5.] = -1
        distances[np.array(failed_matches)] = -1

        distances[distances > 0] = 1 - distances[distances > 0] 
        return distances


        