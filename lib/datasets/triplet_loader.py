import torch.utils.data as data
import numpy as np
from random import randrange, choice, shuffle
import os.path as path
from lib.utils.file_utils import ImagePoseIndexer, IndicesFilter
from lib.utils.image_utils import random_yaw_cropping, yaw_cropping_noise, normalize_image, denormalize_image
import lib.utils.visualize_utils as visualize_utils
import matplotlib.pyplot as plt
import cv2


def random_jittering(image, p=0.5):
    if np.random.random() > p:
        return image
    image = (image.transpose(1,2,0) * 255).astype(np.uint8)
    h,w,c=image.shape
    noise = np.random.randint(0,5,(h,w))
    zitter = np.zeros_like(image)
    zitter[:,:,0] = noise
    return np.expand_dims(cv2.add(image, zitter),2).transpose(2,0,1).astype(np.float32)/255.

class Dataset(data.Dataset):
    def __init__(self,data_dir,split, data_cfg, n_clusters, show_triplets):
        super(Dataset, self).__init__()
        self.split=split
        self.resize_ratio = data_cfg['resize_ratio']
        self.n_positives = data_cfg['n_positives']
        self.n_negatives = data_cfg['n_negatives']
        self.show_triplets = show_triplets
        self.jittering = data_cfg['augmentation']['jittering']
        self.pose_dir = path.join(data_dir,'poses')
        self.img_dir = path.join(data_dir, 'imgs')
        visible_only = self.split == 'train'
        lost = self.split == 'lost'
        self.filter_name = data_cfg['filter']
        anchor_filter = IndicesFilter(self.filter_name, self.split, True)
        other_filter = IndicesFilter(self.filter_name, self.split, False)
        self.anchor_indexer = ImagePoseIndexer(self.img_dir, self.pose_dir, visible_only, indices_filter=anchor_filter, resize_ratio=self.resize_ratio)
        self.other_indexer = ImagePoseIndexer(self.img_dir, self.pose_dir,visible_only, indices_filter=other_filter, resize_ratio=self.resize_ratio)
        self.anchor_indexer.load(lost)
        self.other_indexer.load(lost)
        self.n_clusters = n_clusters
        # this tree is for positives and negatives only
        if not (self.split == 'test' or self.split == 'lost'):
            if self.split == 'train':
                pn_fn = data_cfg['pos_neg_map_train']
            else:
                pn_fn = data_cfg['pos_neg_map_val']
            if not path.exists(pn_fn):
                self.pos_neg_map = visualize_utils.generate_positive_negative_map(self.anchor_indexer, self.other_indexer)
                np.save(pn_fn,self.pos_neg_map)
            else:
                self.pos_neg_map = np.load(pn_fn)
            if self.n_clusters is not None:
                self.pose_tree = self.other_indexer.tree_from_poses()
                self.anchor_labels = self.anchor_indexer.cluster_poses(self.n_clusters)
                if self.split == 'train': 
                    print("Anchor clusters = ", n_clusters, np.max(self.anchor_labels))

        self.is_lost_injected = False

    def set_labels(self, labels, centroids):
        for cindex,instance in enumerate(self.anchor_indexer):
            cpose = np.expand_dims(instance.pose,0) # 1x3
            diffs = np.linalg.norm(centroids-cpose)
            best_index = np.argmin(diffs)
            self.anchor_labels[cindex] = labels[best_index]

    def get_labels(self):
        n_clusters = max(self.anchor_labels) + 1
        centroids = np.zeros((n_clusters, 3))
        npoints = np.zeros((n_clusters,1))
        for i in range(len(self.anchor_indexer)):
            cpose = self.anchor_indexer[i].pose
            clabel = self.anchor_labels[i]
            centroids[clabel] += cpose
            npoints[clabel] +=1
        centroids = centroids/npoints
        return [i for i in range(n_clusters)], centroids

    def inject_lost_dataset(self,lost_dir):
        self.is_lost_injected = True
        pose_dir = path.join(lost_dir,'poses')
        img_dir = path.join(lost_dir, 'imgs')
        lost_filter = IndicesFilter(self.filter_name, 'lost', True)
        self.lost_indexer = ImagePoseIndexer(img_dir, pose_dir, False, lost_filter, self.resize_ratio)
        self.lost_indexer.load(True)

    def generate_random_pose(self):
        inf = 0
        sup = len(self.other_indexer)
        array_index = randrange(inf, sup,1)
        anchor_info = self.other_indexer[array_index] 
        return anchor_info

    def generate_n_negatives(self, iterator_index):
        all_negatives = []
        pos_neg_row = self.pos_neg_map[iterator_index,:] # nother
        neg_it_indexes = np.where(pos_neg_row == 0)[0]
        if len(neg_it_indexes) == 0:
            print("Error: no negatives available for anchor",self.anchor_indexer[iterator_index].index)
            quit()
        for _ in range(self.n_negatives):
            c_neg_it_index = neg_it_indexes[randrange(0, len(neg_it_indexes))]
            negative_item = self.other_indexer[c_neg_it_index]
            negative, negative_yaw = random_yaw_cropping(negative_item.image, negative_item.yaw, -1, 1, self.resize_ratio)
            negative = random_jittering(negative, self.jittering)
            negative = normalize_image(negative)
            all_negatives.append(negative)
        return np.array(all_negatives) 

    def generate_n_positives(self, iterator_index):
        pos_neg_row = self.pos_neg_map[iterator_index,:] # nother
        pos_it_indexes = np.where(pos_neg_row == 1)[0]
        if len(pos_it_indexes)==0:
            print("Error: no positives available for anchor",self.anchor_indexer[iterator_index].index)
            quit()
        all_positives = []
        all_positive_indices = []
        all_positive_poses = []
        all_positive_yaws = []
        for _ in range(self.n_positives):
            c_pos_it_index = pos_it_indexes[randrange(0, len(pos_it_indexes))]
            positive_item = self.other_indexer[c_pos_it_index]
            positive, positive_yaw = random_yaw_cropping(positive_item.image, positive_item.yaw, -1, 1, self.resize_ratio)
            positive = random_jittering(positive, self.jittering)
            positive = normalize_image(positive)
            all_positives.append(positive)
            all_positive_yaws.append(positive_item.yaw + positive_yaw)
            all_positive_indices.append(positive_item.index)
            all_positive_poses.append(positive_item.pose)
        return np.array(all_positives), all_positive_indices, all_positive_poses, all_positive_yaws

    def __getitem__(self, index):
        
        if not (self.split == 'test' or self.split == 'lost'):
            aindex, anchor_pose, full_anchor, anchor_visible, anchor_center = self.anchor_indexer[index].as_tuple()
            anchor, yaw = random_yaw_cropping(full_anchor, anchor_center, -1, 1, self.resize_ratio)
            anchor = random_jittering(anchor, self.jittering)
            anchor = normalize_image(anchor)
            ret = {'anchor': anchor, 'anchor_index': aindex, 'anchor_pose':anchor_pose,'anchor_visible': anchor_visible, 'anchor_yaw': (anchor_center + yaw)%360}
            if self.n_clusters is not None:
                alabel = self.anchor_labels[index]
                ret.update({"anchor_label":alabel})
            negative = self.generate_n_negatives(index)
            positive, positive_i, positive_pose, positive_yaw = self.generate_n_positives(index)
            new_ret = {'negative': negative, 'positive':positive}
            ret.update(new_ret)
            if self.show_triplets:
                fig, (ax1,ax2,ax3) = plt.subplots(3,1)
                ax1.set_title('Anchor '+str(aindex)+' '+str(anchor_pose)+' '+str(yaw))
                ax1.imshow(denormalize_image(anchor).transpose(1,2,0))
                ax2.set_title('Positive '+str(positive_i[0])+' '+str(positive_pose[0])+' '+str(positive_yaw[0]))
                ax2.imshow(denormalize_image(positive[0]).transpose(1,2,0))
                ax3.set_title('Negative')
                ax3.imshow(denormalize_image(negative[0]).transpose(1,2,0))
                plt.show()

        else:
            anchor_index, anchor_pose, or_anchor, anchor_visible, anchor_center = self.anchor_indexer[index].as_tuple()       
            or_anchor, yaw = random_yaw_cropping(or_anchor, anchor_center, -1, 1, self.resize_ratio)
            anchor = normalize_image(or_anchor)
            ret = {'anchor': anchor, 'anchor_index': anchor_index, 'anchor_visible': anchor_visible, 'anchor_pose': anchor_pose,  'anchor_yaw': (anchor_center + yaw) % 360, 'or_anchor': or_anchor}

        return ret

    def __len__(self):
        return len(self.anchor_indexer)
        
def make_data_loader(data_dir, batch_size, split, data_cfg, n_clusters, show_triplets=False, lost_dir = None):
    if split != 'train' and split != "validation" and split != "test":
        batch_size=1
    dataset = Dataset(data_dir,split, data_cfg, n_clusters, show_triplets)
    if lost_dir is not None:
        dataset.inject_lost_dataset(lost_dir)
    data_loader = data.dataloader.DataLoader(dataset, batch_size=batch_size,shuffle=(split=='train'))
    return data_loader


        
