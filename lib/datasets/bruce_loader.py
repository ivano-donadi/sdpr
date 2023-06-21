import torch.utils.data as data
import numpy as np
from random import randrange, choice, shuffle
import os.path as path
from lib.utils.bruce_utils import ImagePoseIndexer, IndicesFilter
from lib.utils.image_utils import normalize_image, denormalize_image
import matplotlib.pyplot as plt
import cv2

def random_jittering(image, p=0.5):
    if np.random.random() > p:
        return image
    image = (image.transpose(1,2,0) * 255).astype(np.uint8)
    h,w,c=image.shape
    noise_b = np.random.randint(0,25,(h,w))
    noise_g = np.random.randint(0,25,(h,w))
    noise_r = np.random.randint(0,25,(h,w))
    zitter = np.zeros_like(image)
    zitter[:,:,0] = noise_b
    zitter[:,:,1] = noise_g
    zitter[:,:,2] = noise_r
    return cv2.add(image, zitter).transpose(2,0,1).astype(np.float32)/255.

class Dataset(data.Dataset):
    def __init__(self,data_dir,split, data_cfg, show_triplets):
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
        # this tree is for positives and negatives only
        if not (self.split == 'test' or self.split == 'lost' or self.split == 'full') :
            self.pose_tree = self.other_indexer.tree_from_poses()
        self.is_lost_injected = False

    def __getitem__(self, index):
        
        if not (self.split == 'test' or self.split == 'lost' or self.split == 'full'):
            exit("Error: unable to train on path dataset")
        else:
            anchor_index, anchor_pose, anchor_img, anchor_visible, anchor_center = self.anchor_indexer[index].as_tuple()       
            anchor, yaw = (anchor_img, anchor_center)
            anchor = np.stack([anchor[0], anchor[0], anchor[0]], axis=0)
            anchor = normalize_image(anchor)
            ret = {'anchor': anchor, 'anchor_index': anchor_index, 'anchor_visible': anchor_visible, 'anchor_pose': anchor_pose,  'anchor_yaw': (yaw)%360}

        return ret

    def __len__(self):
        return len(self.anchor_indexer)
        
def make_data_loader(data_dir, batch_size, split, data_cfg, show_triplets=False):
    if split != 'train':
        batch_size=1
    dataset = Dataset(data_dir,split, data_cfg, show_triplets)
    data_loader = data.dataloader.DataLoader(dataset, batch_size=batch_size,shuffle=False)
    return data_loader


        
