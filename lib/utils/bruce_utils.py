from .file_utils import IndicesFilter, load_database, pose_from_file
import numpy as np
import os
import cv2
import scipy.spatial.kdtree as kdtree
import tqdm
from . import image_utils
import torch

def batched_dataset(instances, net, batch_size:int):
    total_n = len(instances)
    all_descriptors = []
    for i in tqdm.tqdm(range(0, total_n, batch_size)):
        start = i
        end = min(i+batch_size, total_n)
        batch = np.array([image_utils.normalize_image(instances[i].image) for i in range(start,end)])
        batch = torch.from_numpy(batch).cuda()
        descs = net(batch)
        all_descriptors.append(descs.detach().cpu())
    all_descriptors = torch.cat(all_descriptors,dim=0).detach().cpu().numpy()
    return all_descriptors

def scan_from_fn(img_path, resize_ratio):
  '''
  Reads image from the provided path and resizes according to the ratio. The image values are 
  transformed to floats in the range [0,1] and the image is rolled so that the pixels at x=0 corresponding
  to the yaw angle 0° in word frame. (Assumption: the image loaded has the pixels at x=0 corresponding
  to the yaw angle -90° in drone frame)
  '''
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.expand_dims(img,2).astype(np.float32).transpose(2,0,1) / 255
  return img

class PoseImageTuple:

  def __init__(self, index, pose, image_fn, visible, yaw, resize_ratio):
    self.index = index
    self.pose = pose
    self.image_fn = image_fn
    self.yaw = yaw
    self.visible=  visible
    self.resize_ratio = resize_ratio
    self.loaded = False

  def as_tuple(self):
    return self.index, self.pose, self.image, self.visible, self.yaw
  
  @property
  def image(self):
    if not self.loaded:
        self.loaded_img = scan_from_fn(self.image_fn,  self.resize_ratio)
        self.loaded = True
    return self.loaded_img

class ImagePoseIndexer:

  def __init__(self, img_dir, pose_dir, visible_only, indices_filter, resize_ratio):
    self.img_dir = img_dir
    self.pose_dir = pose_dir
    self.visible_only = visible_only
    self.indices_filter = indices_filter
    self.loaded = False
    self.resize_ratio = resize_ratio

  def load(self, lost = False):

    if self.loaded:
      return

    all_pose_files = np.array([file for file in os.listdir(self.pose_dir) if 'txt' in file])
    all_indices = np.array([fn.split('.')[0] for fn in all_pose_files]).astype(np.int32)

    filter_pose = self.indices_filter(all_indices)
    all_indices = all_indices[np.where(filter_pose)]

    if len(all_indices) == 0:
      print("Warn:",self.indices_filter.name(), 'has nothing to load')
      self.all_instances = []
      self.loaded = True
      return

    # from here on, the two sets of indices are compatible with each other

    all_poses_yaws = [pose_from_file(os.path.join(self.pose_dir,str(i)+'.txt')) for i in all_indices]
    all_poses = [py[0] for py in all_poses_yaws]
    all_yaws = [py[1] for py in all_poses_yaws]
    all_poses = np.concatenate(all_poses,axis=0)
    all_yaws = np.asarray(all_yaws)
    all_images = [str(i)+'.jpeg' for i in all_indices]
    all_visibles = np.zeros_like(all_indices).astype(bool) if lost else np.ones_like(all_indices).astype(bool) 

    if lost:
      print(self.indices_filter.name(),'non visible images:',len(all_visibles) - all_visibles.sum())
    else:
      print(self.indices_filter.name(),'visible images:',all_visibles.sum())

    all_instances = [PoseImageTuple(index, pose, os.path.join(self.img_dir, image_fn), visible, yaw, self.resize_ratio) for index,pose,image_fn, visible, yaw in zip(all_indices, all_poses, all_images, all_visibles, all_yaws)]

    self.all_instances = all_instances
    self.loaded = True
    self.sort()
    print(self.indices_filter.name(), ' done loading...',sep = '')

  def __getitem__(self, index):
    return self.all_instances[index]

  def __len__(self):
    return len(self.all_instances)

  def sort(self):
    self.load()
    self.all_instances.sort(key=lambda x: x.index)

  def reset(self):
    self.loaded = False
    self.all_instances = None

  def tree_from_poses(self):
    self.load()
    all_poses = np.array([i.pose for i in self.all_instances])
    all_poses_tree = kdtree.KDTree(all_poses, copy_data=True)
    return all_poses_tree

  def compute_descriptor_database(self, net, batch_size):
    self.load()
    self.pose_idx_converter = PoseIndexConverter()
    all_descriptors = batched_dataset(self.all_instances, net, batch_size)
    return all_descriptors

  def tree_from_descriptors(self, net, batch_size):
    all_descriptors = self.compute_descriptor_database(net, batch_size)
    all_descriptors = kdtree.KDTree(all_descriptors, copy_data=True)
    return all_descriptors

  def desc_index_to_pose_index(self,indices):
      return self.pose_idx_converter.desc_index_to_pose_index(indices)

  def desc_index_to_angle(self,indices):
    return self.pose_idx_converter.desc_index_to_angle(indices)

  def pose_index_to_desc_index(self,indices):
    return self.pose_idx_converter.pose_index_to_desc_index(indices)

  def yaw_to_desc_index(self,yaws):
    return self.pose_idx_converter.yaw_to_desc_index(yaws)

class PoseIndexConverter:

  def __init__(self, from_database=False):
    '''
    Class that handles conversions from descriptor index to pose and vice-versa.

    ## Parameters:
    - from_database: bool
        whether or not the poses and descriptors were read from a descriptor database

    '''
    # the number of descriptors which can be extracted from a single image
    self.n_scans = 1
    self.from_database = from_database

  def desc_index_to_pose_index(self,indices):
      if self.from_database:
        return indices
      else:
        return (indices // self.n_scans).astype(np.int32)

  def desc_index_to_angle(self,indices,yaws=None):
    # 120 deg images, cut is always on center
    if self.from_database:
      return yaws[indices]
    else:
      return 0

  def pose_index_to_desc_index(self,indices):
      return indices

  def yaw_to_desc_index(self,yaws):
    raise NotImplementedError()





