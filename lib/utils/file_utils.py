import os
import numpy as np
import cv2
import scipy.spatial.kdtree as kdtree
import tqdm
import torch
from lib.utils import image_utils
import math
import time
from sklearn.cluster import DBSCAN, KMeans

def yaw_from_quaternion(quat):
  '''
  Extracts the yaw in degrees from a quaternion rotation representation. The input quaternion is expected to be formatted as x,y,z,w in that order.
  '''
  x = quat[0]
  y = quat[1]
  z = quat[2]
  w = quat[3]
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
  yaw_z_deg = int((yaw_z * 180)/math.pi)
  return yaw_z_deg

def identity_filter(indices):
  return indices == indices

def none_filter(indices):
  return indices != indices

class IndicesFilter:
  '''
  Dataset indices filter

  Parameters:

   - split: str 
       "train", "validation", "test" or "full". The "full" split selects both train and validation sets
   - name: str
        name of the filter to use in lib/utils/filters
   - anchor: bool
       True to select anchor poses, False to select positive/negative candidates
  
  Returns:

  A callable filter on the dataset indices according to the input parameters
  '''
  def __init__(self, name, split, anchor):
    self.split = split
    self.anchor = anchor
    module = __import__('lib.utils.filters.'+name, fromlist=['lib.utils.filters'])
    self.train_filter = module.train_filter
    self.validation_filter = module.validation_filter
    self.test_filter = module.test_filter
    self.anchor_filter = module.anchor_filter
    self.non_anchor_filter = module.non_anchor_filter


  def name(self):
    '''
    For printing purposes
    '''
    anchor_name = ' (anchors)' if self.anchor else ' (non-anchors)'
    return self.split + anchor_name

  def __call__(self, indices):
    if self.split == 'train':
      filter1 = self.train_filter
    elif self.split == 'validation':
      filter1 = self.validation_filter
    #elif self.split == 'full':
    #  filter1 = identity_filter
    else:
      filter1 = self.test_filter
    
    
    if self.anchor:
      if not (self.split == 'lost' or self.split == 'full'):
        filter2 = self.anchor_filter
      else:
        filter2 = identity_filter
    else:
      if not (self.split == 'lost' or self.split == 'full'):
        filter2 = self.non_anchor_filter
      else:
        filter2 = none_filter
    return filter1(indices) & filter2(indices)

def make_directories(model_dir, resume):
    '''
    Generates the necessary training directories starting from the input model_dir. If resume is not True,
    it also deletes any previous trained models in the same directory
    '''
    if not os.path.isdir(model_dir) :
      os.mkdir(model_dir)
    
    best_model_dir = os.path.join(model_dir, 'best_model')
    
    if not os.path.isdir(best_model_dir) :
      os.mkdir(best_model_dir)
      
    eval_stats_fn = os.path.join(model_dir, 'eval_stats.json')

    if not resume :
      if os.path.exists(eval_stats_fn):
        os.remove(eval_stats_fn)

    return eval_stats_fn, best_model_dir

def batched_dataset(instances, net, batch_size:int, cw_360:int, cw_120:int, stride:int, min_angle:int, desired_width: int):
    '''
    Builds the descriptor dataset starting from the dataset images.

    ## Parameters:

    - instances: array of PoseImageTuple
        the dataset items
    - net: torch.nn.Module
        the network used for descriptor computation
    - batch_size: int
        the batch size for descriptor computation
    - cw_360: int
          the width of the output layer of the ResNet backbone for a 360° scan
    - cw_120: int
          the width of the output layer of the ResNet backbone for a 120° scan
    - stride: int
          the stride to apply to the sliding window over the output layer of the ResNet backbone for a 360° scan to compute 120° descriptors
    - min_angle: int
          the maximum negative offset from the image <<center>> (identified as the yaw of the drone when acquiring the scan) for descriptor extraction.
          Useful to avoid extracting descriptors from empty sections of the scan
    - max_angle: int
          analogous to min angle, but is the maximum positive angle
    - desired_width: int
          width in pixels of the section of the convolutional map to consider (the sliding window slides inside this range)

    ## Returns

    An array of shape [n imgs * cw_360/stride, descriptor size] containing the patch descriptors
    '''
    total_n = len(instances)

    cw_60 = int(cw_120 // 2)
    cw_360 = int(cw_360)

    # number of degrees corresponding to a single step of the sliding window (considering the stride)
    deg_per_step = 360/int(np.ceil(cw_360/stride))
    # the number of steps the sliding window is allowed to take inside the desired range
    n_steps_desired = int(np.ceil(desired_width/stride))

    all_descriptors = []

    ## only needed to generate images
    ####instances = sorted(instances, key = lambda x: x.index)
    
    
    for i in tqdm.tqdm(range(0, total_n, batch_size)):
        start = i
        end = min(i+batch_size, total_n)
        batch = np.array([image_utils.normalize_image(instances[i].image) for i in range(start,end)])
        batch = torch.from_numpy(batch).cuda()
        desc = net.get_feature_map(batch)
        curr_centers = np.array([(-1 * instances[i].yaw)%360 for i in range(start,end)])
        for j in range(desc.shape[0]):
          fw_desc = desc[j]
          ### Extract all possible image patches ###

          # 0=-60 <-> 360=300
          fw_desc = torch.roll(fw_desc, cw_60,dims=2) 
          # first 240 degrees 0 - 240
          patches = [fw_desc[:,:,w_i-cw_60:w_i+cw_60] for w_i in range(cw_60,cw_360-cw_60, stride)]
          #  0=60 <-> 360=420 -> 180 = 240
          fw_desc = torch.roll(fw_desc, -2*cw_60, dims = 2)
          start_offset = stride*len(patches) - cw_60
          # last 120 degrees: 240 - 360
          patches.extend([fw_desc[:,:,w_i-cw_60:w_i+cw_60] for w_i in range(start_offset,cw_360-cw_60, stride)])
          patches = torch.stack(patches)

          ### consider only patches inside the 120 degree view of the drone ###

          # minimum patch index inside the valid range
          min_interest = int(np.ceil((curr_centers[j] + min_angle)/deg_per_step))
          # maximum patch index inside the valid range
          max_interest = min_interest + n_steps_desired

          # bring the maximum patch index inside the length of the patches array by rolling (needed due to 360° periodicity)
          overfloat = max(0,max_interest-patches.shape[0])
          if overfloat > 0:
            patches = patches.roll(-overfloat,0)
            min_interest = min_interest - overfloat
            max_interest = max_interest - overfloat
          
          # bring the minimum patch index inside the length of the patches array by rolling (needed due to 360° periodicity)
          underfloat = max(0, 0-min_interest)
          if underfloat > 0:
            patches = patches.roll(underfloat, 0)
            min_interest = min_interest + underfloat
            max_interest = max_interest + underfloat

          patches = patches[min_interest:max_interest] 

          ### extract descriptors from selected patches ###

          patch_descs = net.apply_head(patches)
          all_descriptors.append(patch_descs.detach().cpu())

    all_descriptors = torch.cat(all_descriptors,dim=0).detach().cpu().numpy()
    return all_descriptors

def pose_from_file(fn):
    '''
    Reads position and yaw orientation from dataset annotation files
    '''
    content = open(fn)
    line = content.readline()
    coords = [x for x in line.split(' ') if x!='\n']
    coords = np.array(coords).astype(np.float32)
    coords[np.abs(coords) < 1e-6] = 0.
    line = content.readline()
    quat = np.array([x for x in line.split(' ') if x!='\n']).astype(np.float32)
    yaw = yaw_from_quaternion(quat) % 360
    return np.expand_dims(coords,0), yaw

def scan_from_fn(img_path, resize_ratio, yaw):
  '''
  Reads image from the provided path and resizes according to the ratio. The image values are 
  transformed to floats in the range [0,1] and the image is rolled so that the pixels at x=0 corresponding
  to the yaw angle 0° in word frame. (Assumption: the image loaded has the pixels at x=0 corresponding
  to the yaw angle -180° in drone frame)
  '''
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (int(img.shape[1] * resize_ratio), int(img.shape[0]*resize_ratio)))
  img = np.expand_dims(img,2).astype(np.float32).transpose(2,0,1) / 255
  # after this image 0 is equal to world frame 0
  img = np.roll(img, image_utils.width_for_degrees(-1*(yaw+180), resize_ratio), axis = 2)
  return img

def scan_from_index(img_dir, index, resize_ratio, yaw):
      img_path = os.path.join(img_dir,str(index)+'.jpeg')
      return scan_from_fn(img_path, resize_ratio, yaw)

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
    #if not self.loaded:
    #    self.loaded_img = scan_from_fn(self.image_fn,  self.resize_ratio, self.yaw)
    #    self.loaded = True
    return scan_from_fn(self.image_fn,  self.resize_ratio, self.yaw) #self.loaded_img

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
    img_width = all_instances[0].image.shape[2]
    self.conv_width_360 = np.ceil(img_width / 8)
    self.conv_width_120 = np.ceil(np.floor(img_width/ 3) / 8)
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
  
  def cluster_poses(self, n_clusters):
    all_poses = np.array([i.pose for i in self.all_instances])
    k = n_clusters
    if len(all_poses) < n_clusters:
        k = 1
    clustering = DBSCAN(0.8, 1).fit(all_poses)
    return clustering.labels_

  def compute_descriptor_database(self, net, batch_size, stride, min_angle, max_angle):
    self.load()
    self.desired_angle = max_angle - min_angle
    desired_proportion = int(360 / self.desired_angle)
    # width in pixels of the section of the convolutional map to consider (the sliding window slides inside this range)
    desired_width = int(self.conv_width_360 / desired_proportion)
    self.conv_width_desired = desired_width
    self.stride = stride
    self.pose_idx_converter = PoseIndexConverter(stride, self.desired_angle, self.conv_width_desired)
    all_descriptors = batched_dataset(self.all_instances, net, batch_size, self.conv_width_360, self.conv_width_120, stride, min_angle, desired_width)
    return all_descriptors

  def tree_from_descriptors(self, net, batch_size, stride, min_angle, max_angle):
    all_descriptors = self.compute_descriptor_database(net, batch_size, stride, min_angle, max_angle)
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

  def __init__(self, stride, desired_angle, conv_width_desired, from_database=False):
    '''
    Class that handles conversions from descriptor index to pose and vice-versa.

    ## Parameters:
    - stride: int
        the stride used when sampling descriptors 
    - desired_angle: int
        the maximum distance in degrees between two patches from the same image
        (patches are sampled at at most desired_angle/2 degrees from the drone's yaw to avoid sampling empty space)
    - conv_with_desired: int
        the width (in pixels) of convolutional patches used to extract descriptors
    - from_database: bool
        whether or not the poses and descriptors were read from a descriptor database

    '''
    self.stride = stride
    self.desired_angle = desired_angle
    self.conv_width_desired = conv_width_desired
    self.n_scans = int(np.ceil(self.conv_width_desired/self.stride))
    self.from_database = from_database

  def desc_index_to_pose_index(self,indices):
      if self.from_database:
        return indices
      else:
        return (indices // self.n_scans).astype(np.int32)

  def desc_index_to_angle(self,indices,yaws=None):
    '''
    If the descriptors are loaded from a database then the corresponding angle is returned (from the yaws input parameter).

    If the descriptors are not loaded from a database then the yaws parameters does not need to be specified. The angle returned is
    relative to the start of the sliding window range for the image from which the descriptor is extracted. This angle can be converted to the absolute 
    one by subtracting it to the center of the image and adding desired_angle/2
    '''
    if self.from_database:
      return yaws[indices]
    else:
      scan_ang_width = self.desired_angle / self.conv_width_desired
      return ((indices % self.n_scans) *  self.stride) * scan_ang_width

  def pose_index_to_desc_index(self,indices):
    if self.from_database:
      return indices
    else:
      return (indices * self.n_scans).astype(np.int32)

  def yaw_to_desc_index(self,yaws):
    '''
    still needs figuring out, do not use !


    (relative vs absolute angle)

    The code written here would work if the inputs are angles relative to the start of the sliding window range. However,
    when accepting inputs from the particle filter, the inputs are absolute angles and there is no way to transform them to absolute angles
    if not by associating each input to a dataset image and using its center to transform the angle to the relative one. Is it worth the work?
    Will there really be an instance when the particle filter will not be using the precomputed dataset?
    '''
    raise NotImplementedError()
    scan_ang_width = self.desired_angle / self.conv_width_desired
    inv_stride = 1./self.stride
    return (((yaws / scan_ang_width) * inv_stride) % self.n_scans).astype(np.int32)
  


class PositionYaws:

  def __init__(self, position, max_yaw_diff):
    self.position = position
    self.yaw_descriptors = []
    self.yaw_thresh = max_yaw_diff
  
  def push_yaw(self, yaw, descriptor):
    self.yaw_descriptors.append((yaw, descriptor))

  def match_yaws(self, query_yaw):
    best_desc = None
    best_yaw_dist = 360
    best_yaw = 360

    for yaw, desc in self.yaw_descriptors:
      yaw_diff = abs(yaw - query_yaw) % 360
      if yaw_diff > self.yaw_thresh:
        yaw_diff = 360
      if yaw_diff < best_yaw_dist:
        best_desc = desc
        best_yaw_dist = yaw_diff
        best_yaw = yaw

    return best_desc, best_yaw, best_yaw_dist < self.yaw_thresh



def load_database(fn):
  '''
  Parses a txt database file to extract descriptors and their associated poses.

  ## Parameters:

   - fn: str
      the path to the database file
    
  ## Returns:

  A tuple containing (in this order):
  
   - stride: the stride used when sampling image patches
   - width_desired: int
      the width (in pixels) of convolutional patches used to extract descriptors
   - desired_angle: int
      the maximum distance in degrees between two patches from the same image
      (patches are sampled at at most desired_angle/2 degrees from the drone's yaw to avoid sampling empty space)
   - all_poses_tree: KDTree
      KDTree of all poses associated to a descriptor (elements are in the form [x,y,z,yaw/360])
   - all_descriptors_tree: KDTree
      KDTree of all descriptors
   - yaws: np.ndarray
      array of yaws associated to each descriptor

  the underlying arrays in all_poses_tree, all_descritprs_tree and yaws are matching, meaning that the pose at index i 
  is associated to the yaw at index i and to the descriptor at index i.

  '''
  database_file = open(fn, 'r')

  position_descs = []
  poses = []
  descriptors = []
  yaws = []
  line = database_file.readline()
  pieces = line.split(' ')
  stride = int(pieces[0])
  width_desired = int(pieces[1])
  desired_angle = int(pieces[2])

  print('>> DB header:')
  print('>>>> stride:', stride)
  print('>>>> ',desired_angle,'° width:', width_desired,'px')

  i = 0
  line = database_file.readline()
  while line:
    descriptor, position, yaw = parse_database_line(line)

    new_pose_found = False
    if len(position_descs) == 0:
      new_pose_found = True
    else:
      # float equivalence
      new_pose_found = (np.linalg.norm(position - position_descs[-1].position) > 1e-5)

    if new_pose_found:
      # TODO: parametrize 30
      position_descs.append(PositionYaws(position, 30))   

    position_descs[-1].push_yaw(yaw % 360, descriptor)

    ## just for internal evaluation
    descriptors.append(descriptor)
    yaws.append(yaw)
    poses.append(position)
    line = database_file.readline()
    i = i+1

  positions = np.array([x.position for x in position_descs])
  all_positions_tree = kdtree.KDTree(positions, copy_data=True)

  descriptors = np.array(descriptors)
  poses = np.array(poses)
  yaws = np.array(yaws)
  all_descriptors_tree = kdtree.KDTree(descriptors, copy_data=True)
  all_poses_tree = kdtree.KDTree(poses, copy_data=True)
  print('>> DB content:')
  print('>>> number of positions', len(position_descs))
  print('>>> number of descriptors:',len(descriptors))
  return stride, width_desired, desired_angle, all_poses_tree, all_descriptors_tree, yaws, position_descs, all_positions_tree
  

def parse_database_line(line):
  pieces = line.split(']')
  # avoid including first open bracket
  descriptor = np.fromstring(pieces[0][1:], sep=', ', dtype=np.float32)
  pose = np.fromstring(pieces[1][2:], sep=', ', dtype=np.float32)
  # avoid including \n
  yaw = int(pieces[2][:-2])
  return descriptor, pose, yaw

   

