import numpy as np
import os
import cv2
import scipy.spatial.kdtree as kdtree
import tqdm
from . import image_utils
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tqdm
from sklearn.manifold import TSNE
import math
from . import sonar_utils

__SONAR_RANGE__ = None
__SONAR_WIDTH__ = None
__SONAR_ANGLE_THRESHOLD__ = None
__SONAR_SIM_THRESHOLD__ = None

def set_sonar_parameters(max_range, width, max_angle, min_sim):
  global __SONAR_RANGE__
  global __SONAR_WIDTH__
  global __SONAR_ANGLE_THRESHOLD__
  global __SONAR_SIM_THRESHOLD__
  __SONAR_RANGE__ = max_range
  __SONAR_WIDTH__ = width
  __SONAR_ANGLE_THRESHOLD__ = max_angle
  __SONAR_SIM_THRESHOLD__ = min_sim



def generate_gt_similarity_matrix(indices, poses, yaws):
  global __SONAR_RANGE__
  global __SONAR_WIDTH__
  global __SONAR_ANGLE_THRESHOLD__
  global __SONAR_SIM_THRESHOLD__

  nquery = len(indices)
  idxs = np.argsort(indices)
  poses = poses[idxs,:]
  yaws = yaws[idxs]

  print("Generating gt similarity matrix ...")
  image = np.zeros((nquery, nquery), dtype = np.uint8)
  for i in tqdm.tqdm(range(nquery)):
    query_p = poses[i]
    query_y = yaws[i]
    for j in range(i, nquery):
      target_p = poses[j]
      target_y = yaws[j]
      score = sonar_utils.sonar_overlap_score(query_p, target_p, query_y, target_y, __SONAR_RANGE__, __SONAR_WIDTH__, __SONAR_ANGLE_THRESHOLD__)
      if score > __SONAR_SIM_THRESHOLD__:
        image[i,j] = 1
        image[j,i] = 1
  print("Done")

  return image

def generate_distance_image(indices, descriptors, scale, optimal_threshold = 0., mask = False, hide_diagonal=False):
  nimages = len(indices)
  desc_size = descriptors[0].shape[0]
  
  # column - row
  image = np.zeros((nimages, nimages,desc_size), dtype = np.float32)

  if mask:
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
    nbanned = len(banned_indices)
    print("nbanned", nbanned)
    nlegal = nimages - nbanned
    masked_img = np.zeros((nlegal, nlegal,desc_size), dtype = np.float32)
    legal_descriptors = []
    for i, d in zip(indices,descriptors):
      if not (i in banned_indices):
        legal_descriptors.append((i,d))
    legal_descriptors.sort(key=lambda x: x[0])

    for i,d in enumerate(legal_descriptors):
      masked_img[:, i,:] += d[1][None, :]
      masked_img[i,:,:] -= d[1][None, :]

    masked_img = diff_to_sim(masked_img, hide_diagonal, optimal_threshold)
    masked_img = cv2.resize(masked_img,(nlegal*scale, nlegal * scale))

  for i,descriptor in zip(indices,descriptors):
    image[:, i,:] += descriptor[None, :]
    image[i,:,:] -= descriptor[None, :]

  image = diff_to_sim(image, hide_diagonal, optimal_threshold)

  image = cv2.resize(image,(nimages*scale, nimages * scale))

  if mask:
    return image, masked_img, cover_mask(np.ones_like(image), banned_indices)
  else:
    return image

def cover_mask(mask, banned):
  for i in banned:
    mask[i, :] = 0
    mask[:, i] = 0
  return mask


def generate_distance_image_db_query(indices, descriptors, db_descriptors, threshold):
    nquery = len(indices)
    ndb = len(db_descriptors)
    desc_size = descriptors[0].shape[0]
    # sort so that array index corresponds to table index
    idxs = np.argsort(indices)
    descriptors = descriptors[idxs,:]


  
    print("Generating db/query similarity matrix ... ")
    image = np.zeros((nquery, ndb), dtype = np.float32)
    for db_i, db_desc in tqdm.tqdm(enumerate(db_descriptors), total=ndb):
        for query_i, query_desc in enumerate(descriptors):
            sim = np.dot(db_desc, query_desc)#np.linalg.norm(db_desc - query_desc)
            if threshold is None:
              image[query_i, db_i] = sim
            elif sim > threshold:
              image[query_i, db_i] = 1.
    print("Done")
    return image

def generate_gt_distance_image_db_query(indexer, db_poses, db_yaws):
    global __SONAR_RANGE__
    global __SONAR_WIDTH__
    global __SONAR_ANGLE_THRESHOLD__
    global __SONAR_SIM_THRESHOLD__

    nquery = len(indexer)
    ndb = len(db_poses)
    indexer.sort()
  
    # column - row
    print("Generating gt db/query similarity matrix ... ")
    image = np.zeros((nquery, ndb), dtype = np.uint8)
    for db_i, (db_p, db_y) in tqdm.tqdm(enumerate(zip(db_poses,db_yaws)), total = ndb):
      for q_i,instance in enumerate(indexer):

        q_p = instance.pose
        q_y = instance.yaw%360
        score = sonar_utils.sonar_overlap_score(q_p, db_p, q_y, db_y, __SONAR_RANGE__, __SONAR_WIDTH__, __SONAR_ANGLE_THRESHOLD__)
        if score > __SONAR_SIM_THRESHOLD__:
          image[q_i, db_i] = 1
    print("Done")
    return image

def generate_gt_distance_image_db_query_bruce(db_poses, db_yaws):
    global __SONAR_RANGE__
    global __SONAR_WIDTH__
    global __SONAR_ANGLE_THRESHOLD__
    global __SONAR_SIM_THRESHOLD__

    ndb = len(db_poses)
  
    print("Generating gt db/query similarity matrix ... ")
    image = np.zeros((ndb, ndb), dtype = np.uint8)
    for dbi1 in tqdm.tqdm(range(ndb)):
      db_p1, db_y1 = (db_poses[dbi1], db_yaws[dbi1]) 
      for dbi2 in range(dbi1, ndb):
        db_p2, db_y2  = (db_poses[dbi2], db_yaws[dbi2]) 
        score = sonar_utils.sonar_overlap_score(db_p1, db_p2, db_y1, db_y2, __SONAR_RANGE__, __SONAR_WIDTH__, __SONAR_ANGLE_THRESHOLD__)
        if score > __SONAR_SIM_THRESHOLD__:
          image[dbi1, dbi2] = 1
          image[dbi2, dbi1] = 1
    print("Done")
    return image

def generate_gt_distance_image_indexer_indexer(indexer_one, indexer_two):
    global __SONAR_RANGE__
    global __SONAR_WIDTH__
    global __SONAR_ANGLE_THRESHOLD__
    global __SONAR_SIM_THRESHOLD__

    nquery = len(indexer_one)
    ndb = len(indexer_two)
    indexer.sort()
  
    # column - row
    print("Generating gt iterator/iterator similarity matrix ... ")
    image = np.zeros((nquery, ndb), dtype = np.uint8)
    for db_i, db_instance in enumerate(indexer_two):
      db_p = db_instance.pose
      db_y = db_instance.yaw%360
      for q_i,q_instance in enumerate(indexer_one):
        q_p = q_instance.pose
        q_y = q_instance.yaw%360
        score = sonar_utils.sonar_overlap_score(q_p, db_p, q_y, db_y, __SONAR_RANGE__, __SONAR_WIDTH__, __SONAR_ANGLE_THRESHOLD__)
        if score > __SONAR_SIM_THRESHOLD__:
          image[q_i, db_i] = 1
    print("Done")
    return image

def generate_positive_negative_map(anchor_indexer, non_anchor_indexer):

  global __SONAR_RANGE__
  global __SONAR_WIDTH__
  global __SONAR_ANGLE_THRESHOLD__
  global __SONAR_SIM_THRESHOLD__

  nanchor = len(anchor_indexer)
  nother = len(non_anchor_indexer)

  print("Generating positive/negative map ...")
  image = np.zeros((nanchor, nother), dtype = np.uint8)
  for i, anchor in tqdm.tqdm(enumerate(anchor_indexer), total = nanchor):
    for j, other in enumerate(non_anchor_indexer):
      score = sonar_utils.sonar_overlap_score(anchor.pose, other.pose, anchor.yaw, other.yaw, __SONAR_RANGE__, __SONAR_WIDTH__, __SONAR_ANGLE_THRESHOLD__)
      if score > __SONAR_SIM_THRESHOLD__:
        image[i,j] = 1
  print("Done")
  return image

def generate_distance_image_big(indices, descriptors, threshold):
  nquery = len(indices)
  desc_size = descriptors[0].shape[0]

  idxs = np.argsort(indices)
  descriptors = descriptors[idxs,:]

  print("Generating similarity matrix ...")
  image = np.zeros((nquery, nquery), dtype = np.float32)
  for i in tqdm.tqdm(range(nquery)):
    query_desc = descriptors[i]
    for j in range(i, nquery):
      target_desc = descriptors[j]
      sim = np.dot(target_desc, query_desc)
      #if sim > threshold:
      image[i,j] = sim
      image[j,i] = sim

  print("Done")

  return image

def diff_to_sim(image, hide_diagonal, optimal_threshold):
  # get distances
  image = np.linalg.norm(image, axis = 2)
  mask = image > optimal_threshold

  # normalize distances between 0-1
  max_dist = image.max()
  min_dist = image.min()

  if hide_diagonal:
    for i in range(image.shape[0]):
      image[i,i] = max_dist
  
  image = (image - min_dist)/(max_dist-min_dist)

  # 1 - distance to get similarity
  image = 1 - image
  image[mask] = 0

  # turn into uint8 type
  #image = np.ndarray.astype(image, np.uint8)
  return image

def show_similarity_img(anchor_indexes, anchor_descs, lost_descs, optimal_threshold):
    distance_img = generate_distance_image(anchor_indexes, anchor_descs, 1, optimal_threshold)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(distance_img)


    max_aind= np.max(np.array(anchor_indexes))
    lost_indexes = [max_aind + 1 + i for i in range(len(lost_descs))]

    anchor_indexes.extend(lost_indexes)
    anchor_descs.extend(lost_descs)

    distance_img2 = generate_distance_image(anchor_indexes, anchor_descs, 1, optimal_threshold)
    ax2.imshow(distance_img2)

    plt.show()

def show_similarity_img_big(anchor_indexes, anchor_descs, lost_descs, optimal_threshold):

    max_aind= np.max(np.array(anchor_indexes))
    lost_indexes = [max_aind + 1 + i for i in range(len(lost_descs))]

    anchor_indexes.extend(lost_indexes)
    anchor_descs.extend(lost_descs)

    distance_img2 = generate_distance_image_big(anchor_indexes, anchor_descs, optimal_threshold)
    plt.imshow(distance_img2)
    plt.show()

def tsne_clustering(anchor_indexes, anchor_descs, anchor_poses, lost_indices=[]):
    tsne = TSNE(n_components=2, random_state=42)
    projections = tsne.fit_transform(anchor_descs)

    min_x = np.min(projections[:,0])
    max_x = np.max(projections[:,0])
    min_y = np.min(projections[:,1])
    max_y = np.max(projections[:,1])

    step_x = 1/(max_x-min_x)
    step_y = 1/(max_y-min_y)

    #_, ax = plt.subplots(1,2)

    for index in range(len(projections)):
        r = (projections[index][0]-min_x)*step_x
        g = (projections[index][1]-min_y)*step_y
        b = 0
        #ax[0].plot(projections[index][0], projections[index][1],marker = 'o', color=(r,g,b))

    for i,(ai,pose) in enumerate(zip(anchor_indexes, anchor_poses)):
        if ai in lost_indices:
            r = 0
            g = 0
            b = 1
        else:
            r = (projections[i][0]-min_x)*step_x
            g = (projections[i][1]-min_y)*step_y
            b = 0
        #ax[1].plot(pose[0], pose[1], marker = 'o', color=(r,g,b))
        plt.plot(pose[0], pose[1], marker = 'o', color=(r,g,b))
    plt.gcf().set(dpi=200)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return list(int(h[i:i+2], 16) for i in (0, 2, 4))

def show_maps(img, maps, max_nmaps = 256):
    nmaps = min(maps.shape[1], max_nmaps)
    nmaps_total = maps.shape[1]
    height = maps.shape[2]
    width = maps.shape[3]
    if nmaps == 64:
        nrows = 8 + 1
        ncols = 8
    elif nmaps == 32:
        nrows = 4 + 1
        ncols = 8
    elif nmaps == 16:
        nrows = 4 + 1
        ncols = 4
    elif nmaps == 128:
        nrows = 16 + 1
        ncols = 8
    elif nmaps == 256:
        nrows = 16 + 1
        ncols = 16
    elif nmaps == 512:
        nrows = 16 + 1
        ncols = 32
    else:
        nrows = 8
        ncols = 1
    
    fig = plt.figure()
    #gs0 = gridspec.GridSpec(1, 2, figure=fig)
    #gs00 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs0[1])
    gs0 = gridspec.GridSpec(1, 1, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs0[0])

    #ax0 = fig.add_subplot(gs0[0])
    #ax0.imshow(image_utils.denormalize_image(img).transpose(1,2,0))
    #ax0.axis('off')
    or_img = image_utils.denormalize_image(img)
    or_img_width = or_img.shape[2]
    or_img_height = or_img.shape[1]
    or_img_rescaled = cv2.resize(or_img,(width, height))
    #plt.show()
    map_sum = np.zeros_like(or_img[0,:,:])
    for j in range(nmaps_total):
        if j < nmaps:
            row = int(j // ncols)
            column = int(j % ncols)
            axs = fig.add_subplot(gs00[row, column])
            axs.axis('off')
            #axs.imshow(maps[0,j] * or_img_rescaled)
            map_rescaled = cv2.resize(maps[0,j],(or_img_width, or_img_height))
            axs.imshow( maps[0,j]) # * or_img[:,:,0]
        map_sum = map_sum + cv2.resize(maps[0,j],(or_img_width, or_img_height))
    axs = fig.add_subplot(gs00[nrows-1,0])
    axs.axis('off')
    #axs.imshow(or_img_rescaled)
    axs.imshow(image_utils.apply_cfar(or_img, bgr=True).transpose(1,2,0))


    max_m = map_sum.max()
    min_m = map_sum.min()
    map_sum = (map_sum - min_m)/(max_m - min_m)
    axs = fig.add_subplot(gs00[nrows-1,1])
    axs.axis('off')
    axs.imshow(map_sum)

    axs = fig.add_subplot(gs00[nrows-1,2])
    axs.axis('off')
    axs.imshow(image_utils.apply_cfar(or_img, bgr=True).transpose(1,2,0))
    axs.imshow(map_sum, alpha=0.5,cmap="Greys")

    plt.axis('off')
    plt.show()
