import numpy as np
import torch
import random
import cv2

dataset_mean = np.expand_dims(np.array([0.485, 0.456, 0.406]),(1,2)).astype(np.float32)
dataset_std = np.expand_dims(np.array([0.229, 0.224, 0.225]),(1,2)).astype(np.float32)

from lib.utils.cfar.CFAR import CFAR
__cfar_detector__ = None
__cfar_alg__ = None
__cfar_threshold__ = None

__train_img_width__ = None
__test_img_width__ = None
__test_img_height__ = None

def set_train_img_width(w):
  global __train_img_width__
  __train_img_width__ = w

def set_test_img_size(w,h):
  global __test_img_width__
  global __test_img_height__
  __test_img_width__ = w
  __test_img_height__ = h

def set_detector(name):
  global __cfar_detector__
  global __cfar_alg__
  global __cfar_threshold__
  if name == "GOCA":
    Ntc = 20
    Ngc = 4
    Pfa = 5e-2
    rank = Ntc // 2
    __cfar_threshold__ = 1
    __cfar_alg__ = "GOCA"
    __cfar_detector__ = CFAR(Ntc, Ngc, Pfa, rank)
  elif name == "SOCA":
    Ntc = 40
    Ngc = 10
    Pfa = 0.1
    rank = 10
    __cfar_threshold__ = 16
    __cfar_alg__ = "SOCA"
    __cfar_detector__ = CFAR(Ntc, Ngc, Pfa, rank)
  elif name == "SOCA2":
    Ntc = 40
    Ngc = 10
    Pfa = 5e-2
    rank = Ntc//2
    __cfar_threshold__ = 128
    __cfar_alg__ = "SOCA"
    __cfar_detector__ = CFAR(Ntc, Ngc, Pfa, rank)
  else:
    print("Error: CFAR detector {0} not supported, availabe values are SOCA,GOCA".format(name))
    quit()

def apply_cfar(img, bgr=False):
  global __cfar_detector__
  global __cfar_alg__
  global __cfar_threshold__
  global __test_img_width__
  global __test_img_height__
  #if len(img.shape) >= 3 and img.shape[0] > 1:
  #  img = cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_BGR2GRAY)
  img = img[0]
  img = img * 255
  peaks = __cfar_detector__.detect(img, __cfar_alg__)
  peaks &= img > __cfar_threshold__
  peaks = cv2.resize(peaks, (int(__test_img_width__), int(__test_img_height__)))
  if bgr:
      peaks = np.concatenate([peaks[None,:,:].astype(np.float32), peaks[None,:,:].astype(np.float32), peaks[None,:,:].astype(np.float32)], axis = 0)
  else:
      peaks = peaks[None,:,:].astype(np.float32)
  return peaks


def width_for_degrees(degrees: int, resize_ratio = 1):
    '''
    Return the width in pixels for the requested number of degrees scaled according to the resize ratio

    '''
    global __train_img_width__
    deg_1_width = __train_img_width__/360
    res = deg_1_width * degrees * resize_ratio
    return int(res)

def normalize_image(img):
  '''
  Normalizes images according to the dataset statistics used when pretraining the network (ImageNet mean and std)

  ## Parameters:
  
  - img: np.ndarray of shape [n channels, height, width]

  ## Returns

  The normalized image with the same shape as the input
  '''
  img = apply_cfar(img, bgr=False)
  #img = img - dataset_mean
  #img = img / dataset_std
  return img

def denormalize_image(img):
  '''
  Turns back a normalized image into [0.,1.] range

  ## Parameters:
  
  - img: np.ndarray of shape [n channels, height, width]

  ## Returns

  The de-normalized image with the same shape as the input
  '''
  #img = img * dataset_std[0]
  #img = img + dataset_mean[0]
  #minv = np.min(img)
  #maxv = np.max(img)
  #img = (img - minv)
  #if maxv > minv:
  #    img = img/(maxv-minv)
  return img

def random_yaw_cropping(img, center:int, min:int, max:int, resize_ratio = 1):
  '''
  Crops a 120° view of the input image with a random yaw offset between [min;max] degrees from the image center.

  ## Parameters:

  - img: np.ndarray of shape [n channels, height, width] (360° image)
  - center: int
        the yaw of the drone when capturing the image
  - min: int
        minimum yaw offset from the center (can be negative)
  - max: int
        maximum yaw offset from the center
  - resize_ratio: int
        resize ratio applied to the image. It is used to compute the output width
        and not to resize the input image, which should be already correctly scaled

  ## Returns:

  A tuple with the cropped image as the first element and the random yaw offset as the second
  '''
  random_yaw = random.randint(min,max)
  return yaw_cropping(img, center, random_yaw, resize_ratio)

def yaw_cropping_noise(img,center, yaw, resize_ratio = 1, max_noise=30):
  '''
  Crops a 120° view of the input image with a yaw offset from the image center perturbed by a random noise of +- max_noise degrees from the input offset.

  ## Parameters:

  - img: np.ndarray of shape [n channels, height, width] (360° image)
  - center: int
        the yaw of the drone when capturing the image
  - yaw: int
        the input yaw offset
  - max_noise: int
        maximum distance in degrees between the input yaw offset and the final one
  - resize_ratio: int
        resize ratio applied to the image. It is used to compute the output width
        and not to resize the input image, which should be already correctly scaled

  ## Returns:

  A tuple with the cropped image as the first element and the random yaw offset as the second
  '''
  noise = random.randint(-1*max_noise,max_noise)
  new_yaw = yaw + noise
  return yaw_cropping(img, center, new_yaw, resize_ratio)
  

def yaw_cropping(img, center, yaw, resize_ratio = 1):
  '''
  Crops a 120° view of the input image with a yaw offset from the image center.

  ## Parameters:

  - img: np.ndarray of shape [n channels, height, width] (360° image)
  - center: int
        the yaw (in degrees) of the drone when capturing the image
  - yaw: int
        the input yaw offset in degrees.
  - resize_ratio: int
        resize ratio applied to the image. It is used to compute the output width
        and not to resize the input image, which should already be correctly scaled

  ## Returns:

  A tuple with the cropped image as the first element and the input yaw offset as the second
  '''
  width_60 = width_for_degrees(60,resize_ratio)
  orig_yaw = yaw

  yaw = ((-1*center) - yaw)%360
  
  # roll the image so that the center angle (yaw) has at least 60° at both sides and modify yaw to 
  # reflect the angle shift given by the roll
  if yaw > 300:
    yaw = yaw - 300
    if torch.is_tensor(img):
      img = torch.roll(img, width_for_degrees(-300,resize_ratio),dims=3)
    else:
      img = np.roll(img, width_for_degrees(-300, resize_ratio), axis = 2)

  if yaw < 60:
    yaw = yaw + 60
    if torch.is_tensor(img):
      img = torch.roll(img, width_for_degrees(60,resize_ratio),dims=3)
    else:
      img = np.roll(img, width_for_degrees(60, resize_ratio), axis = 2)

  # center the cut at the disired angle and then crop 60° to the left and to the right
  yaw_width = width_for_degrees(yaw, resize_ratio)
  if torch.is_tensor(img):
    crop = img[:,:,:,yaw_width-width_60:yaw_width+width_60]
  else:
    crop = img[:,:,yaw_width-width_60:yaw_width+width_60]
    
  return crop, orig_yaw
