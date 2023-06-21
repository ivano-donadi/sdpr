import numpy as np
import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import scipy.ndimage
import lib.utils.image_utils as image_utils
import pywt
import argparse
import yaml

def saturated(img):
    res = img.copy()
    res[res < 0] = 0
    res[res > 255] = 255
    return res

def gaussian_noise(img):
    row,col= img.shape
    mean = 0
    var = 15
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    gauss[gauss < 0] = 0
    return gauss

def full_range(img):
    res = img.copy()
    res = (res-np.min(res))/(np.max(res)-np.min(res))
    return res * 255

def BM_threshold(data, M, alpha, nlevels, level, mode):
    K = int(M / (nlevels + 1 - level))
    sorted_data = np.sort(data.flatten())
    threshold = sorted_data[-K]
    data = pywt.threshold(data, threshold, mode=mode)

def parse_args():
    parser = argparse.ArgumentParser(description='Image denoising tool')
    parser.add_argument('-d', '--data_dir', 
                        help='Input directory containing the dataset', 
                        required=True)
    parser.add_argument('-s','--save', action='store_true',
                        help='If this option is specified the images will be overwritten, otherwise the denoised images will just be displayed')
    parser.add_argument('--cfg_file', help='Configuration file', required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

args = parse_args()
img_dir = args.data_dir
save_imgs = args.save

with open(args.cfg_file, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Exception(exc)

image_utils.set_detector(cfg["data"]["cfar_algorithm"])
image_utils.set_test_img_size(cfg["data"]['test_width'],cfg["data"]['test_height'])
fns = os.listdir(img_dir)
fns.sort()

tot_img = len(fns)

sum_img = None

for fn in tqdm.tqdm(fns):
    img = cv2.imread(os.path.join(img_dir,fn),cv2.IMREAD_UNCHANGED).astype(np.float64)
    if len(img.shape) > 2:
        img = img[:,:,0]
    if sum_img is None:
        sum_img = img
    else:
        sum_img = sum_img + img

series_avg = sum_img/tot_img
thr = np.max(series_avg)*0.07
series_avg += int(thr)
series_avg = series_avg

for fn in tqdm.tqdm(fns):
     img = cv2.imread(os.path.join(img_dir,fn),cv2.IMREAD_UNCHANGED).astype(np.float64)
     if len(img.shape) > 2:
        img = img[:,:,0]
     
     _,ax = plt.subplots(2,2)
     ax[0][0].imshow(saturated(saturated(img-series_avg)), cmap='gray', vmin=0, vmax = 255)
     ax[0][0].title.set_text("normalized")

     input_img = img-series_avg
     
     decomp_level = 2
     mode = 'hard'
     wave = 'sym4'
     alpha = 1

     LLs = [input_img]
     HLs = [None]
     LHs = [None]
     HHs = [None]
     
     for i in range(1,decomp_level+2):
        LL, (HL,LH,HH) = pywt.dwt2(LLs[i-1], wave)
        LLs.append(LL)
        HLs.append(HL)
        LHs.append(LH)
        HHs.append(HH)

     m = (LLs[decomp_level+1].shape[0]*LLs[decomp_level+1].shape[1])

     for i in range(decomp_level+1,0,-1):
        if i < decomp_level + 1:
            HLs[i] = BM_threshold(HLs[i], m, alpha, decomp_level, i, mode)
            LHs[i] = BM_threshold(LHs[i], m, alpha, decomp_level, i, mode)
            HHs[i] = BM_threshold(HHs[i], m, alpha, decomp_level, i, mode)
        next_approx = pywt.idwt2((LLs[i],(HLs[i], LHs[i], HHs[i])), wave)
        next_approx = cv2.resize(next_approx, (LLs[i-1].shape[1], LLs[i-1].shape[0]))
        LLs[i-1] = next_approx
     DWT_FILTERED = saturated(LLs[0])

     if save_imgs:
        cv2.imwrite(os.path.join(img_dir,fn), DWT_FILTERED)
     else: 
        filtered_img = image_utils.apply_cfar(DWT_FILTERED[None,:,:]/255)[0]
        ax[0][1].title.set_text("normalized + DWT")
        ax[0][1].imshow(DWT_FILTERED, cmap='gray', vmin=0, vmax = 255)
        ax[1][0].title.set_text("normalized + CFAR")
        ax[1][0].imshow(image_utils.apply_cfar(input_img[None,:,:]/255)[0], cmap='gray', vmin=0, vmax = 1)
        ax[1][1].title.set_text("normalized + DWT + CFAR")
        ax[1][1].imshow(filtered_img, cmap='gray', vmin=0, vmax = 1)
        plt.tight_layout()
        plt.show()