import os
import argparse
import numpy as np
import shutil
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset merging tool for grid datasets.')
    parser.add_argument('-d','--datasets', 
                        help='Input directory containing all datasets to be merged', 
                        required=True)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    all_datasets = [d for d in os.listdir(args.datasets) if os.path.isdir(os.path.join(args.datasets,d)) and d != 'merged']

    all_images_fn = []
    all_poses_fn = []

    for dataset in all_datasets:
        print('dataset name:', dataset)
        dataset_path = os.path.join(args.datasets, dataset)
        imgs_path = os.path.join(dataset_path,'data','imgs')
        poses_path = os.path.join(dataset_path,'data','poses')
        imgs_fn = np.array([f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))])
        poses_fn = np.array([f for f in os.listdir(poses_path) if os.path.isfile(os.path.join(poses_path, f))])
        print('    ','#imgs :', len(imgs_fn))
        print('    ','#poses:', len(poses_fn))
        imgs_indices = np.array([f.split('.')[0] for f in imgs_fn]).astype(np.int32)
        poses_indices = np.array([f.split('.')[0] for f in poses_fn]).astype(np.int32)
        imgs_sorting = np.argsort(imgs_indices)
        poses_sorting = np.argsort(poses_indices)
        imgs_fn = imgs_fn[imgs_sorting]
        poses_fn = poses_fn[poses_sorting]
        all_images_fn.extend([os.path.join(imgs_path, f) for f in imgs_fn])
        all_poses_fn.extend([os.path.join(poses_path, f) for f in poses_fn])

    merged_path = os.path.join(args.datasets, 'merged')
    merged_imgs_path = os.path.join(merged_path,'data','imgs')
    merged_poses_path = os.path.join(merged_path,'data','poses')
    print('Merged dataset:',merged_path)
    print('    ','#imgs :',len(all_images_fn))
    print('    ','#poses:',len(all_poses_fn))


    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)

    os.mkdir(merged_path)
    os.mkdir(os.path.join(merged_path, 'data'))
    os.mkdir(merged_imgs_path)
    os.mkdir(merged_poses_path)


    print('copying files ...')
    for i ,(img_fn, pose_fn) in tqdm.tqdm(enumerate(zip(all_images_fn, all_poses_fn)), total=len(all_images_fn)):
        new_img_fn = os.path.join(merged_imgs_path,str(i)+'.jpeg')
        new_pose_fn = os.path.join(merged_poses_path,str(i)+'.txt')
        shutil.copy(img_fn, new_img_fn)
        shutil.copy(pose_fn, new_pose_fn)
