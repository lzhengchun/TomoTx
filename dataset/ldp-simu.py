#! /homes/zhengchun.liu/usr/miniconda3/envs/torch/bin/python

import glob, h5py, skimage.transform
import numpy as np
from multiprocessing import Pool
import multiprocessing
from random import shuffle

def scale2uint16(d_img):
    _min, _max = d_img.min(), d_img.max()
    if _max == _min:
        r_img = d_img[:] - _max
    else:
        r_img = (d_img[:] - _min) * 65535. / (_max - _min)
    return r_img.astype(np.uint16)

def tomo_sim(image):
    nproj = 180
    theta = np.linspace(0., 180., nproj, endpoint=False)
    
    # pad = round(image.shape[0] - image.shape[0] / np.sqrt(2))
    # image = np.pad(image, pad_width=((pad, pad), (pad, pad)))

    sino  = skimage.transform.radon(image, theta=theta, circle=True)

    return scale2uint16(sino.T)

def pr_proc(h5s, ):
    cmb_imgs = []
    for h5 in h5s:
        _s = h5py.File(h5, 'r')['data'][:]
        cmb_imgs.append(_s)
    return np.concatenate(cmb_imgs, axis=0)

if __name__ == '__main__':
    DDIR = '/lambda_stor/data/zliu/LoDoPaB-CT'
    img_h5s_train = glob.glob(f"{DDIR}/train/image/*.hdf5")
    img_h5s_test  = glob.glob(f"{DDIR}/test/image/*.hdf5")
    h5s = img_h5s_train + img_h5s_test
    shuffle(h5s)

    images = pr_proc(h5s[:])
    pad = 107 # explicitly pad to 576 = 512 + 64 to ease patch process
    images = np.pad(images, pad_width=((0, 0), (pad, pad), (pad, pad)))

    np.random.shuffle(images) # shuffle for random train/validation split
    nproj = 180
    ntrain = int(0.9 * images.shape[0])
    
    with Pool(multiprocessing.cpu_count()) as p:
        simu_res = p.map(tomo_sim, images)

        with h5py.File("ldp-train.h5", "w") as fd:
            sino_ds = fd.create_dataset("sino",  data=simu_res[:ntrain], dtype=np.uint16)
            sino_ds.attrs['std']  = np.std(simu_res[:ntrain])
            sino_ds.attrs['mean'] = np.mean(simu_res[:ntrain])

            img_ds  = fd.create_dataset("image", data=images[:ntrain], dtype=images.dtype)
            img_ds.attrs['std']  = np.std(images[:ntrain])
            img_ds.attrs['mean'] = np.mean(images[:ntrain])

        with h5py.File("ldp-valid.h5", "w") as fd:
            sino_ds = fd.create_dataset("sino",  data=simu_res[ntrain:], dtype=np.uint16)
            sino_ds.attrs['std']  = np.std(simu_res[ntrain:])
            sino_ds.attrs['mean'] = np.mean(simu_res[ntrain:])

            img_ds  = fd.create_dataset("image", data=images[ntrain:], dtype=images.dtype)
            img_ds.attrs['std']  = np.std(images[ntrain:])
            img_ds.attrs['mean'] = np.mean(images[ntrain:])                  

