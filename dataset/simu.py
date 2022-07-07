import skimage.draw, h5py, skimage.transform
import numpy as np
from multiprocessing import Pool
import multiprocessing

def scale2uint16(d_img):
    _min, _max = d_img.min(), d_img.max()
    if _max == _min:
        r_img = d_img[:] - _max
    else:
        r_img = (d_img[:] - _min) * 65535. / (_max - _min)
    return r_img.astype(np.uint16)

def tomo_sim_rnd(args):
    (h, w), nproj = args
    image = skimage.draw.random_shapes(image_shape=(h, w), max_shapes=30, min_shapes=10, \
                                       min_size=w//5, max_size=w//2, intensity_range=(80, 200),\
                                       multichannel=False, allow_overlap=False)[0]
    rr, cc = skimage.draw.disk((h/2, w/2), w/2, )
    mask   = np.zeros((h, w), dtype=np.uint8)
    mask[rr, cc] = 1
    image *= mask

    theta = np.linspace(0., 180., nproj, endpoint=False)
    sino  = skimage.transform.radon(image, theta=theta, circle=True)

    return image, scale2uint16(sino.T)

if __name__ == '__main__':
    nproj, img_size = 180, 256

    n_ds = 100000
    with Pool(multiprocessing.cpu_count()//2) as p:
        simu_res = p.map(tomo_sim_rnd, [((img_size, img_size), nproj),]*n_ds)

        with h5py.File("ds-simu-train.h5", "w") as fd:
            fd.create_dataset("sino",  data=[x[1] for x in simu_res], dtype=np.uint16)
            fd.create_dataset("image", data=[x[0] for x in simu_res], dtype=np.uint16)
            
    n_ds = 2000
    with Pool(multiprocessing.cpu_count()//2) as p:
        simu_res = p.map(tomo_sim_rnd, [((img_size, img_size), nproj),]*n_ds)

        with h5py.File("ds-simu-valid.h5", "w") as fd:
            fd.create_dataset("sino",  data=[x[1] for x in simu_res], dtype=np.uint16)
            fd.create_dataset("image", data=[x[0] for x in simu_res], dtype=np.uint16)