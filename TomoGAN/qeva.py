#! /homes/zhengchun.liu/usr/miniconda3/envs/torch/bin/python

import logging, h5py, skimage.transform, yaml, torch
import sys, skimage.metrics, argparse, glob
import numpy as np
import pandas as pd
from data import DnDataset
# from model import unet

def denorm(a, mean, std):
    return a * std  + mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TomoGAN QEva')
    parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
    parser.add_argument('-dir',    type=str, required=True, help='path to checkpoint dir')
    parser.add_argument('-ckpt',   type=str, default="", help='model checkpoint')
    parser.add_argument('-mr',     type=float,default=0.8, help='mask ratio')
    parser.add_argument('-mbsz',   type=int,default=16, help='batch size')
    parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    logging.basicConfig(filename=f"TomoGAN-qeva.log", level=logging.DEBUG)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = yaml.load(open(f'{args.dir}/config.yaml', 'r'), Loader=yaml.CLoader)
    params['train']['psz'] = 256

    vh5 = params['dataset']['vh5'][:-6] + f"{args.mr}".replace('.', 'p') + '.h5'

    valid_ds = DnDataset(ifn=vh5, params=params)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.mbsz, drop_last=False, shuffle=False)

    if len(args.ckpt) > 0:
        ckpt = args.ckpt
    else:
        ckpt = sorted(glob.glob(f"{args.dir}/*.pth"))[-1]
        logging.info(f"weights from {ckpt} will be used")
    model = torch.jit.load(ckpt, map_location=torch.device('cpu'))
    for key, p in model.named_parameters(): p.requires_grad = False
    model = model.to(torch_dev)

    ssim_sv, ssim_pd, psnr_sv, psnr_pd = [], [], [], []
    for imgs_sv, imgs_fv in valid_dl:
        imgs_dn = model.forward(imgs_sv.to(torch_dev)).cpu().numpy()[:,0]

        imgs_sv = denorm(imgs_sv.numpy()[:,0], std=params['dataset']['std4norm'], \
                        mean=params['dataset']['mean4norm'])
        imgs_fv = imgs_fv.numpy()[:,0]
        dr = imgs_fv.max() - imgs_fv.min()
        for n in range(imgs_sv.shape[0]):
            ssim_sv.append(skimage.metrics.structural_similarity(imgs_fv[n], imgs_sv[n], data_range=dr))
            ssim_pd.append(skimage.metrics.structural_similarity(imgs_fv[n], imgs_dn[n], data_range=dr))

            psnr_sv.append(skimage.metrics.peak_signal_noise_ratio(imgs_fv[n], imgs_sv[n], data_range=dr))
            psnr_pd.append(skimage.metrics.peak_signal_noise_ratio(imgs_fv[n], imgs_dn[n], data_range=dr))

    pd.DataFrame(np.vstack([ssim_sv, ssim_pd, psnr_sv, psnr_pd]).T, \
                 columns=['ssim_sv', 'ssim_pd', 'psnr_sv', 'psnr_pd']).to_csv(f'{args.dir}/q-{args.mr}.csv', index=False)