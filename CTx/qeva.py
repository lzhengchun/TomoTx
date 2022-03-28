#! /homes/zhengchun.liu/usr/miniconda3/envs/torch/bin/python

import logging, h5py, skimage.transform, yaml, torch
import sys, skimage.metrics, argparse, glob
import numpy as np
import pandas as pd
from data import SinogramDataset
from model import CTx

def denorm(a, mean, std):
    return a * std  + mean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SinoTx')
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

    logging.basicConfig(filename=f"CTx-qeva.log", level=logging.DEBUG)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = yaml.load(open(f'{args.dir}/config.yaml', 'r'), Loader=yaml.CLoader)
    valid_ds = SinogramDataset(ifn=params['dataset']['vh5'], params=params)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.mbsz, drop_last=False, shuffle=False)

    model = CTx(in_seqlen=valid_ds.in_seqlen, in_dim=valid_ds.in_cdim, params=params)
    if len(args.ckpt) > 0:
        ckpt = args.ckpt
    else:
        ckpt = sorted(glob.glob(f"{args.dir}/*.pth"))[-1]
        logging.info(f"weigghts from {ckpt} will be used")
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
    model = model.eval()
    for key, p in model.named_parameters(): p.requires_grad = False
    model = model.to(torch_dev)

    theta = np.linspace(0., 180., valid_ds.in_shape[-2], endpoint=False)
    ssim_sv, ssim_pd, psnr_sv, psnr_pd = [], [], [], []
    for sinos, imags in valid_dl:
        _vpred, _vmask = model.forward(sinos.to(torch_dev), mask_ratio=args.mr)
        _vpred = denorm(_vpred.cpu().numpy()[:,0], std=params['dataset']['onorm']['std4norm'], \
                        mean=params['dataset']['onorm']['mean4norm'])
        _vmask = _vmask.cpu().numpy()
        _imags = denorm(imags.numpy()[:,0], std=params['dataset']['onorm']['std4norm'], \
                        mean=params['dataset']['onorm']['mean4norm'])
        sinos  = denorm(sinos[:,0].numpy(), std=params['dataset']['inorm']['std4norm'], \
                        mean=params['dataset']['inorm']['mean4norm'])
        for n in range(_vmask.shape[0]):
            recon_sv = skimage.transform.iradon(sinos[n][_vmask[n]==0].T, \
                                                theta=theta[_vmask[n]==0])
            np.save('gt', _imags[n])
            np.save('sv', recon_sv)
            np.save('pd', _vpred[n])
            ssim_sv.append(skimage.metrics.structural_similarity(_imags[n], recon_sv,  data_range=255))
            ssim_pd.append(skimage.metrics.structural_similarity(_imags[n], _vpred[n], data_range=255))

            psnr_sv.append(skimage.metrics.peak_signal_noise_ratio(_imags[n], recon_sv,  data_range=255))
            psnr_pd.append(skimage.metrics.peak_signal_noise_ratio(_imags[n], _vpred[n], data_range=255))

    pd.DataFrame(np.vstack([ssim_sv, ssim_pd, psnr_sv, psnr_pd]).T, \
                 columns=['ssim_sv', 'ssim_pd', 'psnr_sv', 'psnr_pd']).to_csv(f'{args.dir}/q-{args.mr}.csv', index=False)