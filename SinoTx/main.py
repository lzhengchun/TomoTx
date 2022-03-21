#! /homes/zhengchun.liu/usr/miniconda3/envs/torch/bin/python

from model import SinoTx
import torch, argparse, os, time, sys, shutil, yaml
from util import save2img
from data import SinogramDataset
import numpy as np
import logging
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='TOMOiT')
parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-cfg',    type=str, required=True, help='path to config yaml file')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def main(args):
    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)

    logging.info("Init training dataset ...")
    train_ds = SinogramDataset(ifn=params['dataset']['th5'], params=params)
    train_dl = DataLoader(train_ds, batch_size=params['train']['mbsz'], num_workers=4, \
                          prefetch_factor=params['train']['mbsz'], pin_memory=True, \
                          drop_last=True, shuffle=True)
    logging.info("%d samples will be used for training" % (len(train_ds), ))

    logging.info("Init validation dataset ...")
    valid_ds = SinogramDataset(ifn=params['dataset']['vh5'], params=params)
    valid_dl = DataLoader(valid_ds, batch_size=params['train']['mbsz'], num_workers=4, \
                          prefetch_factor=params['train']['mbsz'], pin_memory=True, \
                          drop_last=False, shuffle=True)
    logging.info("%d samples will be used for validation" % (len(valid_ds), ))

    model = SinoTx(seqlen=train_ds.seqlen, in_dim=train_ds.cdim, params=params)

    if torch.cuda.is_available():
        model = model.to(torch_dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['train']['lr'], betas=(0.9, 0.95))

    logging.info("Start training ...")
    for ep in range(1, params['train']['maxep']+1):
        train_ep_tick = time.time()
        for imgs_tr in train_dl: 
            optimizer.zero_grad()
            loss, pred, mask = model.forward(imgs_tr.to(torch_dev))
            loss.backward()
            optimizer.step()
 
        time_e2e = time.time() - train_ep_tick
        itr_prints = '[Train] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch' % (ep, loss.cpu().detach().numpy(), time_e2e, )
        logging.info(itr_prints)

        val_loss = []
        valid_ep_tick = time.time()
        for imgs_val in valid_dl:
            with torch.no_grad():
                _vloss, _vpred, _vmask = model.forward(imgs_val.to(torch_dev))
                val_loss.append(_vloss.cpu().numpy())

        valid_e2e = time.time() - valid_ep_tick
        _prints = '[Valid] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch\n' % (ep, np.mean(val_loss), valid_e2e)
        logging.info(_prints)

        if ep % params['train']['ckp_steps'] != 0: continue
        save2img(imgs_val[-1].numpy().squeeze(),     '%s/ep%05d-valid-gt.tiff' % (ep, itr_out_dir))
        save2img(_vpred[-1].cpu().numpy().squeeze(), '%s/ep%05d-valid-pd.tiff' % (ep, itr_out_dir))
        
        save2img(imgs_tr[-1].numpy().squeeze(),             '%s/ep%05d-train-gt.tiff' % (ep, itr_out_dir))
        save2img(pred[-1].detach().cpu().numpy().squeeze(), '%s/ep%05d-train-pd.tiff' % (ep, itr_out_dir))

        torch.save(model.state_dict(), "%s/mdl-ep%05d.pth" % (itr_out_dir, ep))
        with open(f'{itr_out_dir}/config.yaml', 'w') as fp:
            yaml.dump(params, fp)

if __name__ == "__main__":
    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    logging.basicConfig(filename="%s/SinoTx.log" % (itr_out_dir, ), level=logging.DEBUG)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)
