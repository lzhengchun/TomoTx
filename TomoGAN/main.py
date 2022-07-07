#! /homes/zhengchun.liu/usr/miniconda3/envs/torch/bin/python

from model import unet
import torch, argparse, os, time, sys, shutil, logging, yaml
from data import DnDataset
import numpy as np
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='denoise with self-supervised learning')
parser.add_argument('-gpus',   type=str, default="1", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-cfg',    type=str, required=True,  help='global config file')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def main(args, logdir):
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)
    logging.info("[%.3f] loading data into CPU memory, it will take a while ... ..." % (time.time(), ))
    ds_train = DnDataset(ifn=params['dataset']['th5'], params=params)
    dl_train = DataLoader(dataset=ds_train, batch_size=params['train']['mbsz'], shuffle=True,\
                          num_workers=4, prefetch_factor=params['train']['mbsz'], drop_last=True, pin_memory=True)
    logging.info(f"loaded %d samples, {ds_train.dim}, into CPU memory for training." % (len(ds_train), ))

    ds_valid = DnDataset(ifn=params['dataset']['vh5'], params=params)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=params['train']['mbsz'], shuffle=True, \
                          num_workers=8, prefetch_factor=params['train']['mbsz'], drop_last=False, pin_memory=True)
    logging.info(f"loaded %d samples, {ds_valid.dim}, into CPU memory for validation." % (len(ds_valid), ))

    torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet().to(torch_devs)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['lr'])

    for epoch in range(1, 1+params['train']['maxep']):
        ep_tick = time.time()
        for X_mb, Y_mb in dl_train:
            X_mb_dev = X_mb.to(torch_devs)
            optimizer.zero_grad()
            pred = model.forward(X_mb_dev)
            loss = criterion(pred, Y_mb.to(torch_devs))
            loss.backward()
            optimizer.step() 

        time_e2e   = time.time() - ep_tick

        val_loss = []
        for X_mb, Y_mb in dl_valid:
            with torch.no_grad():
                pred  = model.forward(X_mb.to(torch_devs))
                _loss = torch.nn.functional.mse_loss(pred, Y_mb.to(torch_devs))
                val_loss.append(_loss.cpu().numpy())

        itr_prints = '[Info] @ %.1f Epoch: %05d, train loss: %.7f, validation loss: %.7f, elapse: %.2fs/ep' % (\
                    time.time(), epoch, loss.cpu().detach().numpy(), np.mean(val_loss), time_e2e, )
        logging.info(itr_prints)

        if epoch % params['train']['ckp_steps'] != 0: continue
        torch.jit.save(torch.jit.trace(model, X_mb_dev[:1]), "%s/script-ep%05d.pth" % (logdir, epoch))
        with open(f'{logdir}/config.yaml', 'w') as fp:
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

    logging.basicConfig(filename=os.path.join(itr_out_dir, 'TomoGAN.log'), level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)s %(module)s: %(message)s')
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        
    main(args, logdir=itr_out_dir)
