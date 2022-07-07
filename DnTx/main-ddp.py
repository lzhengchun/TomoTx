
# torchrun --standalone --nnodes=1 --nproc_per_node=8  ./main-ddp.py -cfg=config/simu.yaml -expName=simu

from model import DnTx
import torch, argparse, os, time, sys, shutil, yaml
from util import save2img, restore_and_save2tiff
from data import SinogramDataset
import numpy as np
import logging
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='DnTx')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-cfg',    type=str, required=True, help='path to config yaml file')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def main(args):
    # env and log init
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_nccl_available():
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)

    itr_out_dir = args.expName + '-itrOut'
    if rank == 0:
        if os.path.isdir(itr_out_dir):
            shutil.rmtree(itr_out_dir)
        os.mkdir(itr_out_dir) # to save temp output
    torch.distributed.barrier()

    logging.basicConfig(filename=f"{args.expName}-itrOut/DnTx.log", level=logging.DEBUG,\
                        format='%(asctime)s %(levelname)s %(module)s: %(message)s')
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # training task init
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)
    logging.info(f"local rank {local_rank} (global rank {rank}) of a world size {world_size} started")

    torch.cuda.set_device(local_rank)

    logging.info(f"rank {rank} Init training dataset ...")
    train_ds = SinogramDataset(ifn=params['dataset']['th5'], params=params, world_size=world_size, rank=rank)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_dl = DataLoader(train_ds, batch_size=params['train']['mbsz']//world_size, num_workers=4, \
                          prefetch_factor=params['train']['mbsz'], pin_memory=True, \
                          drop_last=True, shuffle=True)
    logging.info(f"rank {rank} %d samples, {train_ds.shape}, will be used for training" % (len(train_ds), ))

    if rank == world_size-1: 
        logging.info("Init validation dataset ...")
        valid_ds = SinogramDataset(ifn=params['dataset']['vh5'], params=params)
        valid_dl = DataLoader(valid_ds, batch_size=params['train']['mbsz']//world_size, num_workers=4, \
                              prefetch_factor=params['train']['mbsz'], pin_memory=True, \
                              drop_last=False, shuffle=False)
        logging.info(f"%d samples, {valid_ds.shape},  will be used for validation" % (len(valid_ds), ))

    model = DnTx(seqlen=train_ds.seqlen, in_dim=train_ds.cdim, params=params).cuda()

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['train']['lr'], betas=(0.9, 0.95))

    logging.info(f"rank {rank} Start training ...")
    for ep in range(1, params['train']['maxep']+1):
        train_ep_tick = time.time()
        for sinos_ld, sinos_nd in train_dl: 
            optimizer.zero_grad()
            pred = model.forward(sinos_ld.cuda())
            loss = criterion(pred, sinos_nd.cuda())
            loss.backward()
            optimizer.step()

        if rank != world_size-1: continue

        time_e2e = time.time() - train_ep_tick
        itr_prints = '[Train] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch, %d steps with lbs=%d' % (\
                     ep, loss.cpu().detach().numpy(), time_e2e, len(train_dl), sinos_nd.shape[0])
        logging.info(itr_prints)

        val_loss = []
        valid_ep_tick = time.time()
        for vsinos_ld, vsinos_nd in valid_dl:
            with torch.no_grad():
                _vpred = model.forward(vsinos_ld.cuda())
                _vloss = torch.nn.functional.mse_loss(_vpred, vsinos_nd.cuda())
                val_loss.append(_vloss.cpu().numpy())

        valid_e2e = time.time() - valid_ep_tick
        _prints = '[Valid] Epoch %3d, loss: %.6f, elapse: %.2fs/epoch\n' % (ep, np.mean(val_loss), valid_e2e)
        logging.info(_prints)

        save2img(vsinos_nd[-1].numpy().squeeze(),   '%s/ep%05d-valid-nd.tiff' % (itr_out_dir, ep))
        save2img(_vpred[-1].cpu().numpy().squeeze(),'%s/ep%05d-valid-pd.tiff' % (itr_out_dir, ep))
        save2img(vsinos_ld[-1].numpy().squeeze(),   '%s/ep%05d-valid-ld.tiff' % (itr_out_dir, ep))

        if ep % params['train']['ckp_steps'] != 0: continue
        torch.save(model.module.state_dict(), "%s/mdl-ep%05d.pth" % (itr_out_dir, ep))
        with open(f'{itr_out_dir}/config.yaml', 'w') as fp:
            yaml.dump(params, fp)

if __name__ == "__main__":
    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    if len(args.gpus) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
