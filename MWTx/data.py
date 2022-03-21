import numpy as np 
import h5py, logging
from torch.utils.data import Dataset

class SinogramDataset(Dataset):
    def __init__(self, ifn, params, world_size=None, rank=None):
        ds_cfg = params['dataset']

        h5fd = h5py.File(ifn, 'r')
        n_samples = h5fd[ds_cfg['dkey']].shape[0]

        if world_size is None or rank is None:
            si, ei = 0, None
        else:
            npr = n_samples // world_size
            si = rank * npr
            ei = (rank + 1) * npr
        logging.info(f"rank {rank} loading data from index {si} to {ei}")
        self.targets  = h5fd[ds_cfg['dkey']][si:ei]
        h5fd.close()

        # norm or not will influence pos-encodding
        if ds_cfg['norm']:
            if ds_cfg.get('mean4norm') is None:
                _avg = self.targets.mean()
                _std = self.targets.std()
                self.targets = ((self.targets - _avg) / _std).astype(np.float32)
                logging.info(f'rank {rank} features are normalized with computed mean: {_avg}, std: {_std}')
            else:
                self.targets = ((self.targets - ds_cfg['mean4norm']) / ds_cfg['std4norm']).astype(np.float32)
                logging.info(f"rank {rank} features are normalized with provided mean: {ds_cfg['mean4norm']}, std: {ds_cfg['std4norm']}")

        self.len    = self.targets.shape[0]
        self.seqlen = self.targets.shape[-2] # angle
        self.cdim   = self.targets.shape[-1] # resolution
        self.shape  = self.targets.shape

    def __getitem__(self, idx):
            return self.targets[idx][None].astype(np.float32)

    def __len__(self):
        return self.len